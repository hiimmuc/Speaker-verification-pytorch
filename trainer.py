#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import glob
import subprocess
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from callbacks.earlyStopping import *
from dataloader import train_data_loader
from model import SpeakerEncoder, WrappedModel, ModelHandling
from utils import tuneThresholdfromScore, read_log_file, plot_from_file, cprint

from torch.utils.tensorboard import SummaryWriter
###

# Try to import NSML
try:
    import nsml
    from nsml import HAS_DATASET, DATASET_PATH, PARALLEL_WORLD, PARALLEL_PORTS, MY_RANK
    from nsml import NSML_NFS_OUTPUT, SESSION_NAME
except:
    pass

warnings.simplefilter("ignore")


def main_worker(gpu, nprocs, args):
    """
    https://pytorch.org/docs/stable/_modules/torch/multiprocessing/spawn.html#spawn
    """
    
    args.gpu = gpu  # if gpu == 0 means the main process
    device = torch.device(
        f'{args.device}:{args.gpu}') if args.device == 'cuda' else torch.device('cpu')
    ngpus_per_node = nprocs
    print("Use GPU cuda:{} for training".format(args.gpu))

    # paths
    args.model_save_path = os.path.join(
        args.save_folder, f"{args.model['name']}/{args.criterion['name']}/model")

    args.result_save_path = os.path.join(
        args.save_folder, f"{args.model['name']}/{args.criterion['name']}/result")

    # TensorBoard
    os.makedirs(f"{args.result_save_path}/runs", exist_ok=True)
    if args.gpu == 0:
        writer = SummaryWriter(log_dir=f"{args.result_save_path}/runs")

    # init parameters
    epoch = 1
    min_loss = float("inf")
    min_eer = float("inf")

    # Initialise data loader
    args.dataloader_options['num_workers'] = int(
        args.dataloader_options['num_workers'] / ngpus_per_node)
    train_loader = train_data_loader(args)
    max_iter_size = len(train_loader) // (args.dataloader_options['nPerSpeaker'] * ngpus_per_node) 

    # define net
    s = SpeakerEncoder(**vars(args))

    # NOTE: Data parallelism for multi-gpu in BETA
    # init parallelism create net-> load weight -> add to parallelism
    if args.data_parallel:
        if torch.cuda.device_count() > 1:
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            s = nn.DataParallel(
                s, device_ids=[i for i in range(torch.cuda.device_count())])
            # device = 'cuda'  # to the primary gpu
            s = s.to(device)
        else:
            s = WrappedModel(s).to(device)

    # NOTE: setup distributed data parallelism training
    if args.distributed:
        setup_DDP(rank=args.gpu,
                  world_size=ngpus_per_node,
                  backend=args.distributed_backend,
                  port=args.port)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(
            s, device_ids=[args.gpu], find_unused_parameters=False)

    else:
        s = WrappedModel(s).to(device)

    speaker_model = ModelHandling(s, **dict(vars(args), T_max=max_iter_size))

    # Choose weight as pretrained model
    weight_path, start_lr, init_epoch = choose_model_state(args, priority='previous')
    if weight_path is not None:
        if args.gpu == 0:
            print("Load model from:", weight_path)
        speaker_model.loadParameters(weight_path)
    else:
        if args.gpu == 0:
            print("Train model from scratch!")

    # define early stop
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.es_patience)

    top_count = 1

    # Write args to score_file
    score_file_path = os.path.join(args.result_save_path, 'scores.txt')
    if args.gpu == 0:
        score_file = open(score_file_path, "a+")

    # Training loop
    timer = time.time()
    for epoch in range(int(init_epoch), int(args.number_of_epochs + 1)):
        clr = [x['lr'] for x in speaker_model.__optimizer__.param_groups]
        if args.gpu == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), epoch,
                  "[INFO] Training %s with LR %f ---" % (args.model['name'], max(clr)))

        # Train network
        loss, train_acc = speaker_model.fit(
            loader=train_loader, epoch=epoch, verbose=(args.gpu == 0))

        if args.gpu == 0:
            # save best model
            if loss == min(min_loss, loss):
                cprint(
                    text=f"[INFO] Loss reduce from {min_loss} to {loss}. Save the best state", fg='y')
                speaker_model.saveParameters(
                    args.model_save_path + "/best_state.pt")

                speaker_model.saveParameters(args.model_save_path +
                                             f"/best_state_top{top_count}.pt")
                # to save top 3 of best_state
                top_count = (top_count + 1) if top_count <= 3 else 1
                if args.early_stopping:
                    early_stopping.counter = 0  # reset counter of early stopping

            min_loss = min(min_loss, loss)

            # Validate each interval or save only
            if args.test_interval > 0 and epoch % args.test_interval == 0:
                sc, lab, _ = speaker_model.evaluateFromList(args.valid_annotation,
                                                            distributed=False,
                                                            dataloader_options=args.dataloader_options,
                                                            num_eval=args.num_eval,
                                                            cohorts_path=None)
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])['roc']

                min_eer = min(min_eer, result[1])

                print("[INFO] Evaluating ",
                      time.strftime("%H:%M:%S"),
                      "LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f" %
                      (max(clr), train_acc, loss, result[1], min_eer))

                score_file.write(
                    "epoch %d, LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f\n"
                    % (epoch, max(clr), train_acc, loss, result[1], min_eer))

                with open(os.path.join(args.model_save_path, "model_state_log.txt"), 'w+') as log_file:
                    log_file.write(f"Epoch:{epoch}, LR:{max(clr)}, EER: {0}")

                score_file.flush()

                plot_from_file(args.result_save_path, show=False)

            else:
                # test interval < 0 -> train non stop
                score_file.write("epoch %d, LR %f, TEER/TAcc %2.2f, TLOSS %f\n" %
                                 (epoch, max(clr), train_acc, loss))

                with open(os.path.join(args.model_save_path, "model_state_log.txt"), 'w+') as log_file:
                    log_file.write(f"Epoch:{epoch}, LR:{max(clr)}, EER: {0}")

                score_file.flush()

                plot_from_file(args.result_save_path, show=False)

            # NOTE: consider save last state only or not, save only eer as the checkpoint for iterations
            if args.save_model_last:
                speaker_model.saveParameters(
                    args.model_save_path + "/last_state.pt")
            else:
                speaker_model.saveParameters(
                    args.model_save_path + "/model_state_%06d.pt" % epoch)

            if ("nsml" in sys.modules):
                training_report = {}
                training_report["summary"] = True
                training_report["epoch"] = epoch
                training_report["step"] = epoch
                training_report["train_loss"] = loss

                nsml.report(**training_report)

            if args.early_stopping:
                early_stopping(loss)
                if early_stopping.early_stop:
                    score_file.close()
                    writer.close()
                    break
            if args.ckpt_interval_minutes > 0:
                if ((time.time() - timer) // 60) % args.ckpt_interval_minutes == 0:
                    # save every N mins and keep only top 3
                    current_time = str(time.strftime("%Y%m%d_%H_%M"))
                    ckpt_list = glob.glob(os.path.join(
                        args.model_save_path , '/ckpt_*'))
                    if len(ckpt_list) == 3:
                        ckpt_list.sort()
                        subprocess.call(f'rm -f {ckpt_list[-1]}', shell=True)
                    speaker_model.saveParameters(
                        args.model_save_path + f"/ckpt_{current_time}.pt")

            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Params/learning_rate', max(clr), epoch)

    if args.gpu == 0:
        score_file.close()
        writer.close()
    if args.distributed: 
        cleanup_DDP()
    sys.exit(1)

#######################################
# main fucntion
######################################


def train(args):
    # For REPRODUCIBILITY purpose
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Training params saved
    with open(os.path.join(args.save_folder, f"{args.model['name']}/{args.criterion['name']}/result/settings.txt"), 'a+') as settings_file:
        settings_file.write(
            f'\n[TRAIN]------------------{time.strftime("%Y-%m-%d %H:%M:%S")}------------------\n')
        for items in vars(args):
            settings_file.write('%s %s\n' % (items, vars(args)[items]))
        settings_file.flush()

    # Main run
    print("Data Parallelism training:", args.data_parallel)
    print("Distributed Data Parallelism:", args.distributed)
    try:
        try:
            if args.distributed:
                npugs = torch.cuda.device_count()
                mp.spawn(main_worker, nprocs=npugs, args=(npugs, args))
            else:
                main_worker(0, 1, args)
                
        except Exception as e:
            print(f"Got error: {e} -> try to turn to single-GPU training")
            args.distributed = False
            args.data_parallel = False
            main_worker(0, 1, args)
            
    except KeyboardInterrupt:
        print('Interrupted')
        try: 
            dist.destroy_process_group()  
        except KeyboardInterrupt: 
            os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')")

    sys.exit(1)
    
    
def setup_DDP(rank, world_size, backend='nccl', address='localhost', port='123455'):
    os.environ['MASTER_ADDR'] = address
    os.environ['MASTER_PORT'] = port
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_DDP():
    dist.destroy_process_group()


def choose_model_state(args, priority='defined'):
    if args.gpu == 0:
        print(f"Using pretrained: {args.pretrained['use']}")
    # load state from log file
    if os.path.isfile(os.path.join(args.model_save_path, "model_state_log.txt")):
        start_it, start_lr, _ = read_log_file(
            os.path.join(args.model_save_path, "model_state_log.txt"))
    else:
        start_it = 1
        start_lr = args.lr

    # Load model weights
    model_files = glob.glob(os.path.join(
        args.model_save_path, 'model_state_*.pt'))
    model_files.sort()

    # if exists best model load from epoch and load state from log model file
    prev_model_state = None
    if start_it > 1:
        if os.path.exists(f'{args.model_save_path}/best_state.pt'):
            prev_model_state = f'{args.model_save_path}/best_state.pt'
        elif args.save_model_last:
            if os.path.exists(f'{args.model_save_path}/last_state.pt'):
                prev_model_state = f'{args.model_save_path}/last_state.pt'
        else:
            prev_model_state = model_files[-1]

        if args.number_of_epochs > start_it:
            epoch = int(start_it)
        else:
            epoch = 1

    # NOTE: Priority: defined pretrained > previous state from logger > scratch
    if args.pretrained['use'] and priority == 'defined':
        choosen_state = args.pretrained['path']
        epoch = 1
        args.lr = start_lr

    elif args.pretrained['use'] and prev_model_state and priority== 'previous':
        choosen_state = prev_model_state
        args.lr = start_lr
        epoch = start_it
        
    else:
        epoch = 1
        start_lr = args.lr
        choosen_state = None  # "Train model from scratch!"

    return choosen_state, start_lr, epoch
# ============================ END =============================