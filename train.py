import glob
import os
import sys
import time

from callbacks.earlyStopping import *
from dataloader import get_data_loader
from model import SpeakerNet
from plot_loss import *
from utils import tuneThresholdfromScore


def train(args):
    # Initialise directories
    model_save_path = args.save_path + f"/{args.model}/model"
    result_save_path = args.save_path + f"/{args.model}/result"

    # Load models
    s = SpeakerNet(args, **vars(args))

    it = 1
    min_loss = float("inf")
    min_eer = float("inf")

    # Load model weights
    model_files = glob.glob(os.path.join(model_save_path, 'model_state_*.model'))
    model_files.sort()

    eerfile = glob.glob(os.path.join(model_save_path, 'model_state_*.eer'))
    eerfile.sort()

    # if exists best model load from it
    prev_model_state = None
    if len(eerfile) > 0:
        if os.path.exists(f'{model_save_path}/best_state.model'):
            prev_model_state = f'{model_save_path}/best_state.model'
        elif args.save_model_last:
            prev_model_state = f'{model_save_path}/last_state.model'
        else:
            prev_model_state = model_files[-1]

        # get the last stopped iteration, model_state_xxxxxx.eer, so 12 is index of number sequence
        start_it = int(os.path.splitext(
            os.path.basename(eerfile[-1]))[0][12:]) + 1

        if args.max_epoch > start_it:
            it = start_it
        else:
            it = 1

    if args.initial_model:
        s.loadParameters(args.initial_model)
        print("Model %s loaded!" % args.initial_model)
        if 'checkpoints' in args.initial_model:
            it = 1
    elif prev_model_state:
        s.loadParameters(prev_model_state)
        print("Model %s loaded from previous state!" % prev_model_state)
    else:
        print("Train model from scratch!")

    # schedule the learning rate to stopped epoch
    for _ in range(0, it - 1):
        s.__scheduler__.step()

    # Write args to score_file
    settings_file = open(result_save_path + '/settings.txt', 'a+')
    score_file = open(result_save_path + "/scores.txt", "a+")
    # summary settings
    settings_file.write(
        f'\n[TRAIN]------------------{time.strftime("%Y-%m-%d %H:%M:%S")}------------------\n')
    score_file.write(
        f'\n[TRAIN]------------------{time.strftime("%Y-%m-%d %H:%M:%S")}------------------\n')
    # write the settings to settings file
    for items in vars(args):
        # print(items, vars(args)[items])
        settings_file.write('%s %s\n' % (items, vars(args)[items]))
    settings_file.flush()

    # Initialise data loader
    train_loader = get_data_loader(args.train_list, **vars(args))

    if args.early_stop:
        early_stopping = EarlyStopping(patience=args.es_patience)

    # Training loop
    while True:
        clr = [x['lr'] for x in s.__optimizer__.param_groups]

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it,
              "[INFO] Training %s with LR %f..." % (args.model, max(clr)))

        # Train network
        loss, trainer = s.fit(loader=train_loader, epoch=it)
        # save best model
        if loss == min(min_loss, loss):
            print(
                f"Loss reduce from {min_loss} to {loss}. Save to model_best.model")
            s.saveParameters(model_save_path + "/best_state.model")
            early_stopping.counter = 0  # reset counter of early stopping

        min_loss = min(min_loss, loss)

        # Validate and save
        if it % args.test_interval == 0:

            print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "[INFO] Evaluating...")

            sc, lab, _ = s.evaluateFromList(args.test_list,
                                            cohorts_path=None,
                                            eval_frames=args.eval_frames)
            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

            min_eer = min(min_eer, result[1])

            print(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f" %
                (max(clr), trainer, loss, result[1], min_eer))
            score_file.write(
                "IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f\n"
                % (it, max(clr), trainer, loss, result[1], min_eer))

            score_file.flush()

            # NOTE: consider save last state only or not, save only eer as the checkpoint for iterations
            if args.save_model_last:
                s.saveParameters(model_save_path + "/last_state.model")
            else:
                s.saveParameters(model_save_path + "/model_state_%06d.model" % it)

            with open(model_save_path + "/model_state_%06d.eer" % it, 'w') as eerfile:
                eerfile.write('%.4f' % result[1])

            plot_from_file(args.model, show=False)

        else:

            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "LR %f, TEER/TAcc %2.2f, TLOSS %f" % (max(clr), trainer, loss))
            score_file.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f\n" %
                             (it, max(clr), trainer, loss))

            score_file.flush()

        if it >= args.max_epoch:
            score_file.close()
            sys.exit(1)

        if args.early_stop:
            early_stopping(loss)
            if early_stopping.early_stop:
                score_file.close()
                sys.exit(1)

        it += 1
        print("")
