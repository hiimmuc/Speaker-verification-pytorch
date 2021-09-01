import glob
import os
import sys
import time

from model import SpeakerNet
from utils import get_data_loader, tuneThresholdfromScore


def train(args):
    # Initialise directories
    model_save_path = args.save_path + f"/{args.model}/model"
    result_save_path = args.save_path + f"/{args.model}/result"

    # Load models
    s = SpeakerNet(args, **vars(args))

    it = 1
    min_loss = float("inf")
    sumloss = 0
    min_eer = [100]

    # Load model weights
    model_files = glob.glob(os.path.join(
        model_save_path, 'model_state_*.model'))

    model_files.sort()

    # if exists best model load from it
    if len(model_files) > 0:
        if os.path.exists(f'{model_save_path}/best_state.model'):
            prev_model_state = f'{model_save_path}/best_state.model'
        else:
            prev_model_state = model_files[-1]

    if args.initial_model:
        s.loadParameters(args.initial_model)
        print("Model %s loaded!" % args.initial_model)
    elif len(model_files) >= 1:
        s.loadParameters(prev_model_state)
        print("Model %s loaded from previous state!" % prev_model_state)
    else:
        print("Train model from scratch!")

    # model_state_xxxxxx.model, so 12 is index of number sequence
    start_it = int(os.path.splitext(os.path.basename(model_files[-1]))[0][12:]) + 1
    
    if args.max_epoch > start_it:
        it = start_it
    else:
        it = 1

    for ii in range(0, it - 1):
        s.__scheduler__.step()

    # Write args to score_file
    score_file = open(result_save_path + "/scores.txt", "a+")
    # summary settings
    score_file.write(f'\n------------------{time.strftime("%Y-%m-%d %H:%M:%S")}------------------\n')
    for items in vars(args):
        print(items, vars(args)[items])
        score_file.write('%s %s\n' % (items, vars(args)[items]))
    score_file.flush()

    # Initialise data loader
    train_loader = get_data_loader(args.train_list, **vars(args))

    # Training loop
    while True:
        clr = [x['lr'] for x in s.__optimizer__.param_groups]

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it,
              "[INFO] Training %s with LR %f..." % (args.model, max(clr)))

        # Train network
        loss, trainer = s.train_network(loader=train_loader, epoch=it)
        # save best model
        if loss == min(min_loss, loss):
            print(
                f"Loss reduce from {min_loss} to {loss}. Save to model_best.model")
            s.save_parameters(model_save_path + "/best_state.model")
        min_loss = min(min_loss, loss)

        # Validate and save
        if it % args.test_interval == 0:

            print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "[INFO] Evaluating...")

            sc, lab, _ = s.evaluateFromList(args.test_list,
                                            cohorts_path=None,
                                            eval_frames=args.eval_frames)
            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

            min_eer.append(result[1])

            print(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f" %
                (max(clr), trainer, loss, result[1], min(min_eer)))
            score_file.write(
                "IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f\n"
                % (it, max(clr), trainer, loss, result[1], min(min_eer)))

            score_file.flush()

            s.saveParameters(model_save_path + "/model_state_%06d.model" % it)

            with open(model_save_path + "/model_state_%06d.eer" % it, 'w') as eerfile:
                eerfile.write('%.4f' % result[1])

        else:

            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "LR %f, TEER/TAcc %2.2f, TLOSS %f" % (max(clr), trainer, loss))
            score_file.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f\n" %
                             (it, max(clr), trainer, loss))

            score_file.flush()

        if it >= args.max_epoch:
            score_file.close()
            sys.exit(1)

        it += 1
        print("")
