import glob
import os
import sys
import time

from model import SpeakerNet
from utils import get_data_loader, tuneThresholdfromScore


def train(args):
    # Initialise directories
    model_save_path = args.save_path + f"/{args.model}_model"
    result_save_path = args.save_path + f"/{args.model}_result"

    # Load models
    s = SpeakerNet(**vars(args))

    it = 1
    prevloss = float("inf")
    sumloss = 0
    min_eer = [100]

    # Load model weights
    model_files = glob.glob(f'%{model_save_path}/model0*.model')
    model_files.sort()

    if args.initial_model != "":
        s.loadParameters(args.initial_model)
        print("Model %s loaded!" % args.initial_model)
    elif len(model_files) >= 1:
        s.loadParameters(model_files[-1])
        print("Model %s loaded from previous state!" % model_files[-1])
        it = int(os.path.splitext(
            os.path.basename(model_files[-1]))[0][5:]) + 1

    for ii in range(0, it - 1):
        s.__scheduler__.step()

    # Write args to score_file
    score_file = open(result_save_path + "/scores.txt", "a+")

    for items in vars(args):
        print(items, vars(args)[items])
        score_file.write('%s %s\n' % (items, vars(args)[items]))
    score_file.flush()

    # Initialise data loader
    train_loader = get_data_loader(args.train_list, **vars(args))

    while True:
        clr = [x['lr'] for x in s.__optimizer__.param_groups]

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it,
              "Training %s with LR %f..." % (args.model, max(clr)))

        # Train network
        loss, trainer = s.train_network(loader=train_loader)

        # Validate and save
        if it % args.test_interval == 0:

            print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...")

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

            s.saveParameters(model_save_path + "/model%09d.model" % it)

            with open(model_save_path + "/model%09d.eer" % it, 'w') as eerfile:
                eerfile.write('%.4f' % result[1])

        else:

            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "LR %f, TEER/TAcc %2.2f, TLOSS %f" % (max(clr), trainer, loss))
            score_file.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f\n" %
                             (it, max(clr), trainer, loss))

            score_file.flush()

        if it >= args.max_epoch:
            sys.exit(1)

        it += 1
        print("")
    score_file.close()
