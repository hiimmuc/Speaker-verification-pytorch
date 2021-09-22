import argparse
import os

from inference import inference
from train import train


def main(args):
    if args.do_train:
        # TODO: train model
        train(args)
    elif args.do_infer:
        # TODO: evaluate model
        inference(args)
    else:
        raise 'wrong mode'


parser = argparse.ArgumentParser(description="SpeakerNet")
if __name__ == '__main__':
    # YAML
    parser.add_argument('--config', type=str, default=None)

    # control flow
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_infer', action='store_true', default=False)

    # Data loader
    parser.add_argument('--max_frames',
                        type=int,
                        default=100,
                        help='Input length to the network for training')
    parser.add_argument('--eval_frames',
                        type=int,
                        default=100,
                        help='Input length to the network for testing; 0 for whole files')
    parser.add_argument('--batch_size',
                        type=int,
                        default=320,
                        help='Batch size, number of speakers per batch')
    parser.add_argument('--max_seg_per_spk',
                        type=int,
                        default=100,
                        help='Maximum number of utterances per speaker per epoch')
    parser.add_argument('--nDataLoaderThread',
                        type=int,
                        default=2,
                        help='Number of loader threads')
    parser.add_argument('--augment',
                        action='store_true',
                        default=False,
                        help='Augment input')

    # Training details
    parser.add_argument('--device', type=str,
                        default="cuda", help='cuda or cpu')
    parser.add_argument('--test_interval',
                        type=int,
                        default=10,
                        help='Test and save every [test_interval] epochs')
    parser.add_argument('--max_epoch',
                        type=int,
                        default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--trainfunc',
                        type=str,
                        default="aamsoftmax",
                        help='Loss function')
    parser.add_argument('--save_model_last',
                        type=bool,
                        default=True,
                        help='Save only last checkpoint')

    # Optimizer
    parser.add_argument('--optimizer',
                        type=str,
                        default="adam",
                        help='sgd or adam')
    parser.add_argument('--callbacks',
                        type=str,
                        default="steplr",
                        help='Learning rate scheduler')
    parser.add_argument('--early_stop',
                        action='store_true',
                        default=False,
                        help='Early stopping')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate')
    parser.add_argument("--lr_decay",
                        type=float,
                        default=0.95,
                        help='Learning rate decay every [test_interval] epochs')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0,
                        help='Weight decay in the optimizer')

    # Loss functions
    parser.add_argument("--hard_prob",
                        type=float,
                        default=0.5,
                        help='Hard negative mining probability, otherwise random, only for some loss functions'
                        )
    parser.add_argument("--hard_rank",
                        type=int,
                        default=10,
                        help='Hard negative mining rank in the batch, only for some loss functions'
                        )
    parser.add_argument('--margin',
                        type=float,
                        default=0.2,
                        help='Loss margin, only for some loss functions')
    parser.add_argument('--scale_pos',
                        type=float,
                        default=2.0,
                        help='Loss scale, only for some loss functions')
    parser.add_argument('--scale_neg',
                        type=float,
                        default=50.0,
                        help='Loss scale, only for some loss functions')
    parser.add_argument('--scale',
                        type=float,
                        default=30.0,
                        help='Loss scale, only for some loss functions')

    parser.add_argument('--nPerSpeaker',
                        type=int,
                        default=2,
                        help='Number of utterances per speaker per batch, only for metric learning based losses'
                        )
    parser.add_argument('--nClasses',
                        type=int,
                        default=400,
                        help='Number of speakers in the softmax layer, only for softmax-based losses')

    # Load and save
    parser.add_argument('--initial_model',
                        type=str,
                        default=None,
                        help='Initial model weights')
    parser.add_argument('--save_path',
                        type=str,
                        default="exp",
                        help='Path for model and logs')

    # Training and test data
    parser.add_argument('--train_list',
                        type=str,
                        default="dataset/train.def.txt",
                        help='Train list')
    parser.add_argument('--test_list',
                        type=str,
                        default="dataset/data_public_test.txt",
                        help='Evaluation list')
    parser.add_argument('--test_path',
                        type=str,
                        default="dataset/",
                        help='Absolute path to the test set')
    parser.add_argument('--musan_path',
                        type=str,
                        default="dataset/musan_split",
                        help='Absolute path to the test set')
    parser.add_argument('--rir_path',
                        type=str,
                        default="dataset/RIRS_NOISES/simulated_rirs",
                        help='Absolute path to the test set')

    # Model definition for MFCCs
    parser.add_argument('--n_mels',
                        type=int,
                        default=64,
                        help='Number of mel filter banks')
    parser.add_argument('--log_input',
                        type=bool,
                        default=True,
                        help='Log input features')
    parser.add_argument('--model',
                        type=str,
                        default="ResNetSE34V2",
                        help='Name of model definition')
    parser.add_argument('--encoder_type',
                        type=str,
                        default="ASP",
                        help='Type of encoder')
    parser.add_argument('--nOut',
                        type=int,
                        default=512,
                        help='Embedding size in the last FC layer')

    # Infer mode
    parser.add_argument('--eval',
                        dest='eval',
                        action='store_true',
                        help='Eval only')
    parser.add_argument('--test',
                        dest='test',
                        action='store_true',
                        help='Test only')
    parser.add_argument('--test_by_pair',
                        dest='test_by_pair',
                        action='store_true',
                        help='Test only')

    # For test only
    parser.add_argument('--initial_model_infer',
                        type=str,
                        default=None,
                        help='Initial model weights')
    parser.add_argument('--prepare',
                        dest='prepare',
                        action='store_true',
                        help='Prepare embeddings')
    parser.add_argument('-t',
                        '--prepare_type',
                        type=str,
                        default='cohorts',
                        help='embed / cohorts')
    parser.add_argument('--predict',
                        dest='predict',
                        action='store_true',
                        help='Predict')
    parser.add_argument('--cohorts_path',
                        type=str,
                        default=None,
                        help='Cohorts path')
    parser.add_argument('--test_threshold',
                        type=float,
                        default=1.81120,
                        help='Test threshold')

    args = parser.parse_args()
    # Initialise directories
    if args.do_train:
        model_save_path = args.save_path + f"/{args.model}/model"
        os.makedirs(model_save_path, exist_ok=True)
        result_save_path = args.save_path + f"/{args.model}/result"
        os.makedirs(result_save_path, exist_ok=True)
    # Run
    main(args)
