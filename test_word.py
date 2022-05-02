import argparse
import glob
import logging
import sys
import time
from pathlib import Path

from sklearn.preprocessing import KernelCenterer
from tqdm.auto import tqdm

from model import SpeakerNet
from processing.wav_conversion import *
from utils import cprint, read_config, similarity_measure

parser = argparse.ArgumentParser(description="Testing word spelling")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


list_of_words = ['apple',
                 'banana',
                 'watermelon',
                 'bench',
                 'flower',
                 'chair',
                 'cup',
                 'lights',
                 'boy',
                 'girl',
                 'television',
                 'book',
                 'pen',
                 'star',
                 'candy']


def prepare_embedings(args, model, emb_path, dts_path, top=3):
    os.makedirs(emb_path, exist_ok=True)

    for classid in tqdm(glob.glob(f"{dts_path}/*"), desc='Preparing embeddings'):
        os.makedirs(f"{emb_path}/{Path(classid).name}", exist_ok=True)
        for i, fpath in enumerate(list(glob.glob(f"{classid}/*.wav"))[:top]):
            emb = np.asarray(model.embed_utterance(fpath,
                                                   eval_frames=args.eval_frames,
                                                   num_eval=args.num_eval,
                                                   normalize=True,
                                                   sr=args.sample_rate))

            np.save(f"{emb_path}/{Path(classid).name}/{i}.npy", np.mean(emb, axis=0))


def check_matching(args, model, source, emb_path, keyword, threshold, convert=False):
    if isinstance(source, str) and convert:
        source = convert_audio_file(source, rate=16000, channels=1, format='wav')

    test_emb = np.asarray(model.embed_utterance(source,
                                                eval_frames=args.eval_frames,
                                                num_eval=args.num_eval,
                                                normalize=True,
                                                sr=args.sample_rate))
    emb_path = glob.glob(f"{emb_path}/{keyword}/*.npy")
    recorded_embs = [np.load(f) for f in emb_path]
    scores = []
    for emb in recorded_embs:
        scores.append(similarity_measure(ref=np.mean(test_emb, axis=0), com=emb))

    logging.info(f"{keyword} \n\
        mean score: {np.mean(scores)} max score: {np.max(scores)} min score: {np.min(scores)}\n\
        {np.mean(scores) > threshold}")


if __name__ == '__main__':
    config_path = 'backup/RawNet2v2/AAmSoftmax/config/config.yaml'
    model_path = str(Path('backup/RawNet2v2/AAmSoftmax/model/best_state.pt'))

    parser.add_argument('--path', '-p',
                        type=str,
                        default=None,
                        help='path to test file')
    parser.add_argument('--keyword', '-k',
                        type=int,
                        default=0,
                        help='keyword index 0 for apple, 1 for banana, 2 for watermelon, \
                        3 for bench, 4 for flower, 5 for chair, 6 for cup, 7 for lights, 8 for boy, ...')
    #######################################################################################

    argv = parser.parse_args()

    args = read_config(config_path)

    t = time.time()
    model = SpeakerNet(**vars(args))
    model.loadParameters(model_path, show_error=False)
    model.eval()

    logging.info(f'Loaded model in {time.time() - t:.2f}s')

    logging.info('Preparing embeddings for known words')
    prepare_embedings(args, model, 'dataset/embedding', dts_path='./dataset/words_dts/')
    logging.info('Done')

    logging.info('Checking matching for unknown words')
    if argv.path is None:

        for audio_f in glob.glob('dataset/test/nam_test_wrong/spell_wrong_0_vad_*.wav'):
            keyword = str(Path(audio_f).name.split('_')[-1].split('.')[0])
            if int(keyword) in range(len(list_of_words)):
                check_matching(args, model,
                               audio_f,
                               'dataset/embedding',
                               str(int(keyword) + 1),
                               args.threshold)
            logging.info('Done')
    else:
        check_matching(args, model,
                       argv.path,
                       'dataset/embedding',
                       str(int(argv.keyword) + 1),
                       args.threshold)
    sys.exit(0)
