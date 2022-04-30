import glob
import logging
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

from model import SpeakerNet
from processing.wav_conversion import *
from utils import cprint, read_config, similarity_measure

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
        embs = []
        for fpath in list(glob.glob(f"{classid}/*.wav"))[:top]:
            emb = np.asarray(model.embed_utterance(fpath,
                                                   eval_frames=args.eval_frames,
                                                   num_eval=args.num_eval,
                                                   normalize=True,
                                                   sr=args.sample_rate))
            embs.append(emb)
        embs = np.vstack(embs)
        np.save(f"{emb_path}/{Path(classid).name}.npy", np.mean(embs, axis=0))


def check_matching(args, model, source, emb_path, keyword, threshold, convert=False):
    if isinstance(source, str) and convert:
        source = convert_audio_file(source, rate=16000, channels=1, format='wav')

    test_emb = np.asarray(model.embed_utterance(source,
                                                eval_frames=args.eval_frames,
                                                num_eval=args.num_eval,
                                                normalize=True,
                                                sr=args.sample_rate))
    recorded_emb = np.load(f"{emb_path}/{keyword}.npy")

    # print(test_emb.shape, recorded_emb.shape)

    score = similarity_measure(ref=np.mean(test_emb, axis=0), com=recorded_emb)

    logging.info(f"{keyword} score: {score} {score >= threshold}")


if __name__ == '__main__':
    config_path = 'backup/config_train.yaml'
    model_path = str(Path('backup/RawNet2v2/AAmSoftmax/model/best_state.pt'))

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
    check_matching(args, model,
                   'dataset/dat_tow_test_dts_vad_0.wav',
                   'dataset/embedding',
                   str(list_of_words.index('apple') + 1),
                   args.threshold)
    logging.info('Done')

    sys.exit(0)
