import os
import csv
from tqdm import tqdm
from pathlib import Path
import argparse

from audio_utils import *
parser = argparse.ArgumentParser(description="Filtering low quality audio")


def get_error_list(imposter_file):
    print("Get information from:", imposter_file)
    
    if os.path.isfile(imposter_file):
        with open(imposter_file, 'r') as rf:
            lines = [line.strip().replace('\n', '') for line in rf.readlines()]

        # invalid_class = list(''.join(x.split(':')[1:]).strip() for x in filter(lambda x: True if ':' in x else False, lines))
        # invalid_files = list(''.join(x.split('-')[1:]).strip() for x in filter(lambda x: True if '-' in x else False, lines))
        # # len(invalid_files), len(invalid_class), invalid_class[-1], glob.glob("dataset/train/*").index(invalid_class[-1])

        invalid_details = {}
        for line in tqdm(lines):
            if ':' in line:
                k = ''.join(line.split(':')[1:]).strip()
                if k not in invalid_details:
                    invalid_details[k] = {}
            elif '.wav' in line:
                fp = ''.join(line.split(' - ')[1:]).strip()
                n = line.split('-')[0].strip().replace('[', '').replace(']', '').split('/')

                rate = float(n[0])/float(n[1])

                k = list(invalid_details.keys())[-1]
                invalid_details[k][fp] = rate

        return invalid_details
    else:
        return None
    
def read_blacklist(id, duration_limit=1.0, 
                   dB_limit=-16, 
                   error_limit=None, 
                   noise_limit=-15, 
                   details_dir="dataset/train_details_full/"):
    '''
    header = ['File name', 'Duration', 'Size(MB)', 'Min level', 'Max level', 
              'Min difference', 'Max difference', 'Mean difference', 'RMS difference', 
              'Peak level dB', 'RMS level dB',   'RMS peak dB', 'RMS trough dB', 
              'Crest factor', 'Flat factor', 'Peak count',
              'Noise floor dB', 'Noise floor count', 'Bit depth', 'Dynamic range', 
              'Zero crossings', 'Zero crossings rate', 'Error rate', 'Full path']
    '''
    blacklist = []
    readfile = str(Path(details_dir, f"{id}.csv"))
    if not error_limit:
        # auto choose
        error_limit = 0.5
        
    if os.path.exists(readfile):
        with open(readfile, 'r', newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for row in spamreader:
                short_length = (float(row[1]) < duration_limit)
                low_amplitude = (float(row[9]) < dB_limit)
                high_err = (float(row[-2]) > error_limit)
                high_noise = (float(row[16]) > noise_limit)
                if short_length or low_amplitude or high_err or high_noise:
                    blacklist.append(Path(row[-1]))
        return list(set(blacklist))
    else:
        return None

def get_audio_properties(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration, rate
    
    
def check_valid_audio(files, duration_lim=1.5, sr=8000):
    filtered_list = []
    files = [str(path) for path in files]
    
    for fname in files:
        duration, rate = get_audio_properties(fname)
        if rate == sr and duration >= duration_lim:
            filtered_list.append(fname)
        else:
            pass
    filtered_list.sort(reverse=True, key = lambda x: get_audio_properties(x)[0])    
    filtered_list = [Path(path) for path in filtered_list]
    return filtered_list


def export_dataset_details(root="dataset/train", save_dir="dataset/train_details/"):
    root = Path(root)
    print("Getting general information")
#     invalid_details = get_error_list('Imposter_callbot.txt')
    _, audio_folder_duration, audio_folder_size = get_dataset_general_infor(root)
    os.makedirs(save_dir, exist_ok=True)
    
    for audio_folder in tqdm(list(root.iterdir()), desc="Processing...", colour='red'):
        writefile = os.path.join(save_dir , f"{audio_folder.name}.csv")
        
        with open(writefile, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            header = ['File name', 'Duration', 'Size(MB)', 'Min level', 'Max level', 
                      'Min difference', 'Max difference', 'Mean difference', 'RMS difference', 
                      'Peak level dB', 'RMS level dB',   'RMS peak dB', 'RMS trough dB', 
                      'Crest factor', 'Flat factor', 'Peak count',
                      'Noise floor dB', 'Noise floor count', 'Bit depth', 'Dynamic range', 
                      'Zero crossings', 'Zero crossings rate', 'Error rate', 'Full path']
            
            spamwriter.writerow(header)
            
            for i, audio_file in enumerate(audio_folder.iterdir()):
                # general infor
                fp = str(Path(audio_folder, audio_file.name))
                duration = audio_folder_duration[audio_folder.name][i]
                size = audio_folder_size[audio_folder.name][i]

                error_rate = 0
                details = get_audio_ffmpeg_astats(fp)
                
                row = [audio_file.name, details['Duration'], details['Size'], details['Min level'],details['Max level'],
                       details['Min difference'],details['Max difference'], details['Mean difference'],details['RMS difference'],
                       details['Peak level dB'],details['RMS level dB'], details['RMS peak dB'],details['RMS trough dB'],
                       details['Crest factor'],details['Flat factor'], details['Peak count'],
                       details['Noise floor dB'],details['Noise floor count'],details['Bit depth'],details['Dynamic range'],
                       details['Zero crossings'],details['Zero crossings rate'], error_rate, fp]

                spamwriter.writerow(row)
                
    return True


def update_dataset_details(root="dataset/train", save_dir="dataset/train_details/", error_file="Imposter_callbot2.txt"):
    root = Path(root)
    print("Getting general information")
    invalid_details = get_error_list(error_file)
    os.makedirs(save_dir, exist_ok=True)
    
    for audio_folder in tqdm(list(root.iterdir())[:], desc="Processing..."):
        reading_file = os.path.join(save_dir , f"{audio_folder.name}.csv")
        writing_file = reading_file
        
        rows = []
        with open(reading_file, 'r', newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for row in spamreader:
                rows.append(row)
        
        with open(writing_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            header = ['File name', 'Duration', 'Size(MB)', 'Min level', 'Max level', 
                      'Min difference', 'Max difference', 'Mean difference', 'RMS difference', 
                      'Peak level dB', 'RMS level dB',   'RMS peak dB', 'RMS trough dB', 
                      'Crest factor', 'Flat factor', 'Peak count',
                      'Noise floor dB', 'Noise floor count', 'Bit depth', 'Dynamic range', 
                      'Zero crossings', 'Zero crossings rate', 'Error rate', 'Full path']
            
            spamwriter.writerow(header)
            
            for i, audio_file in enumerate(list(audio_folder.iterdir())):
                fp = rows[i][-1]
                
                if isinstance(invalid_details, dict):
                    if str(root / audio_folder.name) in invalid_details.keys():
                        if fp in invalid_details[str(root / audio_folder.name)]:
                            error_rate = float(invalid_details[str(root / audio_folder.name)][fp])
                        else:
                            error_rate = 0
                    else:
                        error_rate = 0
                else:
                    error_rate = 0
            
                row_new = rows[i]
                row_new[-2] = error_rate
                spamwriter.writerow(row_new)  
    return True


def move_low_quality_files(raw_dataset='dataset/train', 
                           details_dir='dataset/details/train_cskh/', 
                           duration_limit=1.0,
                           dB_limit=-10,                          
                           error_limit=0.5,
                           noise_limit=-10,
                           lower_num=None, upper_num = None, 
                           confirm_mode=None):
    all_spks = []
    valid_spks = []
    
    filepaths_lists = []
    wrong_files = []
    
    root = Path(raw_dataset)
    classpaths = [d for d in root.iterdir() if d.is_dir()]
    classpaths.sort()
    
    for classpath in tqdm(list(classpaths)[:], desc="Processing:..."):
        all_spks.append(Path(classpath).name)
        
        filepaths = list(classpath.glob('*.wav'))
        
        # check low quality files
        blist = read_blacklist(str(Path(classpath).name), 
                               duration_limit=duration_limit, 
                               dB_limit=dB_limit, 
                               error_limit=error_limit, 
                               noise_limit=noise_limit,
                               details_dir=details_dir)
        if not blist:
            continue

        filepaths_ft = list(set(filepaths).difference(set(blist)))

        # check duration, sr
        filepaths_ft = check_valid_audio(filepaths_ft, 1.0, 8000)

        # check number of files
        if lower_num:
            if len(filepaths_ft) < lower_num:
                continue
        if upper_num:
            if len(filepaths_ft) >= upper_num:
                filepaths_ft = filepaths_ft[:upper_num]
                
        if len(filepaths_ft) == 0:
            continue
                
        valid_spks.append(Path(classpath).name)
        filepaths_lists.extend(filepaths_ft)
        wrong_files.extend(list(set(filepaths).difference(set(filepaths_ft))))
    
    invalid_spks = list(set(all_spks).difference(set(valid_spks)))
    
    print("# Valid speakers:", len(valid_spks),
          "over", len(all_spks))
    print("Total valid audio files:", len(filepaths_lists),
          "Total wrong files", len(wrong_files))
    
    if confirm_mode == 'remove':
        for spk in tqdm(invalid_spks, desc="Deleting invalid speaker's dir..."):
            path_dir = os.path.join(raw_dataset, spk)
            if os.path.exists(path_dir):
                subprocess.call(f"rm -rf {path_dir}", shell=True)
        for f in tqdm(wrong_files, desc="Deleting low quality speaker's files..."):
            if os.path.exists(f):
                subprocess.call(f"rm -rf {f}", shell=True)
        print('Done!')
    elif confirm_mode == 'move':
        invalid_dir = str(Path(Path(root).parent / 'invalid_spks/'))
        for spk in tqdm(invalid_spks, desc="Moving invalid speaker's dir..."):
            new_path_dir = os.path.join(invalid_dir, spk)
            old_path_dir = os.path.join(raw_dataset, spk)
            
            os.makedirs(new_path_dir, exist_ok=True)
            if os.path.exists(old_path_dir):
                subprocess.call(f"mv {old_path_dir}/*.wav {new_path_dir}", shell=True)
                
    return valid_spks, invalid_spks


# ======================================================================================================
if __name__ == "__main__":
    parser.add_argument('--root', type=str, default="dataset/test_callbot_raw/namdp5")
    parser.add_argument('--details_dir', type=str, default="dataset/details/test_cb_raw")
    parser.add_argument('--wrong_id_file', type=str, default=None)
    parser.add_argument('--mode', type=str, default='export')
    
    args = parser.parse_args()
    if args.mode == 'export':
        export_dataset_details(root=args.root, save_dir=args.details_dir)
    elif args.mode == 'update':
        update_dataset_details(root=args.root, save_dir=args.details_dir, error_file=args.wrong_id_file)
    elif args.mode == 'move':
        move_low_quality_files(raw_dataset=args.root, 
                                 details_dir=args.details_dir,
                                 duration_limit=1.0,
                                 dB_limit=-10,
                                 error_limit=0.5,
                                 noise_limit=-10,
                                 lower_num=10, upper_num = None, 
                                 confirm_mode='move')
    