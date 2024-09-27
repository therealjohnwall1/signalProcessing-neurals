import os
import numpy as np
import librosa
from typing import Tuple

# split into train, test, val and digitalize .wavs into numpy files
# crop quietness and try to 
# since it is a single classifier, inputs have to be the same size

THRESHOLD_QUIET = 30
MEL_BANKS = 40

RAW_PATH = "dataset/recordings"
DATA_PATH = "dataset/forModel"

def read_wav(path: str, intended_len: int) -> np.array:
    # read from path and digitalize + norm + convert to mels + norm
    sig, sr = librosa.load(path)
    # print(f"reading in signal size{sig.shape}")
    sig, _= librosa.effects.trim(sig, top_db = THRESHOLD_QUIET)

    # norm and zero pad to match max size
    sig = librosa.util.normalize(sig)
    if len(sig) < intended_len:
        sig = np.pad(sig, (0, intended_len - len(sig)), 'constant')

    # print(f"new sig size {sig.shape}")
    # get mel spec: stft + convert into mels(mel filter banks)
    mel_spec = librosa.feature.melspectrogram(y=sig,sr=sr, n_mels=MEL_BANKS)
    mel_spec = librosa.power_to_db(mel_spec)
    # print(f"mel spectogram shape {mel_spec.shape}")
    mel_spec = librosa.util.normalize(mel_spec)

    # equal lengths, will not need to pad
    return mel_spec 


def read_files() -> Tuple[np.array, np.array]:
    max_length = max(
        len(librosa.load(os.path.join(RAW_PATH, sample))[0])
        for sample in os.listdir(RAW_PATH)
        )
    dir_len = len(os.listdir(RAW_PATH))

    data = []
    file_names = np.array([])

    for i, sample in enumerate(os.listdir(RAW_PATH)):
        if i % 500 == 0:
            print(f"{i}/{dir_len} samples")
        
        file_name = os.path.join(RAW_PATH, sample)
        x = read_wav(file_name, max_length)
        file_name = file_name.replace(RAW_PATH, "").lstrip(os.sep)
        file_name = file_name[:-4]

        data.append(x)
        file_names = np.append(file_names, file_name)

    res_data = np.array(data)
    print("data, file , and res_data lengths", len(data), len(file_names), len(res_data))
    print("res data", res_data.shape)
    return res_data, file_names

def split_files(data:np.array, file_names:np.array, data_split:tuple) -> None:
    assert(round(sum(data_split)) == 1)
    assert(len(data_split) == 3)
    assert(data.shape[0] == file_names.size)

    for dir_name in ['train', 'val', 'test']:
        dir_path = os.path.join(DATA_PATH, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    

    train_per, test_per, val_per = data_split
    train_per = int(train_per * len(data))
    test_per = int(test_per * len(data))
    val_per = int(val_per * len(data))
    
    dexes = np.arange(len(data))
    print(f"training split over {len(data)} samples: \n \\\
          {train_per, test_per, val_per}")
    
    np.random.shuffle(dexes)

    print("inserting numpy files into forModel/ ")

    for i in dexes[:train_per]:
        np.save(f"{DATA_PATH}/train/{file_names[i]}.npy", data[i])

    for i in dexes[train_per:train_per + val_per]:
        np.save(f"{DATA_PATH}/val/{file_names[i]}.npy", data[i])

    for i in dexes[val_per+train_per:]:
        np.save(f"{DATA_PATH}/test/{file_names[i]}.npy", data[i])

if __name__ == "__main__":
    data, file_names = read_files()
    split_files(data, file_names, (0.7, 0.2, 0.1))
    



