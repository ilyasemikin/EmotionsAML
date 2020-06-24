import numpy as np
import librosa
from tqdm import tqdm

def mfcc_extract(data, sampling_rate, n_params):
    """
    MFCC extraction.
    """
    MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_params)
    MFCC = np.expand_dims(MFCC, axis=0)
    return MFCC

def logspec_extract(data, sampling_rate, n_params):
    """
    Log-melspectogram extraction.
    """
    melspec = librosa.feature.melspectrogram(data, n_melf=n_params)
    logspec = librosa.amplitude_to_db(melspec)
    logspec = np.expand_dims(logspec, axis=0)
    return logspec

def file_data_extract(path, sampling_rate, duration, offset):
    data, _ = librosa.load(
        path,
        sr=sampling_rate,
        res_type="kaiser_fast",
        duration=duration,
        offset=offset
    )

    input_length = sampling_rate * duration

    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length+offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")
    
    return data

def file_features_extract(path, offset: float, duration: float, n_params=30, sampling_rate=44100, extract = mfcc_extract):
    data = file_data_extract(path, sampling_rate, duration, offset)
    
    data = extract(data, sampling_rate, n_params)

    return data

def features_extraction(df, offset: float, duration: float, n_params=30, sampling_rate=44100, extract = mfcc_extract):
    X = np.empty(shape=(df.shape[0], 1, n_params, int(86 * duration) + 1))

    index = 0
    for fpath in tqdm(df.path):
        X[index] = file_features_extract(fpath, offset, duration, n_params, sampling_rate, extract)
        index += 1
    
    return X