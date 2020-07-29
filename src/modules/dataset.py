import numpy as np
import librosa

from pydub import AudioSegment
from pydub.utils import mediainfo
from torch.utils.data import Dataset

class WavParts:
    def __init__(self, audio_path, part_seconds=4):
        info = mediainfo(audio_path)
        
        self.sample_rate = int(info['sample_rate'])
        self.wav_data = AudioSegment.from_wav(audio_path)
        self.s = part_seconds
        self.c = 0
        self.done = False
        self.durations = []

    def __iter__(self):
        self.c = 0
        self.done = False
        return self

    def __next__(self):
        if self.done:
            raise StopIteration

        data = self.wav_data
        part = data[self.c * 1000 : (self.c + self.s) * 1000]

        self.durations.append(part.duration_seconds)

        array = part.get_array_of_samples()

        self.c += self.s
        if self.c > data.duration_seconds:
            self.done = True

        res = np.array(array).astype(np.float32)

        return res


def extract_features(data, sample_rate, n_mfcc=40):
    stft=np.abs(librosa.stft(data))
    result=np.array([])
    mfccs=np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc).T,axis=0)
    result=np.hstack((result, mfccs))
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    result=np.hstack((result, chroma))
    mel=np.mean(librosa.feature.melspectrogram(data, sr=sample_rate).T,axis=0)
    result=np.hstack((result, mel))
    return result

class AudioLoader:
    def __init__(self, audio_path, duration):
        parts = WavParts(audio_path, part_seconds=duration)
        sr = parts.sample_rate

        f = lambda x: extract_features(x, sr)
        self.features = list(map(f, parts))
        self.durations = parts.durations