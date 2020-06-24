#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

from ..dirs import DIR_DATA_PROCESSED

TESS = str(DIR_DATA_PROCESSED / "kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/")
RAV = str(DIR_DATA_PROCESSED / "kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/")
SAVEE = str(DIR_DATA_PROCESSED / "kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/")
CREMA = str(DIR_DATA_PROCESSED / "kaggle/input/cremad/AudioWAV/")

output = str(DIR_DATA_PROCESSED / "data_path.csv")

def get_datasets(em_get, gn_get, convert, dir_path):
    emotions = []
    em_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            emotion = em_get(file)
            if emotion in convert:
                emotions.append(gn_get(file) + "_" + convert[emotion])
                em_paths.append(root + "/" + file)
    return (emotions, em_paths)

def get_dataframe(name, emotions, paths):
    df = pd.DataFrame(emotions, columns=["labels"])
    df["source"] = name
    df = pd.concat([df, pd.DataFrame(paths, columns = ['path'])], axis = 1)
    return df

if __name__ == "__main__":
    # SAVEE dataset

    SAVEE_convert = {
        "_a": "angry",
        "_d": "disgust",
        "_f": "fear",
        "_h": "happy",
        "_n": "neutral",
        "sa": "sad"
        # "su": "surprise"
    }

    [emotions, paths] = get_datasets(
        lambda f: f[-8:-6], 
        lambda _: "male", 
        SAVEE_convert, 
        SAVEE
    )
    SAVEE_df = get_dataframe("SAVEE", emotions, paths)

    # TESS
    TESS_convert = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        # "ps": "surprise"
    }

    [emotions, paths] = get_datasets(
        lambda f: f.split(".")[0].split("_")[2], 
        lambda _: "female", 
        TESS_convert, 
        TESS
    )
    TESS_df = get_dataframe("TESS", emotions, paths)

    # RAV
    RAV_convert = {
        "1": "neutral",
        "2": "neutral",
        "3": "happy",
        "4": "sad",
        "5": "angry",
        "6": "fear",
        "7": "disgust",
        # "8": "surprise"
    }

    [emotions, paths] = get_datasets(
        lambda f: str(int(f.split(".")[0].split("-")[2])),
        lambda f: "female" if int(f.split(".")[0].split("-")[6]) % 2 == 0 else "male",
        RAV_convert,
        RAV
    )
    RAV_df = get_dataframe("RAV", emotions, paths)

    # CREMA-D
    CREMA_convert = {
        "ANG": "angry",
        "SAD": "sad",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral"
    }

    def female_pred(file):
        female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
            1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]
        return "female" if int(file.split("_")[0]) in female else "male"

    [emotions, paths] = get_datasets(
        lambda f: f.split("_")[2], 
        female_pred, 
        CREMA_convert, 
        CREMA
    )
    CREMA_df = get_dataframe("CREMA", emotions, paths)

    # Combine all
    df = pd.concat([SAVEE_df, RAV_df, TESS_df, CREMA_df], axis=0)
    df.to_csv(output, index=False)