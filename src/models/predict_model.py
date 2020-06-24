import torch

import pandas as pd

from pathlib import Path

from .model import get_emotions_model
from ..features.features import features_extraction
from .transform import get_X_scaled
from ..dirs import DIR_DATA_LOGS, DIR_DATA_RAW, DIR_DATA_PROCESSED, DIR_DATA_MODELS

if __name__ == "__main__":

    N_CLASS = 12

    model = get_emotions_model(N_CLASS)

    chechpoint = torch.load(str(DIR_DATA_LOGS / "base_model" / "checkpoints" / "best.pth"))
    model.load_state_dict(chechpoint["model_state_dict"])
    model.eval()

    ref = pd.read_csv(str(DIR_DATA_PROCESSED / "data_path_test.csv"))
    inputs = features_extraction(ref, 0.5, 2.5)
    inputs = get_X_scaled(inputs)

    inputs = torch.FloatTensor(inputs[:])

    predict_classes = { 
        0: 'female_angry',
        1: 'female_disgust',
        2: 'female_fear',
        3: 'female_happy',
        4: 'female_neutral',
        5: 'female_sad',
        6: 'male_angry',
        7: 'male_disgust',
        8: 'male_fear',
        9: 'male_happy',
        10: 'male_neutral',
        11: 'male_sad'
    }

    logits = model(inputs)
    predicted = logits.argmax(dim = 1)

    correct_count = 0
    for expc, pred in zip(ref.labels, predicted):
        pred_res = predict_classes[int(pred)]
        if expc == pred_res:
            correct_count = correct_count + 1
        print(f"predicted: {pred_res:15} | expected: {expc:15}")

    print(f"{correct_count} / {len(ref.labels)}")