import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from catalyst.dl import SupervisedRunner
from src.modules.model import MLP
from src.modules.dataset import AudioLoader

from ..dirs import DIR_DATA_MODELS

def predict(file_path, duration=4):
    model = MLP()
    weights_path = DIR_DATA_MODELS.joinpath('ser.pth').as_posix()
    model.init_weights(weights_path)
    model.eval()

    inputs = AudioLoader(file_path, duration=duration)

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

    features = inputs.features
    durations = inputs.durations

    if len(durations) > 1 and durations[-1] < duration / 10:
        features = features[:-1]
        durations = durations[:-1]

    preds = model(torch.FloatTensor(features)).detach()
    probs = F.softmax(preds, dim=1)

    targets = list(map(lambda x: x.argmax().tolist(), probs))
    labels = list(map(lambda x: predict_classes[x], targets))

    return list(zip(durations, labels))