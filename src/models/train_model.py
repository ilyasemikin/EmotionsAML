# Import torch
import torchvision
import torch
import torch.nn as nn
from .model import get_emotions_model
from torch.utils.data import DataLoader, TensorDataset

# Import catalyst
from catalyst.dl import SupervisedRunner
from catalyst.dl.utils import set_global_seed, prepare_cudnn
from catalyst.dl.callbacks import AccuracyCallback, PrecisionRecallF1ScoreCallback, VerboseLogger, ConfusionMatrixCallback

# Other  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# My
from .auc import AUCCallback
from .transform import get_X_scaled
from ..dirs import DIR_DATA_PROCESSED, DIR_DATA_LOGS

if __name__ == '__main__':

    BATCH_SIZE = 16
    N_CLASS = 12

    ref = pd.read_csv(str(DIR_DATA_PROCESSED / 'data_path.csv'))
    mfcc = np.load(str(DIR_DATA_PROCESSED / 'MFCC_PREPARE_AUG.npy'))

    # Split between train and test 
    X_train, X_test, y_train, y_test = train_test_split(mfcc,
        ref.labels,
        test_size=0.25,
        shuffle=True,
        random_state=8
    )

    # one hot encode the target 
    lb = LabelEncoder()

    y_train = lb.fit_transform(y_train) # to_categorical
    y_test = lb.fit_transform(y_test) # to_categorical

    # Normalization as per the standard NN process
    X_train = get_X_scaled(X_train)
    X_test = get_X_scaled(X_test)

    classes_names = list(lb.classes_)

    # model
    model = get_emotions_model(N_CLASS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
    featuresTrain = torch.FloatTensor(X_train[:])
    targetsTrain = torch.from_numpy(y_train[:]).type(torch.LongTensor) # data type is long

    # create feature and targets tensor for test set.
    featuresTest = torch.FloatTensor(X_test[:])
    targetsTest = torch.from_numpy(y_test[:]).type(torch.LongTensor) # data type is long

    trainSet = TensorDataset(featuresTrain,targetsTrain)
    validSet = TensorDataset(featuresTest,targetsTest)

    # data loader
    train_loader = DataLoader(trainSet, batch_size = BATCH_SIZE, shuffle = True)
    valid_loader = DataLoader(validSet, batch_size = BATCH_SIZE, shuffle = False)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Init catalyst components
    runner = SupervisedRunner(device=device)
    loaders = { 'train': train_loader, 'valid': valid_loader }
    logdir = str(DIR_DATA_LOGS / "base_model")
    callbacks = [
        AUCCallback(num_classes = N_CLASS, class_names= classes_names),
        AccuracyCallback(),
        ConfusionMatrixCallback(num_classes = N_CLASS, class_names= classes_names),
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        callbacks=callbacks,
        logdir=logdir,
        num_epochs=100,
        verbose=True,        
    )