from .features import features_extraction
import numpy as np
import pandas as pd
from ..dirs import DIR_DATA_PROCESSED

if __name__ == "__main__":
    input = str(DIR_DATA_PROCESSED / "data_path.csv")
    output = str(DIR_DATA_PROCESSED / "MFCC_PREPARE_AUG_302.npy")

    ref = pd.read_csv(input)

    mfcc = features_extraction(ref)
    np.save(output, mfcc)