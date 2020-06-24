from pathlib import Path

import pandas as pd

from ..dirs import DIR_DATA_RAW, DIR_DATA_PROCESSED
from .load_datasets import get_dataframe

if __name__ == "__main__":
    
    data_path = Path(DIR_DATA_RAW / "test")
    
    data = pd.read_csv(str(data_path / "data.csv"))

    output_path = str(data_path / DIR_DATA_PROCESSED / "data_path_test.csv")
    print(output_path)

    paths = []
    for item in data["ID"]:
        paths.append(str(data_path / f"{(item)}.mp3"))

    emotions = []
    for emotion, gender in zip(data["EMOTION"], data["GENDER"]):
        emotions.append(f"{gender}_{emotion}")

    df = get_dataframe("MY", emotions, paths)
    df.to_csv(output_path, index=False)