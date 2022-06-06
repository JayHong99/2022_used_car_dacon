from pathlib import Path
import pandas as pd


def load_data(data_path : Path
    ) -> pd.DataFrame : 
    train = pd.read_csv(data_path.joinpath('train.csv'))
    test  = pd.read_csv(data_path.joinpath('test.csv'))
    return train, test
