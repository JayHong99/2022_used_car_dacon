import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.load_data import load_data
from src.preprocess import preprocess
from src.modeling import modeling


data_path = Path('./data/')
save_path = Path('./result')
save_path.mkdir(exist_ok= True)


def main() : 

    train,test = load_data(data_path)
    processor = preprocess(train, test)
    new_train, new_test = processor.process()   
    new_train.to_csv('./data/new_train.csv')
    new_test.to_csv('./data/new_test.csv')
    return


if __name__ == "__main__"  :
    main()