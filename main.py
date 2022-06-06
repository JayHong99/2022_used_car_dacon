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
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                            new_train.drop(columns = ['target']),
                                                            new_train['target'],
                                                            test_size = 0.1,
                                                            shuffle = True,
                                                            random_state = 0,
                                                        )
    training_models = modeling(np.array(X_train), np.array(X_valid), np.array(y_train), np.array(y_valid))
    linear_pred = training_models.linear_regression()
    rf_pred = training_models.rf()
    ext_pred = training_models.ext()
    lgbm_pred = training_models.lgbm()
    xgb_pred = training_models.xgb()

    best_score = np.inf
    for p1 in tqdm(np.arange(0, 1, 0.05), total = 20) : 
        for p2 in np.arange(0, 1, 0.05) : 
            for p3 in np.arange(0, 1, 0.05) : 
                for p4 in np.arange(0, 1, 0.05) : 
                    for p5 in np.arange(0, 1, 0.05) : 
                        if (p1 + p2 + p3 + p4 + p5) != 1 : 
                            continue
                        pred1 = linear_pred * p1
                        pred2 = rf_pred * p2
                        pred3 = ext_pred * p3
                        pred4 = lgbm_pred * p4
                        pred5 = xgb_pred * p5
                        ensemble = pred1 + pred2 + pred3 + pred4 + pred5
                        score = training_models.score(ensemble)
                        if score < best_score : 
                            best_score = score
                            best_combination = [p1, p2, p3, p4, p5]    

    test_models = modeling(np.array(new_train.drop(columns = ['target'])), np.array(new_test), 
                            np.array(new_train['target']), None)
    linear_pred = test_models.linear_regression()
    rf_pred = test_models.rf()
    ext_pred = test_models.ext()
    lgbm_pred = test_models.lgbm()
    xgb_pred = test_models.xgb()

    ensemble = [p * pred for p, pred in zip(best_combination, [linear_pred, rf_pred, ext_pred, lgbm_pred, xgb_pred])] 
    ensemble = np.mean(ensemble, axis = 0)
    sample = pd.read_csv(data_path.joinpath('sample_submission.csv'))
    sample['target'] = ensemble
    sample.to_csv(save_path.joinpath('2022_06_05_result2.csv'), index=False)

    return


if __name__ == "__main__"  :
    main()