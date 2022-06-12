import pandas as pd
from torch import seed

from src.seed import seed_everything
from src.modeling import RF

seed_everything(2022)

save_path = './result/2022_0611.csv'



def main() : 
    train = pd.read_csv('./data/new_train.csv').iloc[:,1:]
    test = pd.read_csv('./data/new_test.csv').iloc[:,1:]
    sample = pd.read_csv('./data/sample_submission.csv')
    pbounds = {
        'max_depth' : (1,10),
        'n_estimators' : (30, 300)
    }

    rf = RF(train, test, pbounds)
    rf.get_best_model(5, 5)
    pred = rf.predict()
    sample['target'] = pred
    sample.to_csv(save_path, index=False)

    rf.explain()

    return


if __name__ == "__main__" : 
    main()