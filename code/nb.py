import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from warnings import simplefilter 


def train_(xTrain, yTrain):
    # grid search and cross validation
    grid_param = {'var_smoothing': np.logspace(0,-9, num=100)}
    model = GridSearchCV(GaussianNB(),
                         grid_param,
                         cv=10,
                         verbose=1,
                         n_jobs=-1)
    model.fit(xTrain, yTrain)
    print('Optimal Hyper-Parameters:', model.best_params_)
    return model


def predict(model, xTest):
    return model.predict(xTest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",
                        default="../data/Train.csv",
                        help="file name of the training dataset")
    parser.add_argument("--test",
                        default="../data/Test.csv",
                        help="file name of the testing dataset")
    args = parser.parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    simplefilter(action='ignore', category=DeprecationWarning)
    simplefilter(action='ignore', category=FutureWarning)

    # separating x and y
    yTrain = train['HeartDisease'].copy(deep=True)
    xTrain = train.drop(columns=['HeartDisease'])
    yTest = test['HeartDisease'].copy(deep=True)
    xTest = test.drop(columns=['HeartDisease'])

    # proper scaling
    stdScale = StandardScaler().fit(xTrain)
    xTrain = stdScale.transform(xTrain)
    xTest = stdScale.transform(xTest)

    dt = train_(xTrain, yTrain)
    yHat = predict(dt, xTest)

    # score
    print('The accuracy score using the optimal Hyper-Parameters to train the model:', accuracy_score(yTest, yHat))
    print('The f1 score using the optimal Hyper-Parameters to train the model:',
          f1_score(yTest, yHat))


if __name__ == '__main__':
    main()
