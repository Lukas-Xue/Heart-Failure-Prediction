import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from warnings import simplefilter 


def train_(xTrain, yTrain):
    # grid search and cross validation
    grid_param = {"loss":["deviance"],
                "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                # "learning_rate": [0.075, 0.1, 0.15],
                "min_samples_split": np.linspace(0.1, 0.5, 12),
                # "min_samples_split": [0.1, 0.2],
                "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                "max_depth":[3,5,8],
                # "max_depth":[8],
                "max_features":["log2","sqrt"],
                # "max_features":["sqrt"],
                "criterion": ["friedman_mse",  "mae"],
                # "criterion": ["mae"],
                "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                # "subsample":[0.618, 0.8, 0.85],
                "n_estimators":[10]}
    model = GridSearchCV(GradientBoostingClassifier(),
                         grid_param,
                         cv=2)
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
