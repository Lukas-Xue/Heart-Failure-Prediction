import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from warnings import simplefilter 


def train_(xTrain, yTrain):
    knn = neighbors.KNeighborsClassifier(metric='manhattan', n_neighbors=15, weights='uniform')
    dt = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=5)
    lr = LogisticRegression(C=0.01, penalty='l2', solver='newton-cg')
    # nb = GaussianNB(var_smoothing=0.0533669923120631)
    svc = svm.SVC(C=1, gamma=0.1, kernel='rbf')
    rf = RandomForestClassifier(max_depth=30, max_features='sqrt', min_samples_leaf=2, n_estimators=162)
    
    model = VotingClassifier(estimators=
                            [('knn', knn), ('dt', dt), ('lr', lr), ('svm', svc), ('rf', rf)], voting='hard')
    model.fit(xTrain, yTrain)
    # print('Optimal Hyper-Parameters:', model.best_params_)
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
