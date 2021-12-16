from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


# train the knn model
def train_(xTrain, yTrain):
    # use GridSearchCV for 5-fold cv to tune the hyper parameter
    grid_param = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'C': [100, 10, 1.0, 0.1, 0.01]}

    knn = GridSearchCV(LogisticRegression(),
                       grid_param,
                       cv=10)

    knn.fit(xTrain, yTrain)
    print('Optimal Hyper-Parameters:', knn.best_params_)
    return knn


# make the prediction and see the score
def test_(knn, xTest):
    return knn.predict(xTest)


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

    # separating x and y
    yTrain = train['HeartDisease'].copy(deep=True)
    xTrain = train.drop(columns=['HeartDisease'])
    yTest = test['HeartDisease'].copy(deep=True)
    xTest = test.drop(columns=['HeartDisease'])

    # proper scaling
    stdScale = MinMaxScaler().fit(xTrain)
    xTrain = stdScale.transform(xTrain)
    xTest = stdScale.transform(xTest)

    # train and test
    model = train_(xTrain, yTrain)
    yHat = test_(model, xTest)
    print('The accuracy score using the optimal Hyper-Parameters to train the model:', accuracy_score(yTest, yHat))
    print('The f1 score using the optimal Hyper-Parameters to train the model:',
          f1_score(yTest, yHat))
    model.fit(xTrain, yTrain)
    metrics.plot_roc_curve(model, xTest, yTest)
    plt.show()


if __name__ == '__main__':
    main()
