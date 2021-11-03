import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


# train the knn model
def train_(xTrain, yTrain):
    # use GridSearchCV for 5-fold cv to tune the hyper parameter
    grid_param = {'n_neighbors': [i for i in range(1, 50)],
                  'weights': ['distance', 'uniform'],
                  'metric': ['euclidean', 'manhattan']}

    knn = GridSearchCV(neighbors.KNeighborsClassifier(),
                       grid_param,
                       cv=10)

    knn.fit(xTrain, yTrain)
    print('Optimal Hyper-Parameters:', knn.best_params_)
    return knn
          

# make the prediction and see the score
def test_(knn, xTest, yTest):
    y_hat = knn.predict(xTest)
    return y_hat


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
    stdScale = StandardScaler().fit(xTrain)
    xTrain = stdScale.transform(xTrain)
    xTest = stdScale.transform(xTest)

    # train and test
    model = train_(xTrain, yTrain)
    yHat = test_(model, xTest, yTest)
    acc = accuracy_score(yTest, yHat)
    print('The accuracy score using the optimal Hyper-Parameters to train the model:', acc)
    print('The f1 score using the optimal Hyper-Parameters to train the model:',
          f1_score(yTest, test_(model, xTest, yTest)))


if __name__ == '__main__':
    main()
