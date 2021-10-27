import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler


# train the knn model
def train_(xTrain, yTrain):
    knn = neighbors.KNeighborsClassifier(10)
    knn.fit(xTrain, yTrain)
    return knn


# make the prediction and see the score
def test_(knn, xTest, yTest):
    y_hat = knn.predict(xTest)
    return accuracy_score(yTest, y_hat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",
                        default="./Train.csv",
                        help="file name of the training dataset")
    parser.add_argument("--test",
                        default="./Test.csv",
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
    acc = test_(model, xTest, yTest)
    print(acc)


if __name__ == '__main__':
    main()
