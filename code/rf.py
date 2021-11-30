import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def tuneRF(xTrain, yTrain):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(50, 500, num=5)]
    # Number of features to consider at every split
    max_features = ['sqrt', 'auto']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num=5)]
    max_depth.append(None)
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    parameters = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf,
                  }

    model = GridSearchCV(RandomForestClassifier(), parameters, cv=5, scoring='f1_macro', verbose=1)
    model.fit(xTrain, yTrain)
    return model


def train_test(xTrain, yTrain, xTest, yTest):
    rf = RandomForestClassifier(random_state=42).fit(xTrain, yTrain)
    print("Parameter using: \n", rf.get_params())
    score(rf, xTest, yTest)
    print("\nTuning:")
    model = tuneRF(xTrain, yTrain)
    print("Best parameters: \n", model.best_params_)
    score(model, xTest, yTest)
    return model


def score(model, xTest, yTest):
    yHat = model.predict(xTest)
    print(classification_report(yTest, yHat))
    print("F1 score: ", f1_score(yTest, yHat))
    return


# def train_(xTrain, yTrain):


# def test_(model, xTest):


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

    model = train_test(xTrain, yTrain, xTest, yTest)


if __name__ == '__main__':
    main()

# Fitting 5 folds for each of 180 candidates, totalling 900 fits
# Best parameters: 
#  {'max_depth': 17, 'max_features': 'auto', 'min_samples_leaf': 2, 'n_estimators': 50}
#               precision    recall  f1-score   support

#            0       0.83      0.87      0.85       121
#            1       0.91      0.88      0.90       182

#     accuracy                           0.88       303
#    macro avg       0.87      0.88      0.87       303
# weighted avg       0.88      0.88      0.88       303

# F1 score:  0.8969359331476322
