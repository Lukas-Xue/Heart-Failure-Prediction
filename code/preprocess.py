import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


# encoding sex into binary: [M -> 0, F -> 1]
def sex(df):
    for i in range(df.shape[0]):
        df.iloc[i, 1] = 0 if df.iloc[i, 1] == 'M' else 1
    return df


# encoding ChestPainType into integer: [ASY -> 0, NAP -> 1, ATA -> 2, TA -> 3]
def chest_pain_type(df):
    for i in range(df.shape[0]):
        if df.iloc[i, 2] == 'ASY':
            df.iloc[i, 2] = 0
        elif df.iloc[i, 2] == 'NAP':
            df.iloc[i, 2] = 1
        elif df.iloc[i, 2] == 'ATA':
            df.iloc[i, 2] = 2
        else:
            df.iloc[i, 2] = 3
    return df


# encoding RestingECG into integer: [Normal -> 0, LVH -> 1, ST -> 2]
def RestingECG(df):
    for i in range(df.shape[0]):
        if df.iloc[i, 6] == 'Normal':
            df.iloc[i, 6] = 0
        elif df.iloc[i, 6] == 'LVH':
            df.iloc[i, 6] = 1
        else:
            df.iloc[i, 6] = 2
    return df


# encoding ExerciseAngina into binary: [N -> 0, Y -> 1]
def ExerciseAngina(df):
    for i in range(df.shape[0]):
        df.iloc[i, 8] = 0 if df.iloc[i, 6] == 'N' else 1
    return df


# encoding ST_Slope into integer: [Up -> 0, Flat -> 1, Down -> 2]
def ST_Slope(df):
    for i in range(df.shape[0]):
        if df.iloc[i, 10] == 'Up':
            df.iloc[i, 10] = 0
        elif df.iloc[i, 10] == 'Flat':
            df.iloc[i, 10] = 1
        else:
            df.iloc[i, 10] = 2
    return df


# do the train test split. Train:Test = 2:1
def split(df):
    train, test = train_test_split(df, test_size=0.33, random_state=1)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="../heart_disease.csv",
                        help="filename of the input data")
    args = parser.parse_args()
    train, test = split(ST_Slope(ExerciseAngina(RestingECG(chest_pain_type(sex(pd.read_csv(args.data)))))))
    train.to_csv('Train.csv', index=False)
    test.to_csv('Test.csv', index=False)


if __name__ == '__main__':
    main()
