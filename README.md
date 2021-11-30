## Heart Failure Prediction Using Machine Learning

#### Section I, Proposal

1. [Overview & Motivation](#overview)
2. [Description](#description)
3. [Reference](#ref)

____

#### Section II, Project Progress

1. [Data Preprocessing](#preprocess)
2. [KNN](#knn)
3. [Decision Tree](#dt)
4. [Random Forest](#rf)
5. [Support Vector Machine](#svm)

<br>

### Section I, Proposal 

#### 1. OVERVIEW AND MOTIVATION <a name='overview'></a>

Heart disease is the leading cause of death for people of most racial and ethnic groups all around the world over the past few decades. According to the data from Centers for Disease Control and Prevention, More than 600,000 Americans die of heart disease each year - that is one in every four deaths in this country [CDC facts](https://www.cdc.gov/heartdisease/facts.htm). About one in five people who had experienced heart attacks were not aware of it when the damage was done. If more people among them can be conscious of and predict the potential risk of heart disease, they are more likely to get an early diagnosis and prompt management of the disease. Besides, from 2016 to 2017, the United States spent about 363 billion dollars on various kinds of costs due to heart disease [CDC facts](https://www.cdc.gov/heartdisease/facts.htm). We could have saved a large amount of money on the difficult laboratory tests and the subsequent healthcare services as well as lost productivity if we had predicted the heart disease cases.

![Heart Disease Death Rates](./images/heart_disease_death_rate.png)



This project aims to create a prediction model to forecast the probability of developing heart disease so as to help individuals be aware of their own health risk and to help doctors make reliable diagnosis more quickly.

The dataset is from UCI Machine Learning Repository, which is the largest heart disease dataset available so far for research purposes (see [Heart Failure Prediction Dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction)). We are going to process and analyze the complex medical data related to heart disease with the help of machine learning techniques including some supervised learning algorithms such as K-Nearest Neighbor, Decision Tree, Support Vector Machine, Random Forest, and Naïve Bayes.

<br>

#### 2. DESCRIPTION <a name='description'></a>

The dataset we decided to use is the Heart Failure Prediction Dataset from University of California Irvine. This dataset was created by combining five independent dataset, which made it the largest heart disease dataset so far for research purposes. The dataset contains one csv file with 12 columns and 918 records, which contains 11 features of patients and 1 target variable. The target variable that describes the presence or absence of heart failure of patients is binary (0, 1) as follows.

| #    |                    Attribute Description                     |               Value               |
| ---- | :----------------------------------------------------------: | :-------------------------------: |
| 1    |              __*Age*__ - the age of the person               | Numerical values between 28 - 77  |
| 2    | _**Sex**_ - the gender of the person.<br/> [‘M’ means male and ‘F’ means female] |            ‘M’ or ‘F’             |
| 3    | _**ChestPainType**_ - the chest pain type of the person. <br/>[TA: Typical Angina, ATA: Atypical Angina, <br/>NAP: Non-Anginal Pain, ASY: Asymptomatic] |   ‘TA’, ‘ATA’, ‘NAP’, or ‘ASY’    |
| 4    | _**Resting BP**_ - the resting blood pressure of the person in millimeters of mercury (mmHg). | Numerical values between 80 - 200 |
| 5    | _**Cholesterol**_ - the serum cholesterol of the person in millimeters per deciliters (mm/dL). | Numerical values between 0 - 603  |
| 6    | _**Fasting BP**_ - the fasting blood pressure of the person. <br>[0: normal and 1 means abnormal] |          0 or 1 (binary)          |
| 7    | _**RestingECG**_ - the resting electrocardiogram results of the person. <br>[‘LVH’: showing probable or definite left ventricular hypertrophy by Estes' criteria, ‘Normal’: Normal, ‘ST’: having ST-T wave abnormality] |     ‘LVH’, ‘Normal’, or ‘ST’      |
| 8    | _**MaxHR**_ - the maximum heart rate achieved by the person. | Numerical values between 60 - 202 |
| 9    | _**ExerciseAngina**_ - exercise-induced angina of the person. <br>[Y: Yes, N: No] |            ‘Y’ or ‘N’             |
| 10   |     _**Oldpeak**_ - the depression status of the person      | Numerical values between 0 - 6.2  |
| 11   | _**ST_Slope**_ - the slope of the person’s peak exercise ST segment. <br>[Up: upsloping, Flat: flat, Down: downsloping] |       ‘Up’, ‘Down’, 'Flat'        |
| 12   | _**HeartDisease**_ - the target variable; if the person has heart disease or not |          0 or 1 (binary)          |

Since this is a classification problem (presence or absence of heart failure), we will be training and tuning K-Nearest Neighbor, Decision Tree, Support Vector Machine, Random Forest, and Naive Bayes models on the dataset, and try to find the best model for our problem.

We will be splitting 918 records into two portions: the training set and the testing set. We will use all of the data to train and test, but not at the same time. We will

1. Use cross validation techniques such as k-fold cross validation or monte-carlo cross validation to train the model using the training set and tune the hyperparameters;
2. Use the best hyperparameter value to train the model using the entire training dataset;
3. Evaluate the model (model assessment) using the test dataset and different evaluation metrics such as accuracy score and F-1 score;
4. Train the model using the best hyperparameter we found and the entire train and test dataset.

<br>

#### 3. REFERENCE <a name='ref'></a>

There are totally 63 code submissions using our dataset on Kaggle. Most of them focused on data visualization and prediction model training. So far, the best model we could find on Kaggle is using the Random Forest model with a 90.2% accuracy score by SVEN ESCHLBECK.

There are also several works done by other researchers using another heart failure dataset from UCI which contains 303 records (which is included in this larger dataset). Since the dataset is too small, final models developed are likely to have low complexity, which will cause high bias towards the result.

<br>

<br>

### Section II, Project Progress

___

#### DATA PREPROCESSING <a name='preprocess'></a>

We encoded the categorical features into numerical representations. The table below shows the original values and encoded values. 

|  #   |   Attribute    |                    Encoding                     |
| :--: | :------------: | :---------------------------------------------: |
|  1   |      Sex       |              ['M' -> 0, 'F' -> 1]               |
|  2   | ChestPainType  | ['ASY' -> 0, 'NAP' -> 1, 'ATA' -> 2, 'TA' -> 3] |
|  3   |   RestingECG   |     ['Normal' -> 0, 'LVH' -> 1, 'ST' -> 2]      |
|  4   | ExerciseAngina |              ['N' -> 0, 'Y' -> 1]               |
|  5   |    ST_Slop     |      ['Up' -> 0, 'Flat' -> 1, 'Down' -> 2]      |

We also splited the data into [training dataset](./data/Train.csv) and [testing dataset](./data/Test.csv), and the split ratio is 2:1, respectively. 

<br>

#### KNN <a name='knn'></a>

For code see [knn.py](./code/knn.py). We impletemented K Nearest Neighbors from scikit learn library. For hyper-parameter tuning, we implemented K-Fold cross validation technique and set the K-Fold to 10. We used GridSearchCV to find the optimal hyper-parameter. Below is the result:

```bash
$ python knn.py
Optimal Hyper-Parameters: {'metric': 'manhattan', 'n_neighbors': 15, 'weights': 'uniform'}
The accuracy score using the optimal Hyper-Parameters to train the model: 0.8910891089108911
The f1 score using the optimal Hyper-Parameters to train the model: 0.907563025210084
```

We can see that the accuracy score reaches 0.89. Since it is not sufficient to only use accuracy score as the evaluation metric, we also used f1 score to evaluate the model since we want to minimize the number of False Negatives (which is important for medical data prediction). We can see that the F1 score reaches 0.91 for K Nearest Neighbors classifier.

<br>

#### DECISION TREE <a name='dt'></a>

For code see [dt.py](./code/dt.py). We impletemented Decision Tree classifier from scikit learn library. For hyper-parameter tuning, we impletemented K-Fold cross validation technique and set the K-Fold to 10. Using GridSearchCV, we found the optimal hyper-parameter. Below is the result:

```bash
$ python dt.py
Optimal Hyper-Parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 9}
The accuracy score using the optimal Hyper-Parameters to train the model: 0.834983498349835
The f1 score using the optimal Hyper-Parameters to train the model: 0.8579545454545454
```

The accuracy score for the decision tree model is 0.83, and f1 score is 0.86. For the next approach, I will use bagging technique to create a random forest. 

<br>

#### RANDOM FOREST <a name='rf'></a>

For code see [rf.py](./code/rf.py). We impletemented Random Forest from scikit learn library. For hyper-parameter uning, we impletemented GridSearchCV and K-Fold cross validation to estimate the OOB error. 

```bash
$ python rf.py
```
<br>

#### Support Vector Machine <a name='svm'></a>

For code see [svm.py](./code/svm.py). We impletemented Support Vector Machine classifier from scikit learn library. For hyper-parameter tuning, we impletemented K-Fold cross validation technique and set the K-Fold to 10. Using GridSearchCV, we found the optimal hyper-parameter. Below is the result:

```bash
$ python svm.py
Optimal Hyper-Parameters: {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}
The accuracy score using the optimal Hyper-Parameters to train the model: 0.8712871287128713
The f1 score using the optimal Hyper-Parameters to train the model: 0.888888888888889
```

The accuracy score for the decision tree model is 0.87, and f1 score is 0.89.



Optimal Hyper-Parameters: {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}
The accuracy score using the optimal Hyper-Parameters to train the model: 0.8804347826086957
The f1 score using the optimal Hyper-Parameters to train the model: 0.9009009009009009



Logistic Regression

Optimal Hyper-Parameters: {'C': 0.01, 'penalty': 'l2', 'solver': 'newton-cg'}
The accuracy score using the optimal Hyper-Parameters to train the model: 0.894927536231884
The f1 score using the optimal Hyper-Parameters to train the model: 0.9134328358208956
