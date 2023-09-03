# Salary Prediction Model
My First Machine Learning Model!

## Project Overview
In this project, I've used a very basic Salary prediction dataset available on kaggle https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer . The dataset contains Age, Gender, Education Level, Job Title, Years of experience and Salaries of over 376 employs of a particular company, located in a particular country and have a particular currency, which is not mentioned in the provided dataset. The reason for this is that, this is my first machine learning project so I wanted to keep it as simple as possible (for now).

## Project Details
This machine learning model is build on Python programming language and the libraries used in this project are Pandas and Scikit learn. Scikit Learn library is a really big python machine learning library and the most important functions I have used from it are train_test_split, cross_val_score, RandomForestRegressor, SimpleImputer, OneHotEncoder, ColumnTransformer and Pipelines.

## Project Steps
##### Step #1:
In the First step, I have separated the features(X) from the prediction target(y), then I have splitted those features and prediction target into training data(90% of overall data) and testing data(10% of overall data) using the train_test_split function of sklearn library.
##### Step #2:
After that I have applied SimpleImputer and OneHotEncoding preprocessing onto the training data to replace missing values and to encode the categorical values into binary values. (The reason for this is that, the model I have used neither takes NaN values as an input not categorical values as an input)
##### Step #3:
Then I have written a code which will go through applying 10 different n_estimators values to the model (on training data) and calculate which n_estimator value is giving the lowest Mean Absolute Error through cross validation (on training data).
##### Step #4: 
In the next step I am applying that n_estimators value on our model to predict the salaries of our testing data. (Except this prediction part I've not used the testing data anywhere else, so it acts as a unseen data and helps to avoid overfitting in the model)
##### Step #5:
lastly, I have saved all the result into a data frame and then convereted it to a output csv file which contains the result of my models prediction.
