#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# creating a pandas data_frame
file_path="C:/Users/katif/Documents/Salary Data.csv"
df = pd.read_csv(file_path)

# assigning features (x) and prediction target (y) values
y = df.Salary
X = df.drop(['Salary'], axis=1, inplace=False)

# spliting available data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=9999)

# Hi, Peter please explain me the step #2 and #3 which are mentioned below:
y_imputer= SimpleImputer(strategy='mean') # step #1
y_train = y_imputer.fit_transform(y_train.values.reshape(-1, 1))  # step #2
y_train = pd.Series(y_train_imputed.flatten(), name='Salary')  # step #3

# preprocessing columns present in features(X_train)
numerical_cols=[]
for col in X_train.columns:
    if X_train[col].dtype == 'float64'or X_train[col].dtype == 'int':
        numerical_cols.append(col)
        
categorical_cols=[]
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        categorical_cols.append(col)
        
numerical_transformer= Pipeline(steps=[
    ('num_imputer', SimpleImputer(strategy='mean'))
])
categorical_transformer= Pipeline(steps=[
    ('str_imputer', SimpleImputer(strategy='most_frequent')),
    ('OH_encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessing= ColumnTransformer(transformers=[
    ('num_transformer_x', numerical_transformer, numerical_cols),
    ('cat_transformer', categorical_transformer, categorical_cols)
])

# creating a function that would return mae(by cross validation) for the estimator value we provide
def get_mae(es):   
    pipeline_model = Pipeline(steps=[
        ('preprocessor', preprocessing),
        ('model', RandomForestRegressor(n_estimators= es, random_state=0))
    ])

    scores= -1*cross_val_score(pipeline_model,
                             X_train,
                             y_train,
                             cv= 5, # nothing fancy just using 5 folds for cross validation
                             scoring= 'neg_mean_absolute_error')
    return scores.mean()

# getting 10 different mae(by cross validation) for 10 different estimator values
estimators_list=[50,100,150,200,250,300,350,400,450,500,550]
mae_list=[]
for es in estimators_list:
    mae_list.append(get_mae(es))

# identifying lowest mae and the estimator responsible for it
min_mae= min(mae_list)
index= mae_list.index(min_mae)
best_estimator= estimators_list[index]

# using the best estimator value to train our model.
best_model= Pipeline(steps=[
    ('preprocessor', preprocessing),
    ('best_model', RandomForestRegressor(n_estimators= best_estimator, random_state=0))
])
best_model.fit(X_train,y_train)

# using the best model to do the salary prediciton on our testing data(only contains features)
predicted_y= best_model.predict(X_test)

# making an output csv file containing the result of our model
errors = abs( y_test - predicted_y )
output_df= pd.DataFrame({ 'Index_number': X_test.index,
                         'Age': X_test['Age'],
                         'Gender': X_test['Gender'],
                         'Education Level': X_test['Education Level'],
                         'Job Title': X_test['Job Title'],
                         'Years of Experience': X_test['Years of Experience'],
                         'Actual Salary': y_test,
                         'Predicted Salary': predicted_y,
                         'Error': errors
                        })
output_df= output_df.sort_index()

output_csv_path= "Salary_predicition_model_5.0.csv"
output_df.to_csv(output_csv_path, index=False)

print(output_df.head())
print()
print('Lowest MAE: ', min_mae, ' Estimator responsible for it: ', best_estimator)


# In[ ]:





# In[ ]:




