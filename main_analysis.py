#!/usr/bin/env python
# coding: utf-8

# ## Libraries
print('Loading libraries')

import random
import pandas as pd
import numpy as np
from namedlist import namedlist
import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score
import warnings

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor


from data_manipulation import *
from data_preprocessing import *
from best_feature_extraction import *


print('Locating and Unzipping data files')
# data directory
data_dir = '.\data'
# extract_files() is a function in data_manipulation module that simply unzips files
extract_files(data_dir)


print('Loading sales data') 
data = extract_sales_data(data_dir)
print(data.iloc[:5,:10])# head()
# ### Weekly sales
"""
Note that the data in raw form is at hourly resolution.
In this resolution, too many data points are missing.
While we believe amenities features can predict sales, 
predicting at hourly scale is a big ask.
Therefore, we accumulate data over a week. 
"""
weekly_sales = extract_weekly_sales(data)
# weekly_sales.iloc[[0,1,2,3,95,96,97,98], :10]
print(weekly_sales.iloc[-5:,:10])# .head()
## 'Trend in Sales and new stores'
"""
In order to design a target variable that can be explained with the amenities around POS,
it is imperative to understand the relative trends in the sales. 
"""
# We compute a time series of overall sales per week across all the stores. 
total_sales_across_time = weekly_sales.sum(axis=1)
"""
We compute the total number of stores in operation at a particular week. 
We assume that once a sale is made at a store, it has started its operation
and does not halt its operation. (*a store is never closed*).
See demarcate_biz_start() function in data_manipulation module for 
details on how we demaracate business start date. 
"""
numb_stores_across_time = pd.notna(weekly_sales).sum(axis=1)

"""
We plot 2 time series;
firstly, the total sales across all stores and
secondly, the number of stores in operation. 
From the figure, we find that the our sales operations are successful
and management opens up a lot of new stores to cope up with the sales. 
"""

fig, ax1 = plt.subplots()
color = 'black'
ax1.set_xlabel('Week Number')
ax1.set_ylabel('Weekly Sales', color=color)
ax1.plot(total_sales_across_time.index, total_sales_across_time, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'blue'
ax2.set_ylabel('Number of Stores', color=color)  # we already handled the x-label with ax1
ax2.plot(total_sales_across_time.index, numb_stores_across_time, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.show()
fig.savefig('sales_and_stores.png')

# ### Sales trend of Stores
"""
While we observe huge increase in the overall sales, 
that might be deceptive and might owe to larger number of
stores.
To understand it, we take a look at the sales trend in individual stores. 
"""

# seed for reproducibility. 
random.seed(0)
# stores list
stores = weekly_sales.columns.tolist()
# Pick random stores
selected_stores = random.sample(stores, 5)
# Fixing the store_code for reproducibility across systems
selected_stores = [75053] # [selected_stores[0]]

# Computing overall mean weekly sales of the selected store
mean_value = pd.DataFrame([weekly_sales[selected_stores].mean()] * 99, index=weekly_sales.index, columns=selected_stores)
"""
Simple arithmetic mean does not indicate any trends and just averages any trends out.
If the trends change, we should be interested and our target variable should be based 
on recent trends.
exponential moving average is one such measure that can capture trends by weighting
recent samples more. and Apart from 
with ewma(t) = alpha * x(t) + (1-alpha) * ewma(t-1), alpha is a design choice.
Roughly, a measurement (1/alpha) time units (weeks in this case) in the past 
get a weight that is (1/3) of the weight of the most recent measurement. 
We set alpha=0.2, implying impact of weeks older than 5 from now is less than (1/3) of current week. . 
"""
exp_mean_value = weekly_sales[selected_stores].ewm(alpha = 0.2, adjust=False).mean()


### Plotting weekly sales in the store, overall mean and ewma. 

fig, ax = plt.subplots( nrows=1, ncols=1 ) 
ax.plot(weekly_sales[selected_stores], color='blue')
ax.plot(mean_value, color='black')
ax.plot(exp_mean_value, color='red')
ax.legend(['Weekly Sales', 'Mean Weekly Sales', 'Exp. weighted Mean Sales'], loc='upper left')

plt.xlabel('Week Number')
plt.ylabel('Sales')
plt.title('Comparison of simple and exponential weighted average sales')
plt.show()
fig.savefig('sales_trends_at_store.png')


### Selecting target

"""Based on these observations, find that exponentially weighted average tracks the recent trend and hence can be a good target. 
This measure reflect recent trends while also being able to track steady sales. """


""" ewma can be susceptible to initial conditions and it takes a while to get away with the effects of initial conditions.
Based on this, stores new into the operation can have quite faulty sales estimates """
# Finding the least time an arbitrary store started operation. 
print('Store with least time: Maximuum weeks in operation ',weekly_sales.notna().sum().min())

"""
There is at least one stores that opened just 1 week prior to the data close and hence would be biased by initial values. 
We can circumvent it simply by backward filling the weekly sales data frame by 15 places 
to subside the effects of initialization.  
with alpha = 0.2, the weight of the 15th value preceeding the current value is 0.8 ^ 15 ~ 0.035 times the current weight.
 """
weekly_sales.fillna(method = 'bfill', limit = 15, inplace = True)
## Computing ewm and taking the last value that would server as our target variable. 
sales_volume = weekly_sales.ewm(alpha = 0.2, adjust=False).mean().iloc[-1]


print('Data preparation for ML')
# ## Data preparation for ML
# ### Loading JSON data
# extracting json data. See function extract_json_data() in module data_manipulation
json_data = extract_json_data(data_dir)
# ### Feature extraction
# Creating a class instance for getting processed data for machine learning
processed_data = DataSet(json_data)
# This function extracts the dataframe
features_df = processed_data.extract_dataframe()
# This returns the list of amenities that we are considering to rate for sales
# amenities = processed_data.extract_amenities()
# ### Combining features with the sales volume (target)
df = merge_data(features_df, sales_volume)
print(df.head())
print(df.shape)


print('Train-Test split')
# ### Creating a train-test split

# stratefied sampling; see function train_test_split() in module best_feature_extraction
train, test = train_test_split(df, test_ratio = 0.2, n_splits=1, best_split = False) 

print('Running feature extraction module wiht simple models')
# ### Extracting best features using different models
"""
A dictionary of models_name and models_sklearn representation.
We use these models in our work. 
"""
models = {'LinearRegression':LinearRegression(), 
          'DecisionTreeRegressor':DecisionTreeRegressor(random_state=25), 
          # 'RandomForestRegressor':RandomForestRegressor(), 
          # 'AdaBoostRegressor':AdaBoostRegressor(), 
          # 'SVR':svm.SVR(),
          # 'SVR(kernel="linear")':svm.SVR(kernel="linear"),
         }


# Extracting features using base models
models_features = feature_extraction(models, train)
# Displaying feature statistics
# extracted features using linear regression. index 0 stores linear regression
features = models_features[0].extracted_features
"feature_statistics() is a function that displays statistics on features; see in module data_manipulation"
print('Statistics of features extracted with linear regression')
print(feature_statistics(train, features))

# Statistics on Decision trees

features = models_features[1].extracted_features
print('Statistics of features extracted with Decision Trees')
print(feature_statistics(train, features))


print('Running feature extraction module wiht simple models modeling log of sales volume')

# ### Predicting in the log scale
# ### Log scale
# 
# The features extracted in the linear scale are susceptible to learning features that can discriminate stores with very high sales. Since the stores with very high sales are not well represented and very well be outliers. We need to predict in the log space. 
# taking log of the target variable
train['sales_volume'] = take_log(train['sales_volume'])
models_features = feature_extraction(models, train )
train['sales_volume'] = take_exp(train['sales_volume'])


features = models_features[0].extracted_features
print('Statistics of features extracted with linear regression')
print(feature_statistics(train, features))


features = models_features[1].extracted_features
print('Statistics of features extracted with Decision Trees')
print(feature_statistics(train, features))


"""
Features extracted from either linear regression or Decision trees;
trained over in linear or log target variable, 
we find that unstable variables are extracted. 
To circumvent it, we use bagging with simple models to extract features. 
"""

# ###  Extracting features using Bagging followed by simple models (linear output)
print('Features extracted using simple models whether in linear or log of sales volume are unstable')
print('Feature extraction using simple models following by bag of simple models')

train, test = train_test_split(df, test_ratio = 0.2, n_splits=1, best_split = False) 
# See function feature_extraction() in module best_feature_extraction for info into how bagging is done. 
models_features_linear = feature_extraction(models, train, precede_bagging=True)


features = models_features_linear[0].extracted_features
print('Statistics of features extracted with linear regression')
print(feature_statistics(train, features))


features = models_features_linear[1].extracted_features
print('Statistics of features extracted with Decision Trees')
print(feature_statistics(train, features))


# ###  Extracting features using Bagging followed by simple models (log output)
print('Feature extraction using simple models following by bag of simple models in log of sales_volume')
# Same bagging but output is log 
train, test = train_test_split(df, test_ratio = 0.2, n_splits=1, best_split = False) 
train['sales_volume'], test['sales_volume'] = take_log(train['sales_volume']), take_log(test['sales_volume'])
models_features_log = feature_extraction(models, train, precede_bagging=True)
train['sales_volume'], test['sales_volume'] = take_exp(train['sales_volume']), take_exp(test['sales_volume'])

features = models_features_log[0].extracted_features
print('Statistics of features extracted with linear regression')
print(feature_statistics(train, features))

features = models_features_log[1].extracted_features
print('Statistics of features extracted with Decision Trees')
print(feature_statistics(train, features))


print('Running model evaluations')
# ## model evaluation
# 
# So, we have trained 4 models, let us evaluate them along with a base model to make conclusions about best model. 
# We define base model as one which predicts mean of the target variable in training set. 

train, test = train_test_split(df, test_ratio = 0.2, n_splits=1, best_split = False) 


model_performance = []

# Base model
model_performance.append(['base model', *compute_rmse_base(train, test)])
# Linear Regression (Linear output)
model = models_features_linear[0].sklearn_form
features = models_features_linear[0].extracted_features
model_performance.append(['linear regression - Y', *compute_rmse(model, features, train, test)])
# Decision Trees (Linear output)
model = models_features_linear[1].sklearn_form
features = models_features_linear[1].extracted_features
model_performance.append(['Decision Tree - Y', *compute_rmse(model, features, train, test)])
# Linear Regression (Log output)
model = models_features_log[0].sklearn_form
features = models_features_log[0].extracted_features
model_performance.append(['linear regression - log(Y)', *compute_rmse(model, features, train, test, is_log=True)])
# Decision Trees (Log output)
model = models_features_log[1].sklearn_form
features = models_features_log[1].extracted_features

model_performance.append(['Decision Tree - log(Y)', *compute_rmse(model, features, train, test, is_log=True)])

# A data frame showing the performance of all four models. 
performance_df = pd.DataFrame(model_performance, columns=['Model','Train RMSE', 'Test RMSE'])
print(performance_df)


print('Best model')
# ## Best Model
# 
# Linear regression fitting linear sales volume is determined to be the best model. 

model = models_features_linear[0].sklearn_form
features = models_features_linear[0].extracted_features
model.fit(train[features], train['sales_volume'])

# A dataframe showing the regression coefficients of best model
model_coefs = pd.Series(model.coef_, index=features).to_frame()
model_coefs.columns = ['Regression Coefficient']

print(model_coefs)

print('statistics of extracted features in train data')
print(feature_statistics(train, features))

print('statistics of extracted features in test data')
print(feature_statistics(test, features))
