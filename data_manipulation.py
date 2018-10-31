"""
This module contains functions for several operations
like data unzipping, computing statistics, data manipulation etc.
See each function for details. 
"""

import os
import json
import numpy as np
import pandas as pd
import zipfile36 as zipfile
from itertools import chain
from best_feature_extraction import rmse
import warnings
warnings.filterwarnings('ignore')

def extract_files(PATH):
    """This function extract data files from the zipped folder located in PATH"""
    # files path
    data_files = os.path.join(PATH, 'UseCase_3_Datasets.zip')
    
    # Unzipping files
    h = open(data_files, 'rb')
    obj = zipfile.ZipFile(h)
    for name in obj.namelist():
        if name in ['sales_granular.csv', 'Surroundings.json']:
            outpath = PATH
            obj.extract(name, outpath)
    h.close()
    
    
def feature_statistics(df, features):
    """ This function presents summary statistics on mean, standard deviation and sample size on selected variables 
    in the dataframe, df """
    tmp_df = pd.DataFrame()
    # iterate over each col
    for col in features:
        # group data into 2 based on col > 0 and compute summary statistics on sales_volume for each group. 
        tmp = df[[col, 'sales_volume']].groupby(df[col] > 0).agg([np.mean, np.std, len])['sales_volume'].T
        tmp = pd.DataFrame(np.array(tmp), index=tmp.index, columns = [[col, col],list(tmp.columns)])
        tmp_df = pd.concat([tmp_df, tmp], axis=1)
    tmp_df_T = tmp_df.T    
    tmp_df_T.columns = ['mean', 'std','sample size']
    
    return tmp_df_T

def take_exp(arr):
    """ This function reverses what take_log() does. see take_log """
    eps = 1e-1
    return np.exp(arr) - eps

def take_log(arr):
    """This function makes taking log convenient by adding a small number """
    eps = 1e-1
    return np.log(eps + arr)


def extract_json_data(PATH):
    "This function simply deserializes and returns json object "
    h = open(os.path.join(PATH, 'Surroundings.json'))
    # deserializing json data
    return json.load(h)    

def merge_data(features, sales):
    "This function merges features with the target data"
    # store_sales = pd.read_sql_query('SELECT * FROM avg_weekly_sales', conn)
    store_sales = sales.to_frame().reset_index()
    store_sales.columns = ['store_code', 'sales_volume']
    df = pd.merge(features, store_sales, on='store_code', how='inner')
    df = df.set_index('store_code')
    return df


def extract_sales_data(PATH):
    """This function load the sales data from the csv file
    In the stored form, each row in the data represents time-stamped sales from a store; 
    store is identified by a unique store_code and each column represents the time stamps
    with hourly resolution. 
    We transpose the data for convenient operations that we need for analysis. 
    This function returns a dataframe; each column represents a store and each row represents 
    timing stamp.     
    """
    # loading the dataframe from csv file
    data = pd.read_csv(os.path.join(PATH, 'sales_granular.csv'))
    
    # index and columns of new dataframe 
    index = pd.to_datetime(data.iloc[:,1:].T.index) # Timing info
    columns = data['store_code'] # store codes 
    # data in the new dataframe
    tmp_data = np.array(data.iloc[:,1:].T) # Transpose the data 
    
    # new dataframe
    data = pd.DataFrame(tmp_data, columns = columns, index = index)#.sum(axis = 1)
    data.columns.name = 'store_code'
    
    return data

def extract_daily_sales(data):
    """ This function extract daily sales """
    # Groupy the data for each day and sum the sales to obtain sales for the day. 
    daily_sales = data.groupby(data.index.date).sum()
    """ The grouping operation fills missing values with 0 that is problematic;
    since it removes the business start day; we assume the business start day
    as the first non-NA value. 
    Therefore, we need to demarcate business start day by restoring missing values
    at appropriate places. 
    """
    return demarcate_biz_start(daily_sales)


def extract_weekly_sales(data):
    """This function extracts weekly sales from hourly resolution data"""
    # We are just interested in the sales info; hence, we do not need fine timing details of the sales. 
    # We determine daily sales. 
    daily_sales = extract_daily_sales(data)
    # total number of weeks
    total_days = daily_sales.shape[0]
    total_weeks = total_days / 7
    # assigning week number to each day.     
    week_number = list(chain.from_iterable([[i]*7 for i in range(int(total_weeks))]))
    # grouping data by week and obtaining total sales in a week. 
    weekly_sales = daily_sales.groupby(week_number).sum()
    # demarcate business start week. 
    weekly_sales = demarcate_biz_start(weekly_sales)
    weekly_sales.index.name = 'week_numb'
    
    return weekly_sales

def demarcate_biz_start(df):
    """ The grouping operation fills missing values with 0 that is problematic;
    This function demarcates business start day by restoring missing values prior to business start day.
    """
    # We just make a copy of dataframe obtained after grouping operation
    zero_filled = df.copy()
    # We make all the instances with zero sales as missing values. 
    df[df == 0] = np.nan
    """ we forward fill the last dataframe;
    It would fill the missing values only if those missing values come after some sales;
    Indicating 0 sales for the day.
    Problem is that forward fill fills zero places with last sales. 
    """
    f_filled = df.fillna(method='ffill').copy()
    """By taking a difference of forward filled dataframe and zero filled data frame, 
    we can locate places with zero sales but have been filled with last sales """
    diff_df = f_filled - zero_filled
    """ By subtracting diff_df from the forward filled dataframe, we can restore zero sales at 
    appropriate places while retaining missing values prior to first day of sales. """
    df_demarcated = f_filled - diff_df
    
    # df_demarcated represents first day of sales as the first non-missing value in a column. 
    
    return df_demarcated 


def compute_rmse_base(train, test):
    """
    This function computes train and test RMSE of base model.
    We define base model as to spit out a constant prediction irrespective of inputs.
    The constant prediction is the mean of the target variable in train data. 
    """
    predictions = train['sales_volume'].mean()
    
    labels = train['sales_volume']  
    train_rmse = rmse(labels - predictions, 0)
    
    labels = test['sales_volume']  
    test_rmse = rmse(labels - predictions, 0)
    
    return train_rmse, test_rmse


def compute_rmse(model, features, train, test, is_log=False):
    """
    This function computes the train and test RMSE of specified model; 
    trained over given features. 
    This function is also capable of handling models that predict
    over log of the target variable that is specified with is_log parameter. 
    """
    
    # If model is based on log of target variable
    if is_log:
       #  take_log() just like np.log(); see take_log()
        model.fit(train[features], take_log(train['sales_volume']))
        # Since predictions are also in log, we need to inverse log. 
        train_predictions = take_exp(model.predict(train[features]))
        test_predictions = take_exp(model.predict(test[features])) 
    else:
        model.fit(train[features], train['sales_volume'])
        train_predictions = model.predict(train[features])
        test_predictions = model.predict(test[features])    
    
    # train and test labels
    train_labels = train['sales_volume']  
    test_labels = test['sales_volume']  
    
    # train and test rmse    
    train_rmse = rmse(train_labels , train_predictions)
    test_rmse = rmse(test_labels , test_predictions)
    
    return train_rmse, test_rmse
if __name__ == '__main__':
    print('This file is not run as a module')	
