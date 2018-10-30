"""This module implements classes and independent functions related to feature extraction
module of our work. 
To be specific, this module helps identify handful of best features out of humongous number
of features; created from raw data """

import numpy as np
import pandas as pd
from namedlist import namedlist
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit as split
from sklearn.ensemble import BaggingRegressor
# from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


def rmse(labels, predictions):
    """
    This function returns the root mean squared error.
    Note: If predictions is an integer, we interpret it as a constant prediction. 
    """
    
    """ If predictions is an integer, we make it a array to comply with sklearn API """
    if isinstance(predictions, int):
         # create an array same as labels and fill it with constant prediction
        predictions = np.full(labels.shape, predictions)
        
    """ mean_squared_error is an error metric; imported from sklearn. """
    mse = mean_squared_error(labels, predictions)
    return np.sqrt(mse)

    
def train_test_split(data, test_ratio = 0.5, n_splits=10, best_split = True):   
    """ 
    This function splits the data into two using stratified sampling in ratios as determined by test_ratio.
    The strata is constructed out of creating quartile splits on the target variable, i.e., sales_volume.
    If the best_split is True, The split which yields the minimum differen in the means of target is 
    returned. Else, the last split is returned as it is. 
    Note the number of splits are determined by n_splits"""
    
    # Object for stratified sampling 
    split_obj = split(n_splits=n_splits, test_size=test_ratio, random_state=180)
    # Discretizing the target volume to guide stratified sampling
    data['categories'] = pd.qcut(data['sales_volume'], 4, labels=["low", "low mid",'high mid',"high"])
    
    # best split is one that yields least difference in mean sales_volume of both folds
    least_diff_in_means = None
    best_split = None, None
    # Looping over each split
    for idx_train, idx_test in split_obj.split(data, data['categories']):
        train = data.iloc[idx_train]
        test = data.iloc[idx_test]
        
        diff_in_means = abs(train.sales_volume.mean() - test.sales_volume.mean())
        """ Update the best split if best_split=True and 
        either the current split is the first split or the best split.
        """ 
        if best_split and ((least_diff_in_means is None) or (least_diff_in_means > diff_in_means)):
            least_diff_in_means = diff_in_means
            best_split = idx_train, idx_test
            
    if best_split[0] is None:
        best_split = idx_train, idx_test
            
    del data['categories']
    idx_train, idx_test = best_split 
    
    train = data.iloc[idx_train]
    test = data.iloc[idx_test]
    
    return train, test


class IterVarModel:
    """ This class iteratively find best features one by one iteratively; starting from no features """
    
    """ 
    At a particular iteration, all the candidate features are evaluated and the features that 
    yields the best 2-fold cross validation performance are added to the model (best features). 
    Number of best features to extract is determined by max_features_to_extract. 
    If none of the features improves performance beyond already obtained in the previous 
    iteration, The feature search process stops even before finding max_features_to_extract features.      
    """
    
    """
    This class maintains two folds for performance evaluation and comparison. 
    """
    
    "Train fold 1 and evaluate fold 2 and call it performance over fold 2"
    "Train fold 2 and evaluate fold 1 and call it performance over fold 1"
    
    "Note that a specified model is used for all kind of training, testing purposes."
    
    class RMSEFolds:
        """ A nested class that we define to maintain and compare RMSE results over both folds """
        def __init__(self, rmse_1, rmse_2):
            self.fold_1 = rmse_1 # RMSE over fold 1
            self.fold_2 = rmse_2 # RMSE over fold 2
            
        def __lt__(self,other):
            """ 
            __lt__ is a Special method that can define < operator on class instances. 
            We define RMSE1 < RMSE2 if and only if the RMSE1 is strictly lower than RMSE2 
            in both the folds.
            """
            # defining < condition. 
            # Condition 1 - RMSE_1 < RMSE_2 if results of both folds in RMSE_1 are less than that in RMSE_2
            # cond_1 = (self.fold_1 < other.fold_1) and (self.fold_2 < other.fold_2)
            # Condition 2 - RMSE_1 < RMSE_2 if the sum of rmse in both folds of RMSE_1 is less than that in RMSE_2
            # cond = (self.fold_1 ** 2 + self.fold_2 ** 2) < (other.fold_1 ** 2 + other.fold_2 ** 2) 
            cond = (self.fold_1 < other.fold_1) and (self.fold_2 < other.fold_2)
            # RMSE_1 < RMSE_2 if either condition is true
            
            return cond# _1 or cond_2
        
   
    # Special method that gets run on object instantiation. 
    def __init__(self,  data, model, max_features_to_extract):
        # data over which we create folds and extract best features. 
        self.data = data
        # maximum feautres to extract
        self.max_features_to_extract = max_features_to_extract
        # model to be used in feature evaluations
        self.model = model 
        # input columns are all the columns in the dataframe data except the target 
        self.input_variables = [col for col in self.data.columns if col not in ['sales_volume']]
        # maintaining data for the folds. This attribute holds data related to folds. 
        self.folds = None 
        
        # Maintains a list of useful features
        self.extracted_features = [] 
        # Stops the feature extraction process if it becomes True
        self.stop_feature_extraction = False 
        # create 2 folds out of all the data. Basically, split data into folds and also create additional variables. 
        self.create_folds() 
        
        
        
    def standardize_folds_inputs(self):
        """ Standardize inputs in fold 1 from parameters obtained in fold 2 and 
        Standardize inputs in fold 2 from parameters obtained in fold 1 
        Logic: test data cannot know her own mean and variance;
        hence has to be standardize with training set parameters
        """
        
        fold_1_X = self.folds[1].input
        fold_2_X = self.folds[2].input
        
        # get parameters from fold 1 and standardize and update fold 2
        model = StandardScaler() # standard scalar
        model.fit(fold_1_X) # get parameters from inputs in fold 1
        self.folds[2]._update(input=pd.DataFrame(model.transform(fold_2_X), columns=fold_1_X.columns)) # transform inputs in fold 2
        
        # get parameters from fold 2 and standardize and update fold 1
        model = StandardScaler() # standard scalar
        model.fit(fold_2_X) # get parameters from inputs in fold 2
        self.folds[1]._update(input=pd.DataFrame(model.transform(fold_1_X), columns=fold_1_X.columns)) # transform inputs in fold 1

    def add_data_in_folds(self):
        data = self.data
        # We use stratified sampling to split the data; see function train_test_split() for details
        fold_1_data, fold_2_data = train_test_split(data, test_ratio = 0.5, n_splits=10, best_split = True)
        # inputs
        input_features = self.input_variables
        
        ## Now we add inputs and outputs to each fold.
        # update inputs        
        self.folds[1]._update(input=fold_1_data[input_features])
        self.folds[2]._update(input=fold_2_data[input_features])
        
        # update outputs        
        self.folds[1]._update(output=fold_1_data['sales_volume'])
        self.folds[2]._update(output=fold_2_data['sales_volume'])
        
        
    def create_folds(self):
        """
        This function uses stratified sampling to split data into two equal-sized folds
        and maintains these folds using class attribute of folds. 
        We use namedlist; one for each fold to hold its data 
        """
        
        """
        namedlist is a factory function for creating mutable collections of list items;
        it is similar to python's list but enables us to name each component and access using
        dot notation.
        """
        Fold = namedlist('Fold', 'input output rmse')
        
        """
        class attribute folds is a dictionary with 2 keys;
        key=1, refers to namedlist that holds data related to fold 1
        key=2, refers to namedlist that holds data related to fold 2
        """
        self.folds = dict()        
        for i in [1,2]:
            self.folds[i] = Fold(input=None, output=None,rmse=None)
            
        # add inputs and outputs to the folds by intelligently splitting data; see class method add_data_in_folds()
        self.add_data_in_folds()
        # Standardize inputs in the folds for better ML performance; see class method standardize_folds_inputs()         
        self.standardize_folds_inputs()
        
        """
        Now after having inputs and outputs in both folds, we update RMSE.
        As of now, we have not extracted any feature.
        Hence, we consider a base model i.e., one that spits out mean of its training target. 
        """
        # predictions of base model over fold 1 is a constant; mean of target variable in fold 2
        # predictions of base model over fold 2 is a constant; mean of target variable in fold 1
        # updating RMSE based on this logic. 
            
        self.folds[1]._update(rmse=rmse(np.abs(self.folds[1].output - self.folds[2].output.mean()), 0))
        self.folds[2]._update(rmse=rmse(np.abs(self.folds[2].output - self.folds[1].output.mean()), 0))
            
    
    def eval_fold(self, eval_fold_number, features):
        """
        This function evaluates a fold specified by eval_fold_number based on features
        and returns RMSE 
        """
        "fold 1 is evaluated by training over fold 2 and evaluating over fold 1"
        train_fold = 1 if eval_fold_number == 2 else 2
        test_fold = 2 if eval_fold_number == 2 else 1
        
        model = self.model
        # training data from train_fold
        X, Y = self.folds[train_fold].input[features], self.folds[train_fold].output
        # learning
        model.fit(X, Y)
        
        # test data 
        test_X, test_Y = self.folds[test_fold].input[features], self.folds[test_fold].output
        # prediction
        test_predict = model.predict(test_X)
        # evaluate predictions and compute rmse        
        tmp_rmse = rmse(test_Y, test_predict)
        
        return tmp_rmse
        
    def is_new_feature_good(self, features):
        """
        This function evaluates fold 1 and fold 2 with features
        and determines if features leads to better performance 
        compared to extracted best features. 
        """
        # class method eval_fold() is used to evaluate a fold and returns RMSE. see eval_fold()  
        rmse_1 = self.eval_fold(1, features)
        rmse_2 = self.eval_fold(2, features)
        
        # Construct an RMSE object comprising RMSEs of folds resulted from current features. 
        RMSE = self.RMSEFolds(rmse_1, rmse_2)
        
        ## Construct an RMSE object comprising RMSEs of folds resulted best features obtained so far
        RMSE_current = self.RMSEFolds(self.folds[1].rmse, self.folds[2].rmse)
        
        result = False 
        """if RMSE is better than RMSE_current, then set of variables in features are better 
        than ones in best features extracted so far
        """
        if RMSE < RMSE_current:
            # do advanced analysis on residuals. 
            result = True
        
        return result, RMSE
        
    def add_var(self):
        """
        This method search for the variable; if such a variable exists that when
        added can improve performance
        """
        # 
        # We define best_RMSE to be RMSE reached in previous iteration. 
        best_RMSE = self.RMSEFolds(self.folds[1].rmse, self.folds[2].rmse)
        # Initially a None, best_var indicates the candidate variable that can be included. 
        best_var = None # maintain the best variable found in this iteration
        
        # Looping over the candidate variables        
        for col in self.input_variables:
            # candidate variable should not be already in the best extracted features. 
            if col not in self.extracted_features:
                # make a temporary list of features by adding candidate feature to the existing best features. 
                tmp_features = self.extracted_features + [col]
                """
                Evaluate the goodness of candidate feature;
                see class method is_new_feature_good() for further details. 
                is_good=True indicates that the candidate variable can improve the performance. 
                """
                is_good, RMSE = self.is_new_feature_good(tmp_features)
                # Update the best_var if is_good=True and RMSE is better than best RMSE so far. 
                if (is_good and (RMSE < best_RMSE)):
                    best_RMSE = RMSE
                    best_var = col
                    
        # If we find a variable that can improve performance.
        if best_var is not None:
            print('adding_variable: {}'.format(best_var))
            self.extracted_features.append(best_var)
            
            # Update rmse and residuals
            self.folds[1]._update(rmse=best_RMSE.fold_1)
            self.folds[2]._update(rmse=best_RMSE.fold_2)
            
        # If we cannot find a variable that can improve performance.
        else:
            # Turning this to True stops feature extraction. 
            self.stop_feature_extraction = True
            # print('new features cannot be added')
            
        
        
    def extract_features(self):
        """
        This function runs feature extraction routines until either
        the maximum allowed features is reached or when none of the 
        variables can improve the performance by getting added. 
        """
        for _ in range(self.max_features_to_extract):
            # running until stop_feature_extraction=False
            if not self.stop_feature_extraction: 
                # add_var() is a function that looks for best variable to add; see method add_var() for details. 
                self.add_var()
        
        return self.extracted_features
    

def extract_model_features(model, df, max_features=5):
    """ 
    This function extracts features in the dataframe df based on specified model;
    this function initializes instance of class IterVarModel with specified model
    and the number of features to extract. 
    It uses class method extract_features() to extract best features and returns 
    these features 
    """
    # class instance. 
    tmp = IterVarModel(df, model, max_features)
    return tmp.extract_features()



def bagging_feature_extraction(model, df, max_features):
    """
    This function firstly defines a bagging model with specified base model
    and uses this bagging model to extract features using function extract_model_features()
    Bagging helps shortlisting best features and avoids unstable features. 
    base model is subsequently used over the extracted features to extract features. 
    """
    # Bagging model with each estimator utilizing 60% of the data; adding some randomness. 
    bagging_model = BaggingRegressor(base_estimator=model, max_samples=0.6, random_state=25)
    # Extract bagging features 
    print('-'*18)
    print('Bagging Features')
    print('-'*18)
    bagging_features = extract_model_features(bagging_model, df, max_features=15) # 15
    # Modify the dataframe to include only the bagging features. 
    features_to_retain = bagging_features + ['sales_volume']
    df = df[features_to_retain]
    # feature extraction from the remaining features using base model
    print('-'*18)
    print('Final Features')
    print('-'*18)
    base_model_features = extract_model_features(model, df, max_features=max_features)
    return base_model_features

def feature_extraction(models, df, max_features = 5, precede_bagging=False):
    """
    This function uses multiple base models; stored in dictionary models and runs feature extract for each. 
    It may or may not include bagging based on precede_bagging.
    The record for each model is stored in a namedlist. 
    The function returns a list of namedlists; each holding a record for a model. 
    """
    # A named list to maintain data for each model
    """
    namedlist is a factory function for creating mutable collections of list items;
    it is similar to python's list but enables us to name each component and access using
    dot notation.
    """
    Model = namedlist('Model', 'name sklearn_form extracted_features')
    # a list of model features is created in which each component correspond to a model. 
    # list to hold namedlists. 
    models_features = []
    # iterate over each model in models
    for model_name, model in models.items():
        print('\n')
        print('*'*18)
        print(model_name)
        print('*'*18)
        # If the features list should be made smaller with bagging
        if precede_bagging:
            tmp_features = bagging_feature_extraction(model, df, max_features)
        else:
            tmp_features = extract_model_features(model, df, max_features=max_features)
            
        # creating namedlist to store records for the model. 
        tmp_model = Model(name=model_name, sklearn_form=model, extracted_features=tmp_features)
        # adding record to the list
        models_features.append(tmp_model)
        
    return models_features
    
if __name__ == '__main__':
    print('This file is not run as a module')