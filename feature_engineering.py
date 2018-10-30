"""
This module add classes; compliant with sklarn API.
These classes create new features on top of already created features.
One class RemoveUniqueVariable inspects and removes features with no variability. 
"""

import pandas as pd
"""
Inheriting from class TransformerMixin implicity implements fit_transform() method 
given fit() and transform are implemented 
"""
from sklearn.base import TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class BinaryVariable( TransformerMixin):
    """ This class creates binary variables indicating
    whether an amenity exist around a store or not. 
    This class conforms to the SKLearn API"""
    
    def __init__(self, features, binary_vars = True):
        # variable to indicate whether to include these binary variables
        self.add_binary_vars = binary_vars 
        # Variables which needs to be made binary. 
        self.features = features
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        ## We add features that will be indicated by a prefix 'has_'
        bin_columns = ['has_'+ c for c in self.features]
        # Create a boolean and then cast it as int
        bin_X = pd.DataFrame(X[self.features] > 0).astype('int')
        bin_X.columns = bin_columns
        
        # adding the binary dataframe to the original data. 
        return pd.concat([X, bin_X], axis = 1)

class TwoWayCombinations(TransformerMixin):
    """ This class creates 2-way features indicating
    an interaction of 2 variables. 
    This class conforms to the SKLearn API"""
    
    
    def __init__(self, features, add_two_way = True):
        # variable to indicate whether to include these 2-way interaction variables
        self.add_two_way_interactions = add_two_way
        # Variables over which 2-way interactions has to be constructed. 
        self.features = features
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        """We construct these 2-way interactions over binary features
        created in BinaryVariable class"""
        
        binary_features = ['has_'+ c for c in self.features]
        
        two_way_X = pd.DataFrame()
        
        # iteration over each combination of 2 features
        for i in range(len(binary_features)):
            c1 = binary_features[i]
            for j in range(i+1, len(binary_features)):
                c2 = binary_features[j]
                
                # creating a temporary dataframe comprising only two features under consideration. 
                tmp = X[[c1,c2]]
                # creating a 2-way interaction feature
                two_way_feature = (tmp.sum(axis = 1) // 2).to_frame()
                """naming the new feature that is just a combiantion of features 
                out of which it is constructed ."""
                two_way_feature.columns = [c1 + '_and_' + c2]
                
                two_way_X = pd.concat([two_way_X, two_way_feature], axis = 1)
        ## Adding 2-way features to the existing dataframe
        return pd.concat([X, two_way_X], axis = 1)
    
class RemoveUniqueVariable( TransformerMixin):
    """ This class removes all variables that has only 1 unique value. 
        It also conforms to SKLearn API """
        
    
    def __init__(self):
        
        self.features = None
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        # Feature over which to iterate. 
        # all are valid features except store_code
        self.features = [col for col in X.columns if col not in ['store_code']]
        # Iterating over each column
        for c in self.features:
            tmp = X[c]
            # If the number of unique elements is less than 2, delete the feature. 
            if len(tmp.unique()) < 2: 
                del X[c]
        return X
    
if __name__ == '__main__':
    print('This file is not run as a module')