"""
This module implements a class for data preparation and feature engineering from raw data. 
"""
import pandas as pd
from flatdict import FlatDict
from feature_engineering import *
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class DataSet:
    """This class, starting from raw data constructs a dataframe suitable for machine learning.
    This class covers all the aspects of data preparation, feature creation and feature engineering. 
    """
    def __init__(self, raw_data):
        # attribute raw_data holds unserialized JSON object
        self.raw_data = raw_data 
        # holds dataframe for machine learning obtained after processing raw data from JSON object
        self.df = pd.DataFrame() 
        
        "Feature creation methods"
        # process and add count of each amenity to the dataframe for ML.
        # Note argument to method add_dataframe() is also a function. 
        self.add_dataframe(self.extract_amenity_count_around_store) 
        # data is extracted from ordered list and hence, we can safely remove 
        # store_code to avoid duplicity.
        del self.df['store_code']
        
        # process and add ratings of each amenity to the dataframe for ML
        self.add_dataframe(self.extract_ratings_around_store)
        
        
        " type of amenities udner our consideration. "
        self.amenities = list(self.raw_data[0]['surroundings'].keys())
        
        # add variables for popularity i.e., how many amenities and how many types of amenities        
        self.add_popularity_variables()
        
        # adding binary, 2-way variables and removing variables with no variability 
        self.create_additional_features()
        
    def add_popularity_variables(self):
        """This function adds popularity features that 
        represents the total number of all amenities around store 
        and the count of the type of amenities. """
        features = self.amenities
        # How many amenities are there in the surroundings
        self.df['popularity'] = self.df[features].sum(axis=1)
        # A count on the type of amenities.
        self.df['convenience'] = (self.df[features] > 0).sum(axis=1)      
        
        
        
    def create_additional_features(self):
        """
        This function runs an sklearn pipeline and process each component of that pipeline.
        see module feature_engineering for details on the components on this pipeline 
        """
        feature_creation = Pipeline([
            ('BinaryFeatures', BinaryVariable(self.amenities)), # add binary variables
            ('TwoWayFeatures', TwoWayCombinations(self.amenities)), # add variables with 2-way interactions
            ('RemoveUniqueValueFeatures', RemoveUniqueVariable()) # remove variables with no variability
        ])
        
        # running the pipeline on our dataframe created so far. 
        self.df = feature_creation.fit_transform(self.df)
        
    def extract_dataframe(self):
        "Just returns the dataframe"
        return self.df
    
    def extract_amenities(self):
        "Just returns the amenities that we consider so far"
        return self.amenities
        
        
    def extract_rating(self, amenity_list):
        """ For a given store and a particular amenity, 
        this function computes the total and average rating for that amenity. 
        This object appears as a list and each element of that list refers to 1 instance 
        of that amenity. 
        The function calculates the total number of ratings across all the components 
        (i.e., all instances of that amenity around the store) and accumulates the total 
        rating score. 
        Finally it returns the total number of ratings and average ratings. """
        
        total_ratings, avg_rating, total_score = 0,0,0
        if len(amenity_list) == 0: # The store does not have that amenity around it 
            return total_ratings, avg_rating 
        
        # looping over each amenity.
        for i in range(len(amenity_list)): 
            """flatting the nested dictionary that contains multiple elements of an
            instance of amentiy.
            In the flattened dictionary, 'user_ratings_total' is the total number of ratings 
            and 'rating' is the average rating for the amenity. 
            Total score is obtained by multiplying these two components. """ 
            
            tmp_dict = FlatDict(amenity_list[i]) 
            if ('user_ratings_total' in tmp_dict) and ('rating' in tmp_dict): 
                total_ratings += tmp_dict['user_ratings_total']
                total_score += tmp_dict['user_ratings_total'] * tmp_dict['rating'] 
                
        # adding a small value in the denominator to avoid ZeroDivisionError Exception
        return total_ratings, total_score / (1e-6 + total_ratings)   
    
    
    def extract_ratings_around_store(self, store_data):
        """ This function extracts ratings of each amenity for a given store 
        and returns as a dictionary. 
        Each amenity has 2 keys in this dictionary; 
        one corresponding total ratings and 
        the other corresponding to average ratings. 
        """
        
        store_ratings = dict()
        # Storing store code for ID
        store_ratings['store_code'] = store_data['store_code'] 
        # Data for amenities
        d_amenities = store_data['surroundings']
        
        # Loop over each amenity and extract its rating. 
        for amenity, amenity_data in d_amenities.items():
            total_ratings, avg_rating = self.extract_rating(amenity_data)
            
            # note the naming conventions
            store_ratings[amenity + '_rating_count'] = total_ratings
            store_ratings[amenity + '_avg_rating'] = avg_rating
            
        return store_ratings   
    
    def extract_amenity_count_around_store(self, store_data): 
        """ This function extracts count of each amenity for a given store 
        and returns as a dictionary. 
        """
        
        store_count = dict()
        # Storing store code for ID
        store_count['store_code'] = store_data['store_code'] 
        # Data for amenities
        d_amenities = store_data['surroundings']
        
        # Loop over each amenity and extract its rating. 
        for amenity, amenity_data in d_amenities.items():
            # len(amenity_data) is the count on that amenity
            store_count[amenity] = len(amenity_data)
            
        return store_count
    
    def add_dataframe(self, f):
        """ This function takes a function as its input 
        and process the raw data according to input function to extract list of records for 
        each store
        and creates a dataframe out of extracted records 
        and join the dataframe to the existing dataframe for machine learning"""
        
        # maintains record for each store. 
        list_of_records = []
        # loop over each store
        for record in self.raw_data:
            # extract data corresponding to a store according to function f. 
            extracted_data = f(record)
            list_of_records.append(extracted_data)
            
        # creates dataframe for the extracted records
        records_df = pd.DataFrame(list_of_records)
        # adds to the existing dataframe fro machine learning. 
        self.df = pd.concat([self.df, records_df], axis = 1)
        
if __name__ == '__main__':
    print('This file is not run as a module')