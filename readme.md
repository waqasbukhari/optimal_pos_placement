# Optimal POS placement

This repository contains case study for optimal POS placement. 

## Code files

The main analysis and resutls are developed in file main_analysis.py and the code is accompanied by 4 modules that develops different features used in main_analysis.py.
The modules are as follows:
* feature_engineering.py

* data_manipulation.py

* data_preprocessing.py

* best_feature_extraction.py

## Main analysis file


main_analysis.py is basically created out of a IPython notebook that is also enclosed as main_analysis.ipynb. 
The same analysis file can be better viewed in html format enclosed as main_analysis.html 

## Development language and modules 


The code is developed in python 3.7 and the following libraries are used. 

* import os

* import json

* import warnings

* import numpy as np

* import pandas as pd

* from flatdict import FlatDict

* import zipfile36 as zipfile

* from itertools import chain

* from namedlist import namedlist

* from sklearn.metrics import mean_squared_error

* from sklearn.preprocessing import StandardScaler

* from sklearn.model_selection import StratifiedShuffleSplit as split

* from sklearn.ensemble import BaggingRegressor

* from sklearn.base import TransformerMixin

* from sklearn.pipeline import Pipeline