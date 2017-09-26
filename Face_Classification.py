#Loading data
from scipy.io import loadmat
import pandas as pd
import numpy as np

class DataLoader(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""
    def __init__(self, dataset_name='imdb', dataset_path=None):
    
