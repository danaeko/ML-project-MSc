from read_data import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,FactorAnalysis, KernelPCA
import matplotlib.pyplot as plt
import os

datapath='Data_Cortex_Nuclear.xls'


class Init:
    
    def __init__ (self):

        
        data_df=pd.read_excel(datapath)

        
        self.X,self.y,self.labels=transform_and_return_data(data_df)
    