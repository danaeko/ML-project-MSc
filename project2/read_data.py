import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def read_file(filename):

    data=pd.read_csv(filename,delimiter=None)
  
    return data

def transform_and_return_data(dataframe):

    
    labels=np.array(dataframe.columns)

    data=dataframe.values

    nan_indexes=dataframe.isnull()


    print("number of missing values: %d" %(sum(sum(np.array(nan_indexes)==True))))
    imputer = SimpleImputer(missing_values=np.nan,
                                strategy='mean', fill_value=None)

    imputed_data=imputer.fit_transform(data[:,1:78])

        
    X=imputed_data[:,:78]

    print(X.shape)
    y=data[:,81]


    le = LabelEncoder()
    le.fit(y)
    y_train_enc = le.transform(y)
    scaler = StandardScaler()
    scaler.fit(X)


    le.fit(y)
    y = le.transform(y)  


    return X,y,labels[1:]

def desc_statistics():

 
    data_df=pd.read_excel('Data_Cortex_Nuclear.xls')
    labels=np.array(data_df.columns)

    data_df.columns = data_df.columns.str.strip()

    #Change the catagories in order to have the description

    #indexes of categorical variables

    
    ind_cat=[77,78,79,80]

    
    for i in ind_cat:
        
        print(labels[i])
        data_df[labels[i]]=data_df[labels[i]].astype('category')
    print(data_df.describe(include=['category']))
    print(data_df.describe())
    
    data_df[labels[:8]].describe().to_csv('desc1.csv')
    data_df[labels[16:24]].describe().to_csv('desc2.csv')
    data_df[labels[24:32]].describe().to_csv('desc3.csv')
    data_df[labels[32:40]].describe().to_csv('desc4.csv')
    data_df[labels[40:48]].describe().to_csv('desc5.csv')
    data_df[labels[48:56]].describe().to_csv('desc6.csv')
    data_df[labels[56:64]].describe().to_csv('desc7.csv')
    data_df[labels[64:72]].describe().to_csv('desc8.csv')
    data_df[labels[72:]].describe().to_csv('desc9.csv')

desc_statistics()
