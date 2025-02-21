import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')
#type of data

# 1 age integer
# 2 aneamie categorical 0-1
# 3 creatinine_phosphokinase integer
# 4 diabetes categorical 0-1
# 5 ejection fraction integer 
# 6 high_blood_pressure categorical 0-1
# 7 platelets integer
# 8 serum_creatinine real
# 9 serum_sodium integer
# 10 sex categorical 0-1
# 11 smoking categorical 0-1
# 12 time integer

def read_file(filename):

    data=pd.read_csv(filename,delimiter=None)
 
    return data

def transform_and_return_data(dataframe):

    
    labels=np.array(dataframe.columns)

    nan_indexes=dataframe.isnull()
    print("number of missing values: %d" %(sum(sum(np.array(nan_indexes)==True))))

    X=np.array(dataframe[labels[0:12,]])

    #Transform the caterorigal data

    le = LabelEncoder()
    le.fit(X[:,1])
    #print(X[:,1])

    X[:,1]= le.transform(X[:,1])


    le.fit(X[:,3])
    X[:,3] = le.transform(X[:,3])


    le.fit(X[:,5])
    X[:,5] = le.transform(X[:,5])

    le.fit(X[:,9])
    X[:,9] = le.transform(X[:,9])



    le.fit(X[:,10])
    X[:,10] = le.transform(X[:,10])

    y=dataframe[labels[12,]]


    le.fit(y)
    y = le.transform(y)  

    return X,y,labels

def desc_statistics():

 
    data_df=read_file('heart_failure_clinical_records_dataset.csv')
    labels=np.array(data_df.columns)

    data_df.columns = data_df.columns.str.strip()

    #Change the catagories in order to have the description

    #indexes of categorical variables

    ind_cat=[1,3,5,9,10,12]
    
    for i in ind_cat:
        
        
        data_df[labels[i]]=data_df[labels[i]].astype('category')

    print(data_df.describe())
    