from init import *
from read_data import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
import pickle
import glob,os,copy


columns_names_csv=['Model', "Method",'Num of Best Features', 'Performance full', 'Performance selected']


try:
    X
except NameError:
    init=Init()

    X=init.X
    y=init.y
    labels=init.labels




## read all the saved models filenames


metric_lists=['accuracy','f1_macro']




## read all the saved features with un feat sel

dict_mod_un_feat=glob.glob('best_features'+os.path.sep+'*un.sav')

feat_un=[]

## Load the saved features

for filename in dict_mod_un_feat:

    feat_un.extend(pickle.load(open(filename, 'rb')))


## read all the saved features with un feat sel

dict_mod_rfe_feat=glob.glob('best_features'+os.path.sep+'*rfe.sav')


dict_labels_un={}

for ll in labels[:77]:

    dict_labels_un.update({ll:0})


dict_labels_rfe={}

for ll in labels:

    dict_labels_rfe.update({ll:0})



#We use the same random_state as with model selection
#There we train and validated only with the 80%
#Here we train with the 80% and we test with the remaining 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)





for metric in metric_lists:
    print("Testing metic %s "%metric)

    dict_mod_un_feat=glob.glob('best_features'+os.path.sep+metric+os.path.sep+'models'+os.path.sep+'*un.sav')
    dict_mod_rfe_feat=glob.glob('best_features'+os.path.sep+metric+os.path.sep+'models'+os.path.sep+'*rfe.sav')
    feat_un=[]

    ## Load the saved features

    dict_mod_filenames=glob.glob('best_models'+os.path.sep+'*%s.sav'%metric)

    for filename in dict_mod_un_feat:

        feat_un.extend(pickle.load(open(filename, 'rb')))


    ## read all the saved features with un feat sel

   
    models=[]


## Load the saved models

    for filename in dict_mod_filenames:

        models.extend(pickle.load(open(filename, 'rb')))
    counter=0

    results_list=[]

   # results_list_rfe=[]    
    for model in models:

       


        print(5*"**** ****")
        print(5*"**** ****")
        print(5*"**** ****")
        print("Testing model %s with optimal parameters and best features from univariate feature selection "%model.__class__.__name__)
       
        ##copy mode for the second test

        model_copy=copy.copy(model)
        
        print("Selected features")
        results_list_un_c=[]
        results_list_rfe_c=[]
        for filename in dict_mod_un_feat:
            
            if(model.__class__.__name__+'_best_feat_un.sav' in filename):

                indices=pickle.load(open(filename, 'rb'))
               
        print(labels[:77][np.array(indices).reshape((77))])

        sel_fea=labels[:77][np.array(indices).reshape((77))]

        for feat in sel_fea:


            dict_labels_un[feat]+=1

        # Project features
        
    
        X_train_red=X_train[0:,np.array(indices).reshape(77)]


        model.fit(X_train_red,y_train)

        
        X_test_red=X_test[:,np.array(indices).reshape(77)]

        score= model.score(X_test_red,y_test)  


        print("Performance score %.3f "%score)
        model.fit(X_train,y_train)

        score2=  model.score(X_test,y_test) 

       


        results_list_un_c.append( model.__class__.__name__) 
        results_list_un_c.append('Univariate')                                                   
        results_list_un_c.append(sum(indices==1))

       
        results_list_un_c.append('%.3f'%score2) 

        
        results_list_un_c.append('%.3f'%score) 



        
        print(5*"**** ****")
       
        print("Testing model %s with optimal parameters and best features from backward feature selection "%model.__class__.__name__)
       

        print("selected features")
        
        for filename in dict_mod_rfe_feat:
            
            if model.__class__.__name__+'_best_feat_rfe.sav' in filename:

                indices=pickle.load(open(filename, 'rb'))
               
                

        print(labels[:77][np.array(indices).reshape((77))])

        sel_fea=labels[:77][np.array(indices).reshape((77))]

        for feat in sel_fea:


            dict_labels_rfe[feat]+=1

       


            # Project features
        

        X_train_red=X_train[0:,np.array(indices).reshape(77)]

        model_copy=copy.copy(model)
        model_copy.fit(X_train_red,y_train)

        
        X_test_red=X_test[:,np.array(indices).reshape(77)]
        
        print("Performance score %f "%model_copy.score(X_test_red,y_test)) 
        

     

        results_list_rfe_c.append( model.__class__.__name__)

        results_list_rfe_c.append('RFE')                                                     
        results_list_rfe_c.append(sum(indices==1))

        model_copy2=copy.copy(model) 

        model_copy2.fit(X_train,y_train)
        results_list_rfe_c.append('%.3f'%model_copy2.score(X_test,y_test)) 
        results_list_rfe_c.append('%.3f'%model_copy.score(X_test_red,y_test)) 


        results_list.append(results_list_un_c)
        results_list.append(results_list_rfe_c)
      
        counter+=1
    results_file_name='results_%s_best_features.csv'%metric
   
    dfj = pd.DataFrame(results_list, columns=columns_names_csv)
   

    dfj.to_csv(results_file_name)                                                        
    
    
a1_sorted_keys = sorted(dict_labels_un, key=dict_labels_un.get, reverse=True)
for r in a1_sorted_keys:
    print(r, dict_labels_un[r])

a1_sorted_keys = sorted(dict_labels_rfe, key=dict_labels_rfe.get, reverse=True)
for r in a1_sorted_keys:
    print(r, dict_labels_rfe[r])


for r in a1_sorted_keys:
    print(r)

a1_sorted_keys = sorted(dict_labels_rfe, key=dict_labels_rfe.get, reverse=True)
for r in a1_sorted_keys:
    print(r)
