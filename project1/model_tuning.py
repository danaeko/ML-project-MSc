from init import *
from read_data import *
from config_param import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import pickle,copy
import os
import warnings
warnings.filterwarnings('ignore') 
import numpy as np

columns_names_csv=['Model', 'Optimal Parameters', 'Validation Score','Testing Score']

try:
    X
except NameError:
    init=Init()

    X=init.X
    y=init.y
    labels=init.labels

if not os.path.isdir('best_models'):

    os.mkdir('best_models')

metric_lists=['accuracy','f1','roc_auc']

#Number of models to tune
NUM_OF_MODELS=len(model_list_map)
# Number of random trials
NUM_TRIALS = 30

##Inititialisation and setting the dimensions of auxiliary variables

best_score_array=np.zeros((NUM_OF_MODELS,1))
best_param_array=[None]*NUM_OF_MODELS
best_grid_search_array=[None]*NUM_OF_MODELS

best_score=0

best_grid_search=[None]

best_score_train_array=np.zeros((NUM_TRIALS,NUM_OF_MODELS))
train_scores=[None]*NUM_OF_MODELS

param_lists=[None]*NUM_OF_MODELS

grid_search_save=[None]*NUM_OF_MODELS

for i in range(len(model_list_map)):
    comb=1
    for key in model_list_map[i][1]:

        comb*=len(model_list_map[i][1][key])


    train_scores[i]=[[None]*comb]*NUM_TRIALS
    param_lists[i]=[None]*comb
    grid_search_save[i]=[None]*NUM_OF_MODELS

dict_training_arrays={}

### K-fold cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)

metric_counter=0
for metric in metric_lists:

    best_score_train_array=np.zeros((NUM_TRIALS,NUM_OF_MODELS))
    train_scores=[None]*NUM_OF_MODELS

    param_lists=[None]*NUM_OF_MODELS


    grid_search_save=[None]*NUM_OF_MODELS



    for i in range(len(model_list_map)):
        comb=1
        for key in model_list_map[i][1]:

            comb*=len(model_list_map[i][1][key])


        train_scores[i]=[[None]*comb]*NUM_TRIALS
        param_lists[i]=[None]*comb
        grid_search_save[i]=[None]*NUM_OF_MODELS

    for j in range(NUM_TRIALS):

        print(5*"****")
        print(5*"****")
        print("Iteration %d"%(j+1))
        print(5*"****")
        print(5*"****")
        for i in range(len(model_list_map)):
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.

            print("Tuning %s:"%model_list_map[i][0]) 
            inner_cv = KFold(n_splits=4, shuffle=True, random_state=j)
            outer_cv = KFold(n_splits=4, shuffle=True, random_state=j)
            cv_split = KFold(n_splits=4, shuffle=True, random_state=j)

        # Non_nested parameter search and scoring
            grid_search = GridSearchCV(estimator=model_list_map[i][0],  n_jobs=-1, scoring=metric, verbose=0,param_grid=model_list_map[i][1], cv=cv_split,return_train_score=True)
            grid_search.fit(X_train, y_train)

            

            train_scores[i][j]=grid_search.cv_results_['mean_test_score']
            grid_search_save[i][j]=grid_search.cv_results_['mean_test_score']


            param_lists[i]= grid_search.cv_results_['params']
            grid_search_save[i]= grid_search.cv_results_
           

            print('scorer : %s'%grid_search.scorer_)

            nested_score = cross_val_score(grid_search,X_train,y_train, cv=outer_cv)


            cur_best_av_score = nested_score.mean()

         
            print("Best score %f:"%cur_best_av_score) 

            if grid_search.best_score_>best_score_array[i]:

                best_score_array[i]=grid_search.best_score_
                best_param_array[i]=grid_search.best_estimator_.get_params()
                best_grid_search_array[i]= grid_search

            if grid_search.best_score_>best_score:

                best_score=grid_search.best_score_
              
                best_grid_search= grid_search

            nested_score = cross_val_score(grid_search, X=X, y=y, cv=outer_cv)
            #nested_scores[j,i] = nested_score.mean()

        

        # Nested CV with parameter optimization
      #  nested_score = cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv)
      #  nested_scores[i] = nested_score.mean()
    dict_entry={metric:train_scores}
    
    dict_training_arrays.update(dict_entry)
    metric_counter+=1


for metric,val_scores in dict_training_arrays.items():
    print("Results for metric -> %s"%metric)
    results_list=[]
    print("Results for metric -> %s"%metric)

    for i in range(len(model_list_map)):


        results_list_c=[]
        print("Model %s:"%model_list_map[i][0])
       
            
        # print(train_scores[i])

        aver_scores=np.mean(val_scores[i],axis=0  )
      

        best_aver_score=np.max(aver_scores)
        print("Best Average Validation score: %0.3f" % best_aver_score)
        
        
        ind=np.argmax(aver_scores)
        

        print('best parameters')


        print(param_lists[i][ind])
        
        name =model_list_map[i][0].steps[0][0]
        estimator=model_list_map[i][0]
       
        estimator.set_params(**param_lists[i][ind])


        str_p=""
        
        for key,value in param_lists[i][ind].items():

                str_p+=key+" : " +str(value)+" "


        results_list_c.append( name)                                                    
        results_list_c.append(str_p)      
        results_list_c.append('%.3f'%best_aver_score) 

        
        ## Save the model for later use

        filename = name+'_best_par_%s.sav'%metric
        pickle.dump(estimator, open('best_models'+os.path.sep+filename, 'wb')) 

            


        estimator.fit(X_train,y_train)
        
        print("Score for optimum hyperparameters: %0.3f" % estimator.score(X_test,y_test))

        results_list_c.append("%.3f"%estimator.score(X_test,y_test))


        results_list.append(results_list_c)

    results_file_name='results_best_models%s.csv'%metric
   
    dfj = pd.DataFrame(results_list, columns=columns_names_csv) 

    print("Overall Results") 

    print(dfj)  

    dfj.to_csv(results_file_name)   