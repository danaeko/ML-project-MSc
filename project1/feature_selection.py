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
import glob,os,sys


try:
    X
except NameError:
    init=Init()

    X=init.X
    y=init.y
    labels=init.labels



datapath='best_features'


if not os.path.isdir('best_features'):

    os.mkdir('best_features')


metric_lists=['accuracy','f1','roc_auc']

metric_dict={}

## Load the saved models

for metric in metric_lists:
   
    dict_mod_filenames=glob.glob('best_models'+os.path.sep+'*%s.sav'%metric)
    models=[]
    for filename in dict_mod_filenames:

        models.extend(pickle.load(open(filename, 'rb')))

    s1=datapath+os.sep+metric

    if(not os.path.isdir(s1)):

        os.mkdir(s1)

    s1=datapath+os.sep+metric+os.sep+"results_log"

    if not os.path.isdir(s1):

        os.mkdir(s1)

    s1=datapath+os.sep+metric+os.sep+"models"

    if not os.path.isdir(s1):

        os.mkdir(s1)

   
    metric_dict.update({metric:models})

#We use the same random_state as with model selection
#There we train and validated only with the 80%
#Here we train with the 80% and we test with the remaining 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)



old = sys.stdout


for metric in metric_lists:
    counter=0
    sys.stdout=old
    print("testing models that performed better with %s metric  "%metric)
    print(5*"**** ****")
    print(5*"**** ****")
    print(5*"**** ****")

    best_scores_rfe=np.zeros(len(models))
    num_of_features_rfe=np.zeros(len(models))
    best_support_un=[]

    best_scores_un=np.zeros(len(models))
    num_of_features_un=np.zeros(len(models))
    best_support_rfe=[]
    for model in metric_dict[metric]:
        
        datap=datapath+os.sep+metric+os.sep+"results_log"+os.sep
       
        filename='results_%s.txt'%(model.__class__.__name__)
        print("results saved in folder %s"%datap)
        
        
        sys.stdout = open(datap+filename, 'w')
        print("testing best model "+model.__class__.__name__)
        


        best_prediction_score=-1
        best_features=[]
       

        for num_feat in range(1,13):

           
        ####Univariate feature selection
        

            print("testing for %d features"%num_feat)

            anova_filter = SelectKBest(f_classif, k=num_feat)
          

            anova = make_pipeline(anova_filter, model)



            anova.fit(X_train, y_train)

            y_pred = anova.predict(X_test)


          
           # print(classification_report(y_test, y_pred))

            print("anova score %f"%anova.score(X_test,y_test))


            if(anova.score(X_test,y_test)>best_scores_un[counter]):

                best_scores_un[counter]=anova.score(X_test,y_test)
                num_of_features_un[counter]=num_feat
                best_support_un=anova_filter.get_support()


          


        ####Backward feature selection
        ### using the sklearn implementatiion

            try:

                selector = RFE(model, n_features_to_select=num_feat, step=1)

                selector = selector.fit(X_train, y_train)

                selector_score=selector.score(X_test, y_test)
               
                if(selector_score>best_scores_rfe[counter]):

                    best_scores_rfe[counter]=selector_score
                    num_of_features_rfe[counter]=num_feat
                    best_support_rfe=selector.get_support()
                    

            # We cannot use this for knn since it does not return coef_
            except RuntimeError as e :
                if e == 'The classifier does not expose "coef_" or "feature_importances_" attributes':
                    print("The classifier does not support this feature")
                    num_of_features_rfe[counter]=12 
                    continue
                

        print("optimal number of features Univariate feature selection %d"%num_of_features_un[counter])
        print("Optimal score with  Univariate feature selection %f"%best_scores_un[counter])
        print("Optimal features")
        print(labels[:12][best_support_un])

        filename = model.__class__.__name__+'_best_feat_un.sav'
        filename=datapath+os.sep+metric+os.sep+"models"+os.sep+filename
        pickle.dump(best_support_un, open(filename, 'wb')) 
        
       
        

        print("Optimal score with  backward feature selection %s"%best_scores_rfe[counter])
        print("optimal number of features backward feature selection %d"%num_of_features_rfe[counter])
        print("Optimal features")
        print(labels[:12][best_support_rfe])

        filename = model.__class__.__name__+'_best_feat_rfe.sav'
        filename=datapath+os.sep+metric+os.sep+"models"+os.sep+filename
        pickle.dump(best_support_rfe, open(filename, 'wb'))
        

        counter+=1 

