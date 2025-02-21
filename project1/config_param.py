from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm  import SVC



par_tree = {
    'tree__criterion': ("gini","entropy"),
    'tree__max_depth': (2, 3, 4),
'tree__max_features': ("auto","log2",'sqrt'), 

}

par_knn = {
 'knn__n_neighbors': (3, 5,9,15),
    'knn__algorithm': ('ball_tree', 'kd_tree'),
    
}


par_svm_lin = {
 'svm_lin___penalty': ('l2', 'l1')
    
    
}

par_svm_nlin = {
    'svm_nlin__C': (1, 5),
    'svm_nlin__kernel': ('linear', 'poly', 'rbf'),
    
}


par_sgd = {
    
    'clf__max_iter': (50,),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__max_iter': (10, 50, 80),

 
}


par_log = {
    
    
    'log__solver': ('newton-cg', 'lbfgs', 'liblinear'),

   

 
}

par_mlper = {
    
    
    'mlper__activation':  ('relu', 'logistic'),
   'mlper__hidden_layer_sizes':(50,100,150),
      'mlper__learning_rate':('constant', 'invscaling', 'adaptive'),

   

 
}


pip_tree= Pipeline([
    ('tree', DecisionTreeClassifier()),


])

pip_sgd= Pipeline([
 
 ('clf', SGDClassifier( max_iter=150)),

])

pip_log= Pipeline([
 
 ('log', LogisticRegression()),

])


pip_knn= Pipeline([
 
 ('knn', KNeighborsClassifier()),

])




pip_svm_nlin= Pipeline([
 
 ('svm_nlin', SVC(max_iter=10000)),


])


pip_mlper= Pipeline([
 
 ('mlper', MLPClassifier(solver='lbfgs')),

  

])




model_list_map=[[pip_tree,par_tree],[pip_sgd,par_sgd],[pip_log,par_log],[pip_knn,par_knn],[pip_svm_nlin,par_svm_nlin],[pip_mlper,par_mlper]]

