#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import seaborn as sns
import sys
import os
from IPython.display import display
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')


# In[2]:


emp_data = pd.read_excel('C:/Users/GANESHA/Downloads/project/Data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls')


# In[3]:


emp_data['Department_Role'] = emp_data['EmpJobRole'] + " - " + emp_data['EmpDepartment']


# In[4]:


emp_data_columns = emp_data.columns.values
print(emp_data_columns)


# In[5]:


X = emp_data[emp_data_columns[0:29]]
Y = emp_data['PerformanceRating']
X.drop('PerformanceRating', axis=1, inplace=True)
X.drop('EmpNumber', axis=1, inplace = True)


# In[6]:


X_Gender = pd.get_dummies(X['Gender'], drop_first = False , sparse = True)
X_Education = pd.get_dummies(X['EducationBackground'], drop_first = False , sparse = True)
X_MaritalStatus = pd.get_dummies(X['MaritalStatus'], drop_first = False , sparse = True)
X_EmpDepartment = pd.get_dummies(X['EmpDepartment'], drop_first = False , sparse = True)
X_EmpJobRole = pd.get_dummies(X['EmpJobRole'], drop_first = False , sparse = True)
X_BusinessTravelFrequency = pd.get_dummies(X['BusinessTravelFrequency'], drop_first = False , sparse = True)
X_OverTime = pd.get_dummies(X['OverTime'], drop_first = False, sparse = True)
X_Department_Role = pd.get_dummies(X['Department_Role'], drop_first = False , sparse = True)
X_Attrition = pd.get_dummies(X['Attrition'], drop_first = False, sparse = True)


# In[7]:


X['Gender'] = X_Gender
X['EducationBackground'] = X_Education
X['MaritalStatus'] = X_MaritalStatus
X['EmpDepartment'] = X_EmpDepartment
X['EmpJobRole'] = X_EmpJobRole
X['BusinessTravelFrequency'] = X_BusinessTravelFrequency
X['OverTime'] = X_OverTime
X['Department_Role'] = X_Department_Role
X['Attrition'] = X_Attrition
X.shape


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 0, stratify = emp_data['PerformanceRating'])


# In[9]:


print("shape of X train -",X_train.shape)
print("shape of X test - ",X_test.shape)
print("shape of Y train -",Y_train.shape)
print("shape of Y test -",Y_test.shape)


# In[10]:


X.head(4)


# In[11]:


from pandas import set_option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


# In[12]:


def GetBasedModel():
    basedModels = []
    basedModels.append(('LR'   , LogisticRegression()))
    basedModels.append(('LDA'  , LinearDiscriminantAnalysis()))
    basedModels.append(('KNN'  , KNeighborsClassifier()))
    basedModels.append(('CART' , DecisionTreeClassifier()))
    basedModels.append(('NB'   , GaussianNB()))
    basedModels.append(('SVM'  , SVC(probability=True)))
    basedModels.append(('AB'   , AdaBoostClassifier()))
    basedModels.append(('GBM'  , GradientBoostingClassifier()))
    basedModels.append(('RF'   , RandomForestClassifier()))
    basedModels.append(('ET'   , ExtraTreesClassifier()))    
    return basedModels


# In[13]:


SEED = 7
np.random.seed(SEED)


# In[250]:


def BasedLine2(X_train, Y_train, models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds, random_state=SEED)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    return names, results


# In[251]:


class PlotBoxR(object):  
    
    def __Trace(self,nameOfFeature,value):    
        trace = go.Box(
            y=value,
            name = nameOfFeature,
            marker = dict(
                color = 'rgb(0, 128, 128)',
            )
        )
        return trace

    def PlotResult(self,names,results):        
        data = []
        for i in range(len(names)):
            data.append(self.__Trace(names[i],results[i]))
        py.iplot(data)


# In[263]:


models = GetBasedModel()
names,results = BasedLine2(X_train, Y_train,models)
#PlotBoxR().PlotResult(names,results)


# In[264]:


def ScoreDataFrame(names,results):
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
    
        return float(prc.format(f_val))

    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(),4))

    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': scores})
    return scoreDataFrame


# In[265]:


basedLineScore = ScoreDataFrame(names,results)
basedLineScore


# In[266]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def GetScaledModel(nameOfScaler):
    
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()

    pipelines = []
    pipelines.append((nameOfScaler+'LR'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression())])))
    pipelines.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))
    pipelines.append((nameOfScaler+'KNN' , Pipeline([('Scaler', scaler),('KNN' , KNeighborsClassifier())])))
    pipelines.append((nameOfScaler+'CART', Pipeline([('Scaler', scaler),('CART', DecisionTreeClassifier())])))
    pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))
    pipelines.append((nameOfScaler+'SVM' , Pipeline([('Scaler', scaler),('SVM' , SVC())])))
    pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))
    pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))
    pipelines.append((nameOfScaler+'RF'  , Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier())])  ))
    pipelines.append((nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier())])  ))
    return pipelines 


# In[268]:


models = GetScaledModel('standard')
names,results = BasedLine2(X_train, Y_train,models)
#PlotBoxR().PlotResult(names,results)
scaledScoreStandard = ScoreDataFrame(names,results)
compareModels = pd.concat([basedLineScore,
                           scaledScoreStandard], axis=1)
compareModels


# In[274]:


models = GetScaledModel('minmax')
names,results = BasedLine2(X_train, Y_train,models)
#PlotBoxR().PlotResult(names,results)

scaledScoreMinMax = ScoreDataFrame(names,results)
compareModels = pd.concat([basedLineScore,
                           scaledScoreStandard,
                          scaledScoreMinMax], axis=1)
compareModels


# In[275]:


def HeatMap(emp_data,x=True):
        correlations = emp_data.corr()
        ## Create color map ranging between two colors
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        fig = sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',square=True, linewidths=.5, annot=x, cbar_kws={"shrink": .75})
        fig.set_xticklabels(fig.get_xticklabels(), rotation = 90, fontsize = 10)
        fig.set_yticklabels(fig.get_yticklabels(), rotation = 0, fontsize = 10)
        plt.tight_layout()
        plt.show()

HeatMap(emp_data,x=True)


# In[287]:


clf = ExtraTreesClassifier(n_estimators=250, random_state = SEED)
clf.fit(X_train, Y_train)
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.plot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, emp_data.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[289]:


print("Most important factores influencing employee performance are OverTime, EmpEducationLevel, ExperienceYearsInCurrentRole, ExperienceyearsAtThisCompany, EmpWorkLifeBalance")


# In[318]:


from sklearn.decomposition import PCA


# In[319]:


pca = PCA(0.99)


# In[320]:


pca.fit(X_train)


# In[321]:


pca.n_components_


# In[322]:


pca_Xtrain = pca.transform(X_train)
pca_Xtest = pca.transform(X_test)


# In[323]:


models = GetScaledModel('minmax')
names,results = BasedLine2(pca_Xtrain, Y_train,models)
#PlotBoxR().PlotResult(names,results)

scaledScoreMinMax = ScoreDataFrame(names,results)
compareModels = pd.concat([basedLineScore,
                           scaledScoreStandard,
                          scaledScoreMinMax], axis=1)
compareModels


# In[324]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform


# In[325]:


class RandomSearch(object):
    
    def __init__(self,X_train,y_train,model,hyperparameters):
        
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.hyperparameters = hyperparameters
        
    def RandomSearch(self):
        # Create randomized search 10-fold cross validation and 100 iterations
        cv = 10
        clf = RandomizedSearchCV(self.model,
                                 self.hyperparameters,
                                 random_state=1,
                                 n_iter=100,
                                 cv=cv,
                                 verbose=0,
                                 n_jobs=-1,
                                 )
        # Fit randomized search
        best_model = clf.fit(self.X_train, self.y_train)
        message = (best_model.best_score_, best_model.best_params_)
        print("Best: %f using %s" % (message))

        return best_model,best_model.best_params_
    
    def BestModelPridict(self,X_test):
        
        best_model,_ = self.RandomSearch()
        pred = best_model.predict(X_test)
        return pred


# In[326]:


class GridSearch(object):
    
    def __init__(self,X_train,y_train,model,hyperparameters):
        
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.hyperparameters = hyperparameters
        
    def GridSearch(self):
        # Create randomized search 10-fold cross validation and 100 iterations
        cv = 10
        clf = GridSearchCV(self.model,
                                 self.hyperparameters,
                                 cv=cv,
                                 verbose=0,
                                 n_jobs=-1,
                                 )
        # Fit randomized search
        best_model = clf.fit(self.X_train, self.y_train)
        message = (best_model.best_score_, best_model.best_params_)
        print("Best: %f using %s" % (message))

        return best_model,best_model.best_params_
    
    def BestModelPridict(self,X_test):
        
        best_model,_ = self.GridSearch()
        pred = best_model.predict(X_test)
        return pred


# In[349]:


def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" #first cast decimal as str
    #     print(prc) #str format output is {:.3f}
        return float(prc.format(f_val))


# In[352]:


model = LogisticRegression()
# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter distribution using uniform distribution
C = uniform(loc=0, scale=4)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
LR_RandSearch = RandomSearch(X_train,Y_train,model,hyperparameters)
# LR_best_model,LR_best_params = LR_RandSearch.RandomSearch()
Prediction_LR = LR_RandSearch.BestModelPridict(X_test)
print('prediction on test set is:' ,floatingDecimals((Y_test == Prediction_LR).mean(),7))


# In[328]:


model_KNN = KNeighborsClassifier()
neighbors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
param_grid = dict(n_neighbors=neighbors)


# In[331]:


KNN_GridSearch = GridSearch(X_train,Y_train,model_KNN,param_grid)
Prediction_KNN = KNN_GridSearch.BestModelPridict(X_test)
print('prediction on test set is:' ,floatingDecimals((Y_test == Prediction_KNN).mean(),7))


# In[342]:


from scipy.stats import randint
max_depth_value = [3, None]
max_features_value =  randint(1, 4)
min_samples_leaf_value = randint(1, 4)
criterion_value = ["gini", "entropy"]


# In[343]:


param_grid = dict(max_depth = max_depth_value,
                  max_features = max_features_value,
                  min_samples_leaf = min_samples_leaf_value,
                  criterion = criterion_value)


# In[345]:


model_CART = DecisionTreeClassifier()
CART_RandSearch = RandomSearch(X_train,Y_train,model_CART,param_grid)
Prediction_CART = CART_RandSearch.BestModelPridict(X_test)
print('prediction on test set is:' ,floatingDecimals((Y_test == Prediction_CART).mean(),7))


# In[333]:


c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = [ 'linear' , 'poly' , 'rbf' , 'sigmoid' ]
param_grid = dict(C=c_values, kernel=kernel_values)
model_SVC = SVC()


# In[335]:


SVC_GridSearch = GridSearch(X_train,Y_train,model_SVC,param_grid)
Prediction_SVC = SVC_GridSearch.BestModelPridict(X_test)
print('prediction on test set is:' ,floatingDecimals((Y_test == Prediction_SVC).mean(),7))


# In[336]:


learning_rate_value = [.01,.05,.1,.5,1]
n_estimators_value = [50,100,150,200,250,300]

param_grid = dict(learning_rate=learning_rate_value, n_estimators=n_estimators_value)


# In[338]:


model_Ad = AdaBoostClassifier()
Ad_GridSearch = GridSearch(X_train,Y_train,model_Ad,param_grid)
Prediction_Ad = Ad_GridSearch.BestModelPridict(X_test)
print('prediction on test set is:' ,floatingDecimals((Y_test == Prediction_Ad).mean(),7))


# In[339]:


learning_rate_value = [.01,.05,.1,.5,1]
n_estimators_value = [50,100,150,200,250,300]

param_grid = dict(learning_rate=learning_rate_value, n_estimators=n_estimators_value)


# In[340]:


model_GB = GradientBoostingClassifier()
GB_GridSearch = GridSearch(X_train,Y_train,model_GB,param_grid)
Prediction_GB = GB_GridSearch.BestModelPridict(X_test)
print('prediction on test set is:' ,floatingDecimals((Y_test == Prediction_GB).mean(),7))


# In[346]:


from sklearn.ensemble import VotingClassifier


# In[353]:


param = {'C': 2.1326611398920683, 'penalty': 'l1'}
model1 = LogisticRegression(**param)

param = {'n_neighbors': 12}
model2 = KNeighborsClassifier(**param)

param = {'C': 0.5, 'kernel': 'linear'}
model3 = SVC(**param)

param = {'criterion': 'gini', 'max_depth': None, 'max_features': 3, 'min_samples_leaf': 3}
model4 = DecisionTreeClassifier(**param)

param = {'learning_rate': 0.05, 'n_estimators': 250}
model5 = AdaBoostClassifier(**param)

param = {'learning_rate': 0.01, 'n_estimators': 150}
model6 = GradientBoostingClassifier(**param)

model7 = GaussianNB()

model8 = RandomForestClassifier()

model9 = ExtraTreesClassifier()


# In[354]:


estimators = [('LR',model1), ('KNN',model2), ('SVC',model3),
              ('DT',model4), ('ADa',model5), ('GB',model6),
              ('NB',model7), ('RF',model8),  ('ET',model9)]


# In[356]:


kfold = StratifiedKFold(n_splits=10, random_state=SEED)
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X_train,Y_train, cv=kfold)
print('Accuracy on train: ',results.mean())
ensemble_model = ensemble.fit(X_train,Y_train)
pred = ensemble_model.predict(X_test)
print('Accuracy on test:' , (Y_test == pred).mean())


# In[391]:


def get_models():
    """Generate a library of base learners."""
    param = {'C': 0.7678243129497218, 'penalty': 'l1'}
    model1 = LogisticRegression(**param)

    param = {'n_neighbors': 15}
    model2 = KNeighborsClassifier(**param)

    param = {'C': 1.7, 'kernel': 'linear', 'probability':True}
    model3 = SVC(**param)

    param = {'criterion': 'gini', 'max_depth': 3, 'max_features': 2, 'min_samples_leaf': 3}
    model4 = DecisionTreeClassifier(**param)

    param = {'learning_rate': 0.05, 'n_estimators': 150}
    model5 = AdaBoostClassifier(**param)

    param = {'learning_rate': 0.01, 'n_estimators': 100}
    model6 = GradientBoostingClassifier(**param)

    model7 = GaussianNB()

    model8 = RandomForestClassifier()

    model9 = ExtraTreesClassifier()

    models = {'LR':model1, 'KNN':model2, 'SVC':model3,
              'DT':model4, 'ADa':model5, 'GB':model6,
              'NB':model7, 'RF':model8,  'ET':model9
              }

    return models


# In[392]:


def train_predict(model_list,xtrain, xtest, ytrain, ytest):
    """Fit models in list on training set and return preds"""
    P = np.zeros((ytest.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(xtrain, ytrain)
        P.iloc[:, i] = m.predict_proba(xtest)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


# In[393]:


models = get_models()
P = train_predict(models,X_train,X_test,Y_train,Y_test)


# In[361]:


get_ipython().system('pip install mlens')


# In[362]:


from mlens.visualization import corrmat

corrmat(P.corr(), inflate=False)


# In[363]:


corrmat(P.apply(lambda predic: 1*(predic >= 0.5) - Y_test).corr(), inflate=False)


# In[ ]:




