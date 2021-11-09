#%%
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
from hyperopt import hp,STATUS_OK,Trials,fmin,tpe
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB

best = -1
global_para = {}


train = pd.read_csv("combinetrain1025.csv",encoding = 'unicode_escape')
ytrain = train['__label']
dev = pd.read_csv("combinedev1025.csv",encoding = 'unicode_escape')
ydev = dev['__label']
test = pd.read_csv("combinetest1025.csv",encoding = 'unicode_escape')
ytest = test['__label']


# drop the label column and the index column
xtrain = train.drop(columns = ['__label','Unnamed: 0'],axis = 1)
xtest = test.drop(columns = ['__label','Unnamed: 0'],axis = 1)
xdev = dev.drop(columns = ['__label','Unnamed: 0'],axis = 1)
print(xdev.head())

print("1")
def hyperopt_train_test(params):
    # print(params)
    clf = CategoricalNB(**params)
    return cross_val_score(clf, xtrain, ytrain).mean()

space4rf = {
    'alpha': hp.choice('alpha', range(0,1)),
    'fit_prior': hp.choice('fit_prior', [True,False])
    # ,
    # 'class_prior': hp.choice('class_prior', np.power(10,np.random.uniform(-5,5, 50))),
    # 'min_categories': hp.choice('min_categories', np.linspace(-10,10,100))
}

def f(params):
    global best
    acc = hyperopt_train_test(params)
    print(params)
    if acc > best:
        best = acc
        global global_para
        global_para = params
    print ('new best:', best)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4rf, algo=tpe.suggest, max_evals=500, trials=trials)
print("----------selection----------")
print(global_para['alpha'],global_para["fit_prior"],global_para["class_prior"],global_para["min_categories"])


rfc = CategoricalNB(alpha=global_para['alpha'], fit_prior=global_para["fit_prior"], class_prior=global_para["class_prior"], min_categories = global_para["min_categories"])
clf = rfc.fit(xtrain,ytrain)
print("train_score =",clf.score(xtrain,ytrain))
A = confusion_matrix(ytrain,clf.predict(xtrain))
print("----------TRAIN-------------")
print(A)
score_d = clf.score(xdev,ydev)
print("dev_score =",score_d)
B = confusion_matrix(ydev,clf.predict(xdev))
print("----------DEV-------------")
print(B)
score_c = clf.score(xtest,ytest)
print("test_score =",score_c)
C = confusion_matrix(ytest,clf.predict(xtest))
print("----------TEST-------------")
print(C)




# %%
