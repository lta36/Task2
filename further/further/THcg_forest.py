#%%
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
from hyperopt import hp,STATUS_OK,Trials,fmin,tpe
import pandas as pd
import numpy as np

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

def hyperopt_train_test(params):
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, xtrain, ytrain).mean()

space4rf = {
    'max_depth': hp.choice('max_depth', range(1,50000)),
    # 'max_features': hp.choice('max_features', range(1,5)),
    'n_estimators': hp.choice('n_estimators', range(1,100000)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
}

def f(params):
    global best
    acc = hyperopt_train_test(params)
    if acc > best:
        best = acc
        global global_para
        global_para = params
    print ('new best:', best, params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4rf, algo=tpe.suggest, max_evals=100, trials=trials)
print("----------selection----------")
print(global_para['criterion'],global_para["max_depth"],global_para["n_estimators"])

rfc = RandomForestClassifier(random_state = 151,criterion = global_para['criterion'],max_depth = global_para["max_depth"], n_estimators = global_para["n_estimators"])
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
