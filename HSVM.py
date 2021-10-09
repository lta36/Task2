#%%

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

best = -1
global_para = {}

train = pd.read_csv("train923.csv",encoding = 'unicode_escape')
ytrain = train['__label']
dev = pd.read_csv("dev923.csv",encoding = 'unicode_escape')
ydev = dev['__label']
test = pd.read_csv("test923.csv",encoding = 'unicode_escape')
ytest = test['__label']



xtrain = train.drop(['1','0'],axis = 1)
xtest = test.drop(['1','0'],axis = 1)
xdev = dev.drop(['1','0'],axis = 1)


print("1")
def hyperopt_train_test(params):
    # print(params)
    clf = SVC(**params)
    return cross_val_score(clf, xtrain, ytrain).mean()

space4rf = {
    'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    'degree': hp.choice('degree', range(1,10)),
    'C': hp.choice('C', np.power(10,np.random.uniform(-5,5, 50))),
    'coef0': hp.choice('coef0', np.linspace(-10,10,100)),
    'gamma': hp.choice('gamma', ['scale', 'auto'])
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
best = fmin(f, space4rf, algo=tpe.suggest, max_evals=10000, trials=trials)
print("2")
print("----------selection----------")
print(global_para['kernel'],global_para["degree"],global_para["C"],global_para["coef0"],global_para["gamma"])


rfc = SVC(kernel=global_para['kernel'], degree=global_para["degree"], C=global_para["C"], coef0=global_para["coef0"], gamma=global_para["gamma"])
print("3")
clf = rfc.fit(xtrain,ytrain)
print("train_score =",clf.score(xtrain,ytrain))
score_d = clf.score(xdev,ydev)
print("4")
print("dev_score =",score_d)
score_c = clf.score(xtest,ytest)
print("test_score =",score_c)




# %%
