import pandas as pd
df = pd.read_csv('exoplanets.csv', nrows=400)
#print(df.head())

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

X = df.iloc[:,1:]
y = df.iloc[:,0]

def light_plot(index):
    y_vals = X.iloc[index]
    x_vals = np.arange(len(y_vals))
    plt.figure(figsize=(15,8))
    plt.xlabel('Number of Observations')
    plt.ylabel('Light Flux')
    plt.title('Light Plot ' + str(index), size=15)
    plt.plot(x_vals, y_vals)
    plt.show()

#light_plot(1)

from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, recall_score

#y_pred_series = pd.Series(y_pred, index=y_test.index)
#score = len(y_pred_series[y_pred_series == y_test])
#print(np.sum(np.equal(y_pred, y_test))/len(y_test))

from sklearn.model_selection import train_test_split 
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

y_train = y_train.map({1: 0, 2: 1})
y_test = y_test.map({1: 0, 2: 1})

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

model = XGBClassifier(scale_pos_weight=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = accuracy_score(y_pred, y_test)
print('accuracy_score: ' + str(score))
print(recall_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


df['LABEL'] = df['LABEL'].replace([1,2], [0,1])
df['LABEL'].value_counts()
X = df.iloc[:,1:]
y = df.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
# ensuring that each fold is a representative sample of the whole dataset
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=2)
model = XGBClassifier(scale_pos_weight=10, random_state=2)
scores = cross_val_score(model, X, y, cv=kfold, scoring='recall')
print('Recall: ', scores)
print('Recall mean: ', scores.mean())

def grid_search(params, random=False, X=X, y=y, model=XGBClassifier(random_state=2)):
    xgb = model
    if random:
        # will sample 10 random combinations from the parameter space
        grid = RandomizedSearchCV(xgb, params, cv=kfold, n_jobs=-1, random_state=2, scoring='recall')
    else:
        grid = GridSearchCV(xgb, params, cv=kfold, n_jobs=-1, scoring='recall')
    grid.fit(X, y)
    best_params = grid.best_params_
    print("Best params:", best_params)
    best_score = grid.best_score_
    print("Best score: {:.5f}".format(best_score))
    
    means = grid.cv_results_['mean_test_score']
    print("All scores:", means)

# default 100
#grid_search(params={'n_estimators':[50, 200, 400, 800]})
# default 0.3
#grid_search(params={'learning_rate':[0.4, 0.5, 0.6, 0.7, 1.0]})

X_short = X.iloc[:74, :]
y_short = y.iloc[:74]


# grid_search(params={'max_depth':[1, 2, 3], 
#             'colsample_bynode':[0.5, 0.75, 1]}, 
#             X=X_short, y=y_short, 
#             model=XGBClassifier(random_state=2))



#all data
df_all = pd.read_csv('exoplanets.csv')
df_all['LABEL'] = df_all['LABEL'].replace(1, 0)
df_all['LABEL'] = df_all['LABEL'].replace(2, 1)

X_all = df_all.iloc[:,1:]
y_all = df_all.iloc[:,0]

df_all['LABEL'].value_counts()
weight = int(5050/37)

grid_search(params={'max_depth':[1, 2],'learning_rate':[0.001, 0.5, 1]}, X=X_all, y=y_all, 
            model=XGBClassifier(scale_pos_weight=weight))

def final_model(X, y, model):
    model.fit(X, y)
    y_pred = model.predict(X_all)
    score = recall_score(y_all, y_pred,)
    print(score)
    print(confusion_matrix(y_all, y_pred,))
    print(classification_report(y_all, y_pred))

final_model(X_short, y_short, 
            XGBClassifier(max_depth=3, colsample_by_node=0.5, 
                          random_state=2))

final_model(X, y, 
            XGBClassifier(max_depth=3, colsample_bynode=0.5, 
                          scale_pos_weight=10, random_state=2))

final_model(X_all, y_all, 
            XGBClassifier(max_depth=3, colsample_bynode=0.5, 
                          scale_pos_weight=weight, random_state=2))