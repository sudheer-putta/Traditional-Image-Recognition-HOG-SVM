import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

print('Implement dataset loading here.')
print('Expected: X (HOG features), y (labels)')
# Example:
# X=np.load('X.npy'); y=np.load('y.npy')
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
# pipe=Pipeline([('scaler',StandardScaler()),('svm',SVC())])
# params={'svm__C':[0.1,1,10],'svm__kernel':['linear','rbf']}
# grid=GridSearchCV(pipe,params,cv=5)
# grid.fit(X_train,y_train)
# print(classification_report(y_test,grid.predict(X_test)))
# joblib.dump(grid.best_estimator_,'models/svm_model.pkl')
