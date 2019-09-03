import pandas as pd
import numpy as np
import joblib
df = pd.read_csv('./housing.csv')
df.info()
df.corr()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X = df[['RM','LSTAT','PTRATIO']]

y = df[['MEDV']]

# y= preprocessing.scale(y)
# X= preprocessing.scale(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=21)

clf = LinearRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

joblib.dump(clf, open('model.pkl','wb'))

model = joblib.load(open('model.pkl','rb'))
