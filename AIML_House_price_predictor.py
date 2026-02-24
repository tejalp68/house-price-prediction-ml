from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------

df = fetch_california_housing()
df

# -----------------------------

dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names
dataset.head()


# -----------------------------

dataset['price'] = df.target

# -----------------------------

dataset

# -----------------------------

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# -----------------------------

x.head()

# -----------------------------

y.head()

# -----------------------------

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

# -----------------------------

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lin_reg = LinearRegression()
lin_reg.fit(x_train ,y_train)
mse =cross_val_score(lin_reg , x_train ,y_train ,scoring = 'neg_mean_squared_error' , cv =5)
print(mse)

# -----------------------------

np.mean(mse)

# -----------------------------

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV   #will help with hyperparameter 
ridge =Ridge()

# -----------------------------

params = {'alpha' : [1e-15 , 1e-10 ,1e-8 ,1e-3 ,1e-2 ,1 ,5 ,10 ,20,30,35,40,45,50,55,100]}
ridge_regressor = GridSearchCV (ridge ,params ,scoring ='neg_mean_squared_error', cv=5)
ridge_regressor.fit( x_train ,y_train )

# -----------------------------

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

# -----------------------------



# -----------------------------

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV   #will help with hyperparameter 
lasso =Lasso()

params = {'alpha' : [1e-15 , 1e-10 ,1e-8 ,1e-3 ,1e-2 ,1 ,5 ,10 ,20,30,35,40,45,50,55,100]}
lasso_regressor = GridSearchCV (lasso ,params ,scoring ='neg_mean_squared_error', cv=5)
lasso_regressor.fit( x_train ,y_train )

# -----------------------------

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

# -----------------------------

y_pred = lin_reg.predict(x_test)
from sklearn.metrics import r2_score 

r2_score1 = r2_score(y_pred , y_test)


# -----------------------------

print(r2_score1)

# -----------------------------

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# -----------------------------

df =load_breast_cancer()

#independant feature
x = pd.DataFrame(df ['data'],columns = df['feature_names'])

# -----------------------------

x.head()

# -----------------------------

y = pd.DataFrame(df['target'] , columns = ["Target"])
y

# -----------------------------

y['Target'].value_counts()

# -----------------------------

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

# -----------------------------

params = [{'C':[1,5,10]} , {'max_iter' :[100,150]}]

# -----------------------------

model1 =LogisticRegression (C = 100 ,max_iter =100)

# -----------------------------

model = GridSearchCV (model1 ,param_grid = params , scoring ='f1' , cv =5)

# -----------------------------

model.fit(x_train ,y_train)

# -----------------------------

print(model.best_params_)
print(model.best_score_)

# -----------------------------

y_pred =model.predict(x_test)

# -----------------------------

y_pred

# -----------------------------

from sklearn.metrics import confusion_matrix , classification_report ,accuracy_score

# -----------------------------

confusion_matrix(y_test ,y_pred)

# -----------------------------

accuracy_score(y_test , y_pred)

# -----------------------------

print(classification_report(y_test,y_pred))

# -----------------------------

