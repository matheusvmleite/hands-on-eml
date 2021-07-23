# Ieva Zarina, 2016, licensed under the Apache 2.0 licnese

import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import precision_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# use DMatrix for xgboost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# set xgboost params
param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations

bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

# extracting most confident predictions
best_preds = np.asarray([np.argmax(line) for line in preds])
print("Numpy array precision:", precision_score(y_test, best_preds, average='macro'))

# save the models for later
joblib.dump(bst, 'bst_model.pkl', compress=True)

# validating an future call with only one sample

samples = 1
size_each_sample = len(X_test[0])
predict_input = X_test[0].reshape(samples, size_each_sample)
print('Inputs:', predict_input)
predict_result = bst.predict(xgb.DMatrix(predict_input))
print('Test one prediticion:', predict_result)
