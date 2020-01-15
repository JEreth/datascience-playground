import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import pickle
	
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load the data
url = "data/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# split dataset for training
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# use SVN
model = SVC(gamma='auto')
model.fit(X_train, Y_train)

# serialize model
filename = 'model.model'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_validation, Y_validation)
print(result)

# predict certain rows
Xnew = [[7.2,3.6,6.1,2.5]]
# make a prediction
ynew = model.predict(Xnew)
# if we need probabity informatiuon
# ynew = model.predict_proba(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))