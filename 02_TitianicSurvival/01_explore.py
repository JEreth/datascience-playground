import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
	
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
url = "data/train.csv"
dataset = read_csv(url)

# cleanb stuff
dataset.drop('PassengerId', 1) # name is no help to us
dataset.drop('Name', 1) # name is no help to us

# summarize what we have
print("Summarize the data set")
print(dataset.shape)
print(dataset.head(5))
print(dataset.describe())
# class distribution
print(dataset.groupby('Survived').size())

# do some plotting
#dataset.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False)
#pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()