#Import Essential Modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense , Dropout
from keras.models import model_from_json , load_model
from keras import initializers
from keras_tuner import RandomSearch
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import graphviz
from itertools import product
import seaborn as sns

#load model 
model = load_model('app-v2/model/model-cnn-lstm-full-multi-pr-88.keras')


labels = ['BENIGN','DNS/LDAP','MSSQL','NTP', 'NetBios', 'SNMP', 'SSDP/UDP', 'Syn']
#labels = ['BENIGN','DNS/LDAP/SNMP/NetBios','MSSQL','NTP/SSDP/UDP/Syn']
#labels = ['BENIGN','DDoS']

df1 = pd.read_csv('app-v2/dataset/cicdos2019-multi-test-pr.csv')
X_predict = df1.drop(['Label'], axis=1)
Y_predict = df1['Label']

# Normalizing features
ms = MinMaxScaler()
scaler = StandardScaler()
X_predict = scaler.fit_transform(X_predict)
X_predict = ms.fit_transform(X_predict)


classes = model.predict(X_predict)
print(classes)
y_pred = np.argmax(classes, axis=-1)


print(classification_report(Y_predict, y_pred, target_names=labels, digits=6))

confusion_mtx = confusion_matrix(Y_predict, y_pred)
print(confusion_mtx)


cmn = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, cmap="Blues", annot=True, fmt='.4f', xticklabels=labels, yticklabels=labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()