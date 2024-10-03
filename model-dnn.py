 #%tensorflow_version 2.x
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Convolution1D, Dense, Dropout, MaxPooling1D, LSTM, Conv1D, BatchNormalization, AveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from keras import callbacks
from keras.callbacks import CSVLogger
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
from keras import initializers

# Define the model architecture
def create_dnn_model(input_shape, num_classes):
    model = Sequential()

    # Fully connected layer with softmax activation
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model



#loading dataset
df = pd.read_csv("app-v2/dataset/cicdos2019-multi-train-bin-pr.csv")

#labels = ['BENIGN','DNS/LDAP','MSSQL','NTP', 'NetBios', 'SNMP', 'SSDP/UDP', 'Syn']
#labels = ['BENIGN','DNS/LDAP/SNMP/NetBios','MSSQL','NTP/SSDP/UDP/Syn']
labels = ['BENIGN','DDoS']


#separating input and output attributes
x = df.drop(['Label'], axis=1)
y = df['Label']


# Encode labels to integers and then to one-hot
le = LabelEncoder()
y = to_categorical(y)


#normalizing the data
ms = MinMaxScaler()
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = ms.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
     

input_shape=X_train.shape[1:]
num_classes = 2

model = create_dnn_model(input_shape, num_classes)

opt = optimizers.Adam(learning_rate=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=1000, callbacks=None , validation_data=(X_test, y_test), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

# Evaluate the model on the test data
scores = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy: {scores[1]}')

#save model
model.save('model-dnn-bin.keras')


df1 = pd.read_csv('app-v2/dataset/cicdos2019-multi-test-bin-pr.csv')
X_predict = df1.drop(['Label'], axis=1)
Y_predict = df1['Label']


# Normalizing features
X_predict = scaler.fit_transform(X_predict)
X_predict = ms.fit_transform(X_predict)



classes = model.predict(X_predict)
print(classes)
y_pred = np.argmax(classes, axis=-1)

print(classification_report(Y_predict, y_pred, target_names=labels, digits=6))

confusion_mtx = confusion_matrix(Y_predict, y_pred)
print(confusion_mtx)
