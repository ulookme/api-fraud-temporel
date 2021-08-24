# Importing the libraries
import numpy as np
import pandas as pd
import pickle
import sys, os
import sklearn
import pymongo
from pymongo import MongoClient
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from pandas.core.frame import DataFrame
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import TomekLinks
#from imblearn.combine import SMOTETomek
#from imblearn.under_sampling import RandomUnderSampler

import numpy

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU, TimeDistributed, LSTM
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# import libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#import seaborn as sns
#sns.set(color_codes=True)
#import matplotlib.pyplot as plt

import tensorflow as tf



from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
# Multiple Inputs
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Bidirectional
import math, sys, time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from pandas._testing import assert_frame_equal
from pandas.testing import assert_index_equal
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"



# fonction de recupération depuis la base de donné mogodb 
def load_data(nrows):
    client = pymongo.MongoClient("mongodb+srv://transac:Mhajjar3@cluster0.hskyz.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    db = client.transaction
    col = db.collec_init
    data = pd.DataFrame(list(col.find()))
    data = data.drop(columns=["_id","id","oneclick","weight","card_bank","card_country","card_operation_type"])
    print(data)
    #data =  shuffle(data, random_state=42)
    #data = data.iloc[0:40000]
    # Convert integer valued (numeric) columns to floating point
    numeric_columns = data.select_dtypes(["int64", "float64"]).columns
    data[numeric_columns] = data[numeric_columns].astype("float32")
    #data = pd.read_csv('transaction_simplon.csv', nrows=nrows)
    return data




def preprocess(df):
    

    """ Combination of SMOTE and Tomek Links Undersampling

    SMOTE is an oversampling method that synthesizes new plausible examples in the majority class.
    Tomek Links refers to a method for identifying pairs of nearest neighbors in a dataset that have different classes.
    Removing one or both of the examples in these pairs (such as the examples in the majority class)
    has the effect of making the decision boundary in the training dataset less noisy or ambiguous.
    >>> df1 = preprocess(df)
    >>> df1.dtypes()
    Name: fraud, dtype: int64
    client_id     float32
    site_name     float32
    card_type     float32
    reference     float32
    ip_country    float32
    amount        float32
    fraud         float32
    dtype: object

    """
    df_y = df["fraud"]
    print(df_y)
    df_x = df.drop(columns=["fraud"])
    print(df_x)

    counters = Counter(df_y)
    print(counters)
    

    # Convert integer valued (numeric) columns to floating point
    numeric_columns = df_x.select_dtypes(["int64", "float64"]).columns
    df_x[numeric_columns] = df_x[numeric_columns].astype("float32")
    df = pd.concat([df_x, df_y], axis=1)
    print("dataframe apres factorisation verification de fraud")
    print(df["fraud"].value_counts())
    print(df.dtypes)

    return df


def standardize(df, pickle_save_path: str) -> DataFrame:

    """
    Scales numerical columns using their means and standard deviation to get
    z-scores: the mean of each numerical column becomes 0, and the standard
    deviation becomes 1. This can help the model converge during training.
    Args:
      dataframe: Pandas dataframe
      pickle_save_path: folder path to save the MinMaxScaler pickle in
    Returns:
      Input dataframe with the numerical columns scaled to z-scores
    """
    print('# Standardize')
    df_y = df1[["fraud"]]
    print(df_y)
    df_x = df1.drop(columns=["fraud"])
    print(df_x)
    scale = MinMaxScaler().fit(df_x, df_y)
    save_path = os.path.join(pickle_save_path, 'scale_time.pickle')
    print('Pickle save path', save_path)
    with open(save_path, "wb") as f:
        pickle.dump(scale_time, f)

    x_transform = scale.transform(df_x)
    print(x_transform)

    return x_transform


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))





#dataFrame 
#recupération via mongodb
df = load_data(279661)
df = df.dropna()
#df.index = df.reference
#recupération via fichier csv 
#df = pd.read_csv('transaction_simplon.csv', error_bad_lines=False)
print(df)

df1 = preprocess(df)
train_size = int(len(df) * 0.9499)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)

print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)
y_train = train["fraud"]
y_test = test["fraud"]
X_train1 = train.drop(columns=['fraud'])
X_test1 = test.drop(columns=['fraud'])
    
scale_time = MinMaxScaler().fit(X_train1,y_train)

save_path = os.path.join('/Users/charleshajjar/Downloads/api-fraud-temporel-main', 'scale_time.pickle')
print('Pickle save path', save_path)
with open(save_path, "wb") as f:
    pickle.dump(scale_time, f)

    
X_train = scale_time.fit_transform(pd.DataFrame(X_train1))
X_test = scale_time.transform(pd.DataFrame(X_test1))


# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)
#save val for testing  out space train
#print('export de val en csv pour test')
#val.to_csv('test_min.csv') 
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)
    
   
print(f"X Training shape: {X_train.shape}")
print(f"y Training shape: {y_train.shape}")
print(f"X Testing shape: {X_test.shape}")
print(f"y Testing shape: {y_test.shape}")
print(y_test)



class GatedActivationUnit(keras.layers.Layer):
    def __init__(self, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)

    def call(self, inputs):
        n_filters = inputs.shape[-1] // 2
        linear_output = self.activation(inputs[..., :n_filters])
        gate = keras.activations.sigmoid(inputs[..., n_filters:])
        return self.activation(linear_output) * gate


def wavenet_residual_block(inputs, n_filters, dilation_rate):
    """
    TODO commenter
    """
    z = Conv1D(
        2 * n_filters, kernel_size=1, padding="causal", dilation_rate=dilation_rate
    )(inputs)
    z = GatedActivationUnit()(z)
    #z = Bidirectional(LSTM(n_filters,kernel_initializer="uniform", activation="relu"))(z)
    # `->tanh instead of relu for cudnn optimization.

    z = Bidirectional(LSTM(n_filters, activation="relu", return_sequences=True))(z)
    z = GatedActivationUnit()(z)
    z = Conv1D(n_filters, kernel_size=1)(z)
    return keras.layers.Add()([z, inputs]), z


# create model
n_layers_per_block = 3  # 10 in the paper
n_blocks = 3  # 3 in the paper
n_filters = 32  # 128 in the paper

visible1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
z = Conv1D(filters=32, kernel_size=1, strides=2, padding="valid")(visible1)
skip_to_last = []
for dilation_rate in [2 ** i for i in range(n_layers_per_block)] * n_blocks:
    z, skip = wavenet_residual_block(z, n_filters, dilation_rate)
    skip_to_last.append(skip)
z = keras.activations.relu(keras.layers.Add()(skip_to_last))
pool11 = Dropout(rate=0.2)(z)
pool12 = Bidirectional(GRU(60,kernel_initializer="uniform" ,activation="relu", return_sequences=True))(pool11)
flat1 = TimeDistributed(keras.layers.Dense(X_train.shape[2]))(pool12)

model = Model(inputs=[visible1], outputs=flat1)
print('Model summary', model.summary())
#plot_model(model, to_file='wavenetBLSTM.png')
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[rounded_accuracy])
model.fit(
    X_train, y_train, epochs=10, batch_size=100, validation_split=0.2, shuffle=False,
)

X_pred = model.predict(X_test)
print(X_pred)
loss =np.mean(np.abs(X_pred-X_test))
print(loss)
model.save('KILLIA_W')
#create_model(X_train, y_train)
# Convert the model.
#converter = tf.lite.TFLiteConverter.from_keras_model(recurrent_ae)
#tflite_model = converter.convert()

# Save the model.
#with open('recurrent_ae.tflite', 'wb') as f:
#  f.write(tflite_model)
