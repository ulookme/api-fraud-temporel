# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys, os
import sklearn
import pymongo
from pymongo import MongoClient
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas.core.frame import DataFrame
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.pipeline import Pipeline
from keras.layers.convolutional import Conv1D
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.layers import TimeDistributed
from keras.models import Model
from keras import regularizers
from keras.layers  import  LSTM
from keras.optimizers import Adam
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector
from keras.models import Model
import tensorflow as tf



from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
# Multiple Inputs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
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
#from tensorflow.keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import load_model
import math, sys, time
# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU, TimeDistributed, LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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
    col = db.collec1
    data = pd.DataFrame(list(col.find()))
    data = data.drop(columns=["_id"])
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



def create_model(X, y):
    """
    TODO commenter
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    tf.random.set_seed(42)

    # create model
    #####################################################################################################    
    # Constructing the CNN & training 
    # Select the type of the model
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = tf.keras.layers.LSTM(30, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = tf.keras.layers.LSTM(30, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = tf.keras.layers.LSTM(30, activation='relu', return_sequences=True)(L3)
    L5 = tf.keras.layers.LSTM(10, activation='relu', return_sequences=True)(L4)
    output = Dense(1, activation="sigmoid")(L5)    
    model = Model(inputs=inputs, outputs=output)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
    #optimizer Adam low learning rate O,1 and/home/hajjar/Documents/FRAUD/PROJET_KILLA exponentiel
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)  


    model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=[rounded_accuracy])

    print('Model summary', model.summary())

    model.fit(
        X_train, y_train, epochs=10, batch_size=10, validation_split=0.05, shuffle=False,     
         use_multiprocessing=True, workers=23
    )
    # Save model path and Name et N* Version
    # save format tensoflow , format keras, asset, variable register
    #tf.saved_model.save(save_path_model)
    #model.save('KILLA_time')
    #model = load_model('KILLA_time')
    scores = model.evaluate(X_test, y_test)

    # Print out the accuracy
    print("predictions model en cours")
    print("\n")
    print("Accuracy=", scores[1])


    X_pred = model.predict(X_test)
    print(X_pred[:1])
    R =np.mean(np.abs(X_pred-X_test))
    
    

    #model.fit(
        #X_test, y_test, epochs=2, batch_size=10, validation_split=0.05, shuffle=False,     
         #use_multiprocessing=True, workers=23
    #)
    model.save('KIllA_time')



#dataFrame 
#recupération via mongodb
df = load_data(100000)
df = df.dropna()
#df.index = df.reference
#recupération via fichier csv 
#df = pd.read_csv('transaction_simplon.csv', error_bad_lines=False)
print(df)

df1 = preprocess(df)
train_size = int(len(df) * 0.90)
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

save_path = os.path.join('/home/hajjar/Documents/FRAUD/PROJET_KILLA/', 'scale_time.pickle')
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

    
   
print(f"X Training shape: {X_train.shape}")
print(f"y Training shape: {y_train.shape}")
print(f"X Testing shape: {X_test.shape}")
print(f"y Testing shape: {y_test.shape}")
print(y_test)


recurrent_encoder = keras.models.Sequential([
    keras.layers.LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    keras.layers.Dropout(rate=0.2),
    keras.layers.LSTM(30),
    keras.layers.Dropout(rate=0.2),
])
recurrent_decoder = keras.models.Sequential([
    keras.layers.RepeatVector(1, input_shape=[30]),
    keras.layers.LSTM(100, return_sequences=True),
    keras.layers.Dropout(rate=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense((X_train.shape[2]), activation="sigmoid"))
])
recurrent_ae = keras.models.Sequential([recurrent_encoder, recurrent_decoder])
recurrent_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(0.1),
                     metrics=['accuracy'])


history = recurrent_ae.fit(X_train, y_train,batch_size=10 ,epochs=10, validation_data=(X_test, y_test))

X_pred = recurrent_ae.predict(X_test)
print(X_pred)
loss =np.mean(np.abs(X_pred-X_test))
print(loss)
recurrent_ae.save('KILLA_ENCODE')
#create_model(X_train, y_train)
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(recurrent_ae)
tflite_model = converter.convert()

# Save the model.
with open('recurrent_ae.tflite', 'wb') as f:
  f.write(tflite_model)

