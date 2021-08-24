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
import tensorflow as tf
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
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector
from tensorflow.keras.models import Model
# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU, TimeDistributed, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
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
    df = df.dropna()
    df_y = df["fraud"]
    print(df_y)
    df_x = df.drop(columns=["fraud"])
    print(df_x)

    counters = Counter(df_y)
    print(counters)
    # augmentation data start SMOTE
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    #over = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority')) #1
    #under = RandomUnderSampler(sampling_strategy=0.1) #4
    #steps = [("o", over)]
    pipeline = Pipeline(steps)
    # transform the dataset
    df_x1, df_y1 = pipeline.fit_resample(df_x, df_y)
    # summarize the new class distribution
    counter = Counter(df_y1)
    print(counter)

    # Convert integer valued (numeric) columns to floating point
    numeric_columns = df_x1.select_dtypes(["int64", "float64"]).columns
    df_x1[numeric_columns] = df_x1[numeric_columns].astype("float32")
    df = pd.concat([df_x1, df_y1], axis=1)
    print("dataframe apres factorisation verification de fraud")
    print(df["fraud"].value_counts())
    print(df.dtypes)

    return df

def standardize(df):

    """
    Scales numerical columns using their means and standard deviation to get
    z-scores: the mean of each numerical column becomes 0, and the standard
    deviation becomes 1. This can help the model converge during training.
    Args:
      dataframe: Pandas dataframe
      pickle_save_path: folder path to save the MinMaxScaler pickle in
    Returns:
      Input dataframe with the numerical columns scaled to z-scores
    >>> df2 = standardize(df1)
        
    >>> df2.head()
             t1  fraud
    0  0.150552    0.0
    1 -0.214917    0.0
    2 -0.243870    0.0
    3 -0.101539    0.0
    4 -0.132422    0.0
      
    """
    print('# Standardize')
    df_y = df1[["fraud"]]
    dfa = df1.drop(columns=["fraud"])
    minmax = MinMaxScaler().fit(dfa)

    save_path = os.path.join('/Users/charleshajjar/Downloads/api-fraud-temporel-main', 'minmax.pickle')
    print('Pickle save path', save_path)
    with open(save_path, "wb") as f:
        pickle.dump(minmax, f)

    x_transform = minmax.transform(dfa)
    print(x_transform)

    new_data_transforme = pd.DataFrame(
        data=x_transform,
        columns=["client_id","site_name","card_type","reference","ip_country","amount"],)
    #dfa= new_data_transforme["amount"]
    new_data_transforme2 = new_data_transforme.drop(columns=["amount"])
    df3 = new_data_transforme2
    # tranforme Principale composant analyse
    pca = PCA(n_components=1)
    principalComponents = pca.fit(df3)
    save_path = os.path.join('/Users/charleshajjar/Downloads/api-fraud-temporel-main', 'pca.pickle')
    print('Pickle save path', save_path)
    with open(save_path, "wb") as f:
        pickle.dump(pca, f)
    principalComponents = pca.transform(df3)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = [ 't1',])
    print('before concat df3', df3.value_counts())
    df4 = pd.concat([principalDf, df_y], axis=1)
    # supression des nan fait par le min et max
    df4 = df4.dropna()
    print('after concat', df4)
    # Assign the first samples to new dataframe

    print(df4.fraud.value_counts())

    return df4


def create_model(X, y):
    """
    DEEP LEARNING MODEL CONVOLUTIONNAL NEURONAL NETWOK AND LSTM LONG TIME MEMORY

    >>> create_model(X_train, y_train)

    """
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    # create model
    #####################################################################################################    
    # Constructing the CNN & training phase
    # Select the type of the model
    model = Sequential()
    # Add the first Dense layer with 20 neuron units and ReLu activation function
    model.add(Dense(units=200,
                input_dim=1,
                kernel_initializer='uniform',
                activation='relu'))
    # Add Dropout to increase or decrease depending on the number of neurons and the data
    model.add(Dropout(0.5))
 
  
    # Add the second Dense layer with 20 neuron units and ReLu activation function
    model.add(Dense(units=200,
                kernel_initializer='uniform',
                activation='relu'))
    # dropout 9/10 desactivate random neuronal
    model.add(Dropout(rate=0.5))
    # Repeate Vecture change to layer CONVID mask  sequence
    model.add(RepeatVector(X_train.shape[1]))
    # Temporal long time memory layer 200 neurone
    model.add(LSTM(20, return_sequences=True))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))
    #Flattens
    model.add(Flatten())
    


    # Add the second Dense layer with 1 neuron units and Sigmoid activation function
    model.add(Dense(units=1,
                kernel_initializer='uniform',
                activation='sigmoid'))

    print('Model summary', model.summary())

    # leaning rate optimizer exponentialDecay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
    #optimizer Adam low learning rate O,1 and exponentiel
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer Adam  UpPer
    # Training 1
    opt = keras.optimizers.Adam(learning_rate=0.09)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.fit(
        X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, shuffle=False,     
         use_multiprocessing=True, workers=23
    )
    # Save model path and Name et N* Version

    # save format tensoflow , format keras, asset, variable register
    #tf.saved_model.save(save_path_model)
    model.save('KILLA')
    model = load_model('KILLA')

    scores = model.evaluate(X_val, y_val)

    # Print out the accuracy
    print("predictions model en cours")
    print("\n")
    print("Accuracy=", scores[1])
    
    reconstructed_model = keras.models.load_model("KILLA")

    # Let's check:
    np.testing.assert_allclose(
    model.predict(X_val), reconstructed_model.predict(X_val)
    )
    #prediction model de base 
    prediction = model.predict(X_val)

    #prediction model reconstruction 
    # testes controle load model 
    print("precdiction reconstruction model keras KillA0I")
    predict = reconstructed_model.predict(X_val)
    scores2 = reconstructed_model.evaluate(X_val, y_val)
    # Print out the accuracy reconstruction model
    print("\n")
    print("Accuracy2=", scores2[1])
    print(predict)

    # show the inputs and predicted outputs
    for i in range(len(X_val)):
            print("X=%s, Predicted=%s" % (X_val[i], prediction[i]))
    predictionss= model.predict_classes(X_val)
    print("prections by prediction_classes")
    print(predictionss)

    print("prediction by formula argmax")
    predictions = np.argmax(prediction, axis=1)
    print(predictions)
    print("reel valeur y")
    
    print(y_val)

    df_ans = pd.DataFrame({'Real Class': y_val})
    df_ans['Prediction'] = predictionss

    df_ans['Prediction'].value_counts()

    df_ans['Real Class'].value_counts()

    cols = ['Real_Class_1', 'Real_Class_0']  # Gold standard
    rows = ['Prediction_1', 'Prediction_0']  # Diagnostic tool (our prediction)

    B1P1 = len(df_ans[(df_ans['Prediction'] == df_ans['Real Class']) & (df_ans['Real Class'] == 1)])
    B1P0 = len(df_ans[(df_ans['Prediction'] != df_ans['Real Class']) & (df_ans['Real Class'] == 1)])
    B0P1 = len(df_ans[(df_ans['Prediction'] != df_ans['Real Class']) & (df_ans['Real Class'] == 0)])
    B0P0 = len(df_ans[(df_ans['Prediction'] == df_ans['Real Class']) & (df_ans['Real Class'] == 0)])

    conf = np.array([[B1P1, B0P1], [B1P0, B0P0]])
    df_cm = pd.DataFrame(conf, columns=[i for i in cols], index=[i for i in rows])
    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(df_cm, annot=True, ax=ax, fmt='d')

    # Making x label be on top is common in textbooks.
    ax.xaxis.set_ticks_position('top')

    print('Total number of test cases: ', np.sum(conf))
   
# Model summary function MATRICE DE CONFUSION
    def model_efficacy(conf):
        total_num = np.sum(conf)
        sen = conf[0][0] / (conf[0][0] + conf[1][0])
        spe = conf[1][1] / (conf[1][0] + conf[1][1])
        false_positive_rate = conf[0][1] / (conf[0][1] + conf[1][1])
        false_negative_rate = conf[1][0] / (conf[0][0] + conf[1][0])

        print('Total number of test cases: ', total_num)
        print('G = gold standard, P = prediction')
        
        # G = gold standard; P = prediction
        print('G1P1: ', conf[0][0])
        print('G0P1: ', conf[0][1])
        print('G1P0: ', conf[1][0])
        print('G0P0: ', conf[1][1])
        print('--------------------------------------------------')
        print('Sensitivity: ', sen)
        print('Specificity: ', spe)
        print('False_positive_rate: ', false_positive_rate)
        print('False_negative_rate: ', false_negative_rate)
        
    print(model_efficacy(conf))






#dataFrame 
#recupération via mongodb
df = load_data(279661)
df = df.dropna()
#df = data["client_id","site_name","card_type","reference","ip_country","amount","fraud"]
#recupération via fichier csv 
#df = pd.read_csv('transaction_simplon.csv', error_bad_lines=False)
print(df)
#df = df.drop(['Unnamed: 0'], axis=1)
#df.sort_values(by='fraud', ascending=False, inplace=True)
#testing pandas 
df1 = preprocess(df)
assert_frame_equal(df1, df1)
df2 = standardize(df1)
assert_frame_equal(df2, df2)
#testing pandas 

# It is obvious that this data set is highly unbalance
# It Easier to sort the datset by "class" for stratified sampling
# Assign the first "1000" samples to new dataframe
df2.sort_values(by='fraud', ascending=False, inplace=True)
#df2 = df2.iloc[:800000, :]
df2 = shuffle(df2, random_state=42)
print(df2["fraud"].value_counts())
# split train test
train_size = int(len(df2) * 0.9499)
val_size = len(df2) - train_size
train, val = df2.iloc[0:train_size], df2.iloc[train_size : len(df2)]
print("train count fraud")
print(train["fraud"].value_counts())
print("validation data count fraud")
print(val["fraud"].value_counts())
#save val for testing  out space train
#print('export de val en csv pour test')
#val.to_csv('test_min.csv') 

# Spilt each dataframe into "feature" & "lable"
X_train = np.array(train.values[:, 0:1])
y_train = np.array(train.values[:, -1])
X_val = np.array(val.values[:, 0:1])
y_val = np.array(val.values[:, -1])
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.float32)


print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_train)
print(y_val)


#train and test
create_model(X_train, y_train)

#if __name__ == '__main__':
    #import doctest
    #doctest.testmod()