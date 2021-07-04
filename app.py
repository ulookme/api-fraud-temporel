from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import pickle
import json
import pymongo
from pymongo import MongoClient
#from pandas.core.frame import DataFrame
#import sklearn
from keras.models import load_model
#assert sklearn.__version__ >= "0.20"
import pandas as pd
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import sys, os
from time import process_time_ns
#from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
assert sys.version_info >= (3, 5)
from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy  import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import flask_monitoringdashboard as dashboard
import os
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate


app = Flask(__name__)
dashboard.bind(app)
app.config.from_object(Config)


app.config['SECRET_KEY'] = 'secretest'
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# set the database to be use
#SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URI') or \
# 'sqlite:///' + os.path.join(basedir, 'test2.db')
 
#app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI


Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin,db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')


class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard1'))

        return '<h1>Invalid username or password</h1>'    

        #return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'

    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return '<h1>New user has been created!</h1>'
        #return '<h1>' + form.username.data + ' ' + form.email.data + ' ' + form.password.data + '</h1>'

    return render_template('signup.html', form=form)

@app.route('/dashboard1')
@login_required
def dashboard1():
    return render_template('index2.html', name=current_user.username)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/login')
def next():
    return redirect(url_for('index'))

######################################################################

model = load_model('KILLA')
model2 = load_model('KILLA_ENCODE.h5')
minmax = pickle.load(open('minmax.pickle', 'rb'))
#scale_time = pickle.load(open('scale_time.pickle', 'rb'))
print("minmax")
print(minmax)
pca = pickle.load(open('pca.pickle', 'rb'))
print("import pca")
print(pca)


client = pymongo.MongoClient("mongodb+srv://transac:Mhajjar3@cluster0.hskyz.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client.transaction
col = db.collec1


def preprocess_transaction(x):
        """
        Factorize categorical string data to int columns.
        >>> preprocess_transaction(df)

        """
        # Convert categorical columns to numeric
        x["ip_country"], uniques_country = pd.factorize(x["ip_country"])
        x["card_bank"], uniques_card_bank = pd.factorize(x["card_bank"])
        x["site_name"], uniques_site_name = pd.factorize(x["site_name"])
        x["reference"], uniques_ref = pd.factorize(x["reference"])
    

        # Convert integer valued (numeric) columns to floating point
        numeric_columns = x.select_dtypes(["int64", "float64"]).columns
        x[numeric_columns] = x[numeric_columns].astype("float32")

        print(f"{x.dtypes=}")
        print(f"{x=}")
        return x

def preprocess_encode(x):
    
    # Convert integer valued (numeric) columns to floating point
    numeric_columns = x.select_dtypes(["int64", "float64"]).columns
    x[numeric_columns] = x[numeric_columns].astype("float32")

    print(f"{x.dtypes=}")
    print(f"{x=}")
    return x

def standardize_encode(x):

    """
    Scales numerical columns using their means and standard deviation to get

    """
    print('# Standardize')
    x = pd.DataFrame(x, columns = ['client_id','site_name','card_type','reference','ip_country','amount'])
    df_x = x
    print(df_x)
    x_transform = minmax.transform(df_x)
    print(x_transform)
    # reshape inputs for LSTM [samples, timesteps, features]
    data = x_transform.reshape(x_transform.shape[0], 1, x_transform.shape[1])
    

    
    return data




def standardize(x):

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
    x = pd.DataFrame(x, columns = ['client_id','site_name','card_type','reference','ip_country','amount'])
    df_x = x
    print(df_x)
    x_transform = minmax.transform(df_x)
    print(x_transform)

    new_data_transforme = pd.DataFrame(
        x_transform,
        columns=[
            "reference",
            "ip_country",
            "card_bank",
            "client_id",
            "site_name",
            "amount",
        ],
    )
    #dfa= x_transform[5]
    
    dfa= new_data_transforme["amount"]
    new_data_transforme2 = new_data_transforme.drop(columns=["amount"])
    df3 = new_data_transforme2
    # tranforme Principale composant analyse
    #pca = PCA(n_components=1)
    principalComponents = pca.transform(df3)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = [ 't1'])
    #print('before concat df3', df3.value_counts())
    #df4 = pd.concat([principalDf,dfa], axis=1)
    #dfa= new_data_transforme["amount"]
    #new_data_transforme2 = new_data_transforme.drop(columns=["amount"])
    #df3 = new_data_transforme2
    # tranforme Principale composant analyse
    #pca = PCA(n_components=2)
    #principalComponents = pca.fit_transform(df3)
    #principalDf = pd.DataFrame(data = principalComponents
             #, columns = [ 't1', 't2'])
    #print('before concat df3', df3.value_counts())
    #df4 = pd.concat([principalDf,dfa], axis=1)
    # supression des nan fait par 

    #df4 = df4.dropna()
    #print('after concat', df4)
    # Assign the first samples to new dataframe
    #print('fin standardise')
    #print(df4)

    return np.array(principalDf)

def dataframe1(x):

    print('#  dataframe')
    x = pd.DataFrame(x, columns = ['client_id','site_name','card_type','reference','ip_country','amount'])
    return x 



def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 6)
    return to_predict



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    and save input in database mogodb 
    '''
    # connected to database mogodb
    #client = pymongo.MongoClient("mongodb+srv://transac:Mhajjar3@cluster0.hskyz.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    #db = client.transaction
    #col = db.collec1
    #request.form.values for predicted model
    int_features = request.form.values()
    #request.form.to_dict for transforme in daframe to save input
    int_features2 = request.form.to_dict()
    to_predict_list = list(int_features2.values())
    #print(to_predict_list)
    to_predict_list = ValuePredictor(to_predict_list)
    #print(to_predict_list)
    df = dataframe1(to_predict_list)
    print(df)
    records = df.to_dict(orient='records') 
    print(records)
    # insert data in collection mogodb
    col.insert_many(records)
    # standardize form.value to prédict
    df2 = standardize(df)
    print("standiser pca")
    print(df2)
    final_features = [df2]
    print('rendu finale')
    print(final_features)
    #prédiction du model1
    prediction = model.predict(final_features)
    df3 = dataframe1(to_predict_list)
    df3 = standardize_encode(df3)
    print(df3)
    final_features_encode = [df3]
    print(final_features_encode)
    #prediction du model2
    prediction2 = model2.predict(final_features_encode )
    output = prediction[0]
    output1 = prediction2[0]
    output2 =np.mean(np.abs(output1-final_features_encode))
    output2 = round(output2 , 2)
    seuile_encode = 0.4
    seuile_classification = 0.7
    #output3 = "comportement"
    #if output2 > seuile_encode:
        #return render_template('index2.html', prediction_text='Score is Fraude be % {}'.format(output2))
    #if output2 < seuile_encode :
        #return render_template('index2.html', prediction_text='Score no fraud % {}'.format(output2))
    if output2 > seuile_encode and output > seuile_classification:
        return render_template('index2.html', prediction_text='reel fraud % {}'.format(output2))
    elif output2 < seuile_encode and output < seuile_classification:
        return render_template('index2.html', prediction_text='légitime % {}'.format(output2))
    else:

        return render_template('index2.html', prediction_text='Score Fraude be % {}'.format(output2))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    #app.run(host='0.0.0.0',port=5000)
    app.run(debug=False)
