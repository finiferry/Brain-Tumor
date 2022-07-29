

from flask import Flask,render_template,request,make_response,jsonify,session,redirect,flash
app = Flask(__name__)
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt(app)
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import os
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from keras_efficientnets import EfficientNetB3
# from keras_efficientnets import EfficientNetB3

from tensorflow.keras.applications import EfficientNetB2,EfficientNetB3,EfficientNetB5,InceptionResNetV2#,EfficientNetV2S
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import keras
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
#from pretty_confusion_matrix import pp_matrix
#confusion_matrix = metrics.confusion_matrix(y_true=y_test_labels, y_pred=pred_labels)  # shape=(12, 12)
from sklearn.metrics import confusion_matrix

#create labels
classes=[]
filename='dataset'
for sub_folder in os.listdir(os.path.join(filename,'Training')):
    classes.append(sub_folder)

#resize images and put together Training and Testing folder

X_train = []
y_train = []
image_size = 160
for i in classes:
    path_train = os.path.join(filename,'Training',i)
    for j in tqdm(os.listdir(path_train)): #Instantly make your loops show a smart progress meter 
        img = cv2.imread(os.path.join(path_train,j))
        img = cv2.resize(img,(image_size, image_size))
        X_train.append(img)
        y_train.append(i)
    path_test = os.path.join(filename,'Testing',i)
    for j in tqdm(os.listdir(path_test)):
        img = cv2.imread(os.path.join(path_test,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)
        
X_train = np.array(X_train)
y_train = np.array(y_train)    


#data augmentation
X_train, y_train = shuffle(X_train,y_train, random_state=42)
datagen = ImageDataGenerator(
    rotation_range=7, #rotate images
    width_shift_range=0.05,
    height_shift_range=0.05, #shift image in horizontal and vertical
    zoom_range=0.1, #zoom images
    horizontal_flip=True)

datagen.fit(X_train)
X_train.shape
lb = LabelEncoder()

#train and test splitting 
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.15,random_state=42,stratify=y_train)

labels_train=lb.fit(y_train)
y_train=lb.transform(y_train)
y_test=lb.transform(y_test)





#load EfficientNet
EfficientNet=EfficientNetB3(weights='imagenet', include_top=False,input_shape=(160,160,3))
#train the model
tf.random.set_seed(45)
model = EfficientNet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.55)(model)
model = tf.keras.layers.Dense(60,activation='elu',kernel_initializer='GlorotNormal')(model)
model = tf.keras.layers.Dropout(rate=0.3)(model)
model = tf.keras.layers.Dense(4,activation='softmax')(model)
model = tf.keras.models.Model(inputs=EfficientNet.input, outputs = model)
opt = Adam(
    learning_rate=0.000016, beta_1=0.91, beta_2=0.9994,
    epsilon=1e-08)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 


early_stopping_cb=keras.callbacks.EarlyStopping(patience=9,restore_best_weights=True)


 #load the model
model=keras.models.load_model('EfficientNetB3.h5')  





#-----------------------------------------------DATABASE-------------------------------------------------------------------
import sqlite3
conn = sqlite3.connect('mysqlite.db',check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS register
             (username text,email text,password text)''')			
conn.commit()
conn.close()


#----------------------------------------------------------------------------------------------------------------------------

# Model saved with Keras model.save()






 
@app.route('/',methods=['GET','POST'])
def Login():
    if request.method == 'POST':
        data=request.get_json()
        if data['which_condiction']=="signup":
            print("signup")
            conn = sqlite3.connect('mysqlite.db',check_same_thread=False)
            c = conn.cursor()
            c.execute("SELECT * FROM register")
            for query_result in c.fetchall():
                if data['username'] in query_result:
                    return "exists"
                else:  
                    pass   
            password = bytes(data['password'], 'utf-8')
            password = hashing(password)
            data['password']=password
            c.execute("""INSERT INTO register (username,email,password) values (?,?,?)""",(data['username'],data['email'],data['password']))
            conn.commit()
            return "success"
        elif data['which_condiction']=="login":
            print("login")
            conn = sqlite3.connect('mysqlite.db',check_same_thread=False)
            c = conn.cursor()
            c.execute("SELECT * FROM register WHERE username=?", ( data['username'],))
            result = c.fetchall()
            if len(result)==0:     
                return "no"
            for i in result:
                if verify_pass(i[2],data['password']):
                    return "success"
                else:
                    return "error"

    return render_template("login.html")


@app.route('/home/',methods=['GET','POST'])
def Home():
    if request.method == 'POST':
        file = request.files['f']  
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread('output/'+filename)
        img = cv2.resize(img,(image_size,image_size))
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        output = model.predict(img)
        output=np.argmax(output)
        if output==0:
            flash("glioma tumor")
        elif output==1:
            flash("meningioma tumor")
        elif output==2:
            flash("no tumor")
        elif output==3:
            flash("pituitary tumor")
            
        return render_template("home.html", uploaded_image=filename)
    
    return render_template("home.html")



@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

def hashing(password):
    pw_hash = bcrypt.generate_password_hash(password)
    return pw_hash
def verify_pass(password,password1):
    return bcrypt.check_password_hash(password,password1)


if __name__=="__main__":
    app.run()
    app.run(debug=True)