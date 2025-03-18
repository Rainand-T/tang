#from keras.models import Sequential
import tensorflow as tf

import tensorflow_datasets as tfds

#import tensorflow.keras
from tensorflow import keras

from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random 
from PIL import Image
#from tensorflow.keras.optimizers import Adam





def createTrainingData(training,CATEGORIES):
    i=0
    for category in CATEGORIES:
        for filename in category:
            path = path_test + '/'+ filename
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                img_path = path + '/' + img
                img_array = cv2.imread(img_path)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training.append([new_array, class_num])
                i += 1
                print(str(i)+"完成")

def cnn_create():
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_size, img_size, 3)),
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (5,5), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(128, (5,5), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(5,  activation=tf.nn.softmax)
    ])
    return model

batch_size = 16
nb_classes = 5
nb_epochs = 50
img_rows, img_columns = 200, 200
img_channel = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3

#responder (Pz-10)
path_test = "spec/不同人"
group_1 = ["HC/FCz"]
group_2 = ["P/FCz"]
CATEGORIES = [group_1, group_2]
img_size = 200

training = []
createTrainingData(training,CATEGORIES)
random.shuffle(training)

X =[] #features
y =[] #labels
for features, label in training:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)
y = np.array(y)
y = y.astype('int')

# normalization
X = X.astype('float32')/255
from tensorflow.keras.utils import to_categorical
Y = to_categorical(y, 5)

# devide the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

# create model
model_Pz = cnn_create()

# train model
#optimizer = keras.optimizers.Adam()
model_Pz.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['acc'])
history_Pz = model_Pz.fit(X_train, y_train, batch_size = batch_size, epochs = nb_epochs, verbose = 1, validation_data = (X_test, y_test))

# accuracy
score = model_Pz.evaluate(X_test, y_test, verbose = 0 )
print("Test Loss: ", score[0])
print("Test accuracy: ", score[1])

# acc plot
plt.ylim(0,1)
show_data1 = history_Pz.history['acc']
show_data2 = history_Pz.history['val_acc']
x_data = list(range(1,len(show_data1)+1))
ln1, = plt.plot(x_data,show_data2,color='red',linewidth=3.0,linestyle='--')
plt.legend(handles=[ln1],labels=['val_acc'])
plt.show()
