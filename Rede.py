
import numpy as np 
import scipy as sci
import matplotlib.pyplot as plt  
import soundfile as sf
import gdown 
from gdown.download import download
from IPython.display import Audio
import os
import shutil
import zipfile
from zipfile import ZipFile
from time import sleep
import librosa
import numpy as np
import pandas as pd
import librosa.display
import IPython.display
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sklearn.metrics
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense, Activation, Bidirectional
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import MeanSquaredError,CategoricalCrossentropy,BinaryCrossentropy
from tensorflow.keras.activations import sigmoid,tanh,relu,softmax,softplus,softsign,selu,elu
from tensorflow.keras.optimizers import Adam, RMSprop, schedules,Adamax
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
import sys
from scipy.stats import mode
from collections import defaultdict
from itertools import compress
from tensorflow.keras.layers import *
from keras.callbacks import CSVLogger

def seq2one(vocab, input_shape):
    lr_schedule = schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=80e3, decay_rate=0.95, staircase=True)

    opt = Adam(learning_rate=0.0001)
    opt2 = RMSprop(learning_rate = lr_schedule,momentum=0.6)

    model = Sequential()

    model.add(LSTM(1024,input_shape=input_shape))
    model.add(Dense(512,activation="relu"))
   #model.add(tf.keras.layers.Dropout(.7,input_shape=input_shape))
    model.add(Dense(256,activation="relu"))
    #model.add(tf.keras.layers.Dropout(.5,input_shape=input_shape))
    model.add(Dense(vocab))
    model.add(Activation('sigmoid'))
    model.compile(loss="BinaryCrossentropy",
            optimizer=opt2,
            metrics=[precision_m,recall_m,f1_m])
    print("Learning rate inicial: ",model.optimizer._decayed_lr(np.float32).numpy())
    return model
    
def train_model(train_inputs, test_inputs, train_labels, test_labels):
  
  epochs = 100

  batch_size = 64

  vocab = 129
  model = seq2one(vocab, input_shape= (1,129))
  model.fit(np.array(train_inputs),np.array(train_labels),batch_size=batch_size,epochs=epochs,
            callbacks=[TestCallback(np.array(test_inputs), np.array(test_labels)),csv_logger])
  score = model.evaluate(np.array(test_inputs), np.array(test_labels), verbose=1)
  print('Test loss:', score[0])
  print('Test precision:', score[1])
  print('Test recall:', score[2])
  print('Test f1:', score[3])
  results = {'Test loss': score[0],
             'Test precision': score[1],
             'Test recall': score[2],
              'Test f1': score[3]}

  print("Learning rate final: ",model.optimizer._decayed_lr(np.float32).numpy())

  return model, results


class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_inputs,test_labels):
        self.test_inputs = test_inputs
        self.test_labels = test_labels

    def on_epoch_end(self, epoch, logs):
        score = self.model.evaluate(np.array(self.test_inputs), np.array(self.test_labels), verbose=1)
        results = {'Test loss': score[0],
             'Test precision': score[1],
             'Test recall': score[2],
              'Test f1': score[3]}
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def print_scores(sub_ids,sub_scores):
  for n in range(len(sub_ids)):
    print("\nScores for subject: {}\n".format(sub_ids[n]))

  for item in range(len(sub_scores[n])):
    print("Split {}: {}".format(item,np.array(sub_scores[n][item])))

def zero_pad(features):
    """
    zero pad examples to the right until max_len
    """
    shape0 = [item.shape[0] for item in features]
    shape1 = features[0].shape[1]
    max_val = max(shape0)
    pad_values = [max_val - item.shape[0] for item in features]
    for n in range(len(pad_values)):
        if pad_values[n]>0:
            zeros = np.zeros([pad_values[n],shape1])
            features[n] = np.concatenate((zeros,features[n]),axis=0)
    
    return features, max_val

def master_exploder(dataset_list):

  list_1d = []
  index_list = []
  for item in dataset_list:
    item_explodido = np.hsplit(item,item.shape[1]) #explode cada espectrograma em Nx129x1
    list_1d.extend(item_explodido) # adiciona cada entrada 129x1 na lista
    index_list.append(np.array(item_explodido).shape[0]) # salva o valor de N em outra lista para reconstrução

  return list_1d, index_list

def m_2d(dataset_1d):

  data_list_2d = []

  for item in dataset_1d:
    example_list = []
    for example_item in item:
      #example_list.append(np.array([np.abs(example_item[0]),np.imag(example_item[0])]))
      example_list.append(np.array([np.abs(example_item[0])]))
      
      #example_list.append(normalize(example_list))
    data_list_2d.append(np.array(example_list))
 
  return data_list_2d


print('lendo arq npz')
data_test = np.load("/home/rodrigo.fleith/Dataset/data_test.npz",mmap_mode = 'r')
data_train = np.load("/home/rodrigo.fleith/Dataset/data_train.npz",mmap_mode = 'r')
labels_train = np.load("/home/rodrigo.fleith/Dataset/labels_train.npz",mmap_mode = 'r')
labels_test = np.load("/home/rodrigo.fleith/Dataset/labels_test.npz",mmap_mode = 'r')
print('Fim')
print('criando variaveis')

data_test = [data_test[k] for k in data_test]
print('1')
data_train = [data_train[k] for k in data_train]
print('2')
labels_train = [labels_train[k] for k in labels_train]
print('3')
labels_test = [labels_test[k] for k in labels_test]
print('Inicio 1d')
data_train_1d, train_index = master_exploder(data_train)
print('1_1d')
data_test_1d, test_index = master_exploder(data_test)
print('2_1d')
labels_train_1d, labels_train_index = master_exploder(labels_train)
print('3_1d')
labels_test_1d, labels_test_index = master_exploder(labels_test)
print('Inicio 2d')
data_train_2d = m_2d(data_train_1d)
print('Inicio 1_2d')
data_test_2d = m_2d(data_test_1d)
print('Inicio 2_2d')
labels_train_2d = m_2d(labels_train_1d)
print('Inicio 3_2d')
labels_test_2d = m_2d(labels_test_1d)

print('reshape')
data_train_2dT = np.array(data_train).reshape(len(data_train_2dT),1,129)
print('1')
data_test_2dT = np.array(data_test).reshape(len(data_test_2dT),1,129)
print('Incio treino')             
model, results = train_model(data_train_2dT,data_test_2dT,labels_train_2d,labels_test_2d)
csv_logger = CSVLogger("model_history_LAD1.csv", append=True)  
              
fl = 'model_history_LAD1.csv'

csv = pd.read_csv(fl)
