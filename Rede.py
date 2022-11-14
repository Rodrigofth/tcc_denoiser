
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
import plotly.graph_objects as goimport numpy as np
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


#Baixando dataset/labels
!wget "https://brpucrs-my.sharepoint.com/:u:/g/personal/rodrigo_fleith_edu_pucrs_br/ETwYOYhyxSpIrNdoyTEEQO4B02L017F-ktmkg7oJ1BlTEg?e=h3KrCd&download=1" -O data_train.npz
!wget "https://brpucrs-my.sharepoint.com/:u:/g/personal/rodrigo_fleith_edu_pucrs_br/Eb2eZcTvCPJLprEeYrqXoOcBi8s2BkOm_QXj2lxJySs5Uw?e=iHhYqT&download=1" -O data_test.npz
!wget "https://brpucrs-my.sharepoint.com/:u:/g/personal/rodrigo_fleith_edu_pucrs_br/ERPhfoNFYHNCicV5YYSIeXYBYybMavYAqfQ7NCl_v1j53A?e=jZMfE3&download=1" -O labels_test.npz
!wget "https://brpucrs-my.sharepoint.com/:u:/g/personal/rodrigo_fleith_edu_pucrs_br/ERulN2BIYqFPo-uqaPeauBkBq11mt5hx0eweQfGu7TORRw?e=GLsh1Q&download=1" -O labels_train.npz

print('lendo arq npz')
data_test = np.load("data_test.npz",mmap_mode = 'r')
data_train = np.load("data_train.npz",mmap_mode = 'r')
labels_train = np.load("labels_train.npz",mmap_mode = 'r')
labels_test = np.load("labels_test.npz",mmap_mode = 'r')

print('criando variaveis')
data_test = [data_test[k] for k in data_test]
data_train = [data_train[k] for k in data_train]
labels_train = [labels_train[k] for k in labels_train]
labels_test = [labels_test[k] for k in labels_test]

print("explidindo")
data_train_1d, train_index = master_exploder(data_train)
data_test_1d, test_index = master_exploder(data_test)
labels_train_1d, labels_train_index = master_exploder(labels_train)
labels_test_1d, labels_test_index = master_exploder(labels_test)

print("pegando modulo")
data_train_2d = m_2d(data_train_1d)
data_test_2d = m_2d(data_test_1d)
labels_train_2d = m_2d(labels_train_1d)
labels_test_2d = m_2d(labels_test_1d)

data_train_2dT = np.array(data_train_2d).reshape(len(data_train_2d),1,129)
data_test_2dT = np.array(data_test_2d).reshape(len(data_test_2d),1,129)
labels_train_2dT = np.array(labels_train_2d).reshape(len(labels_train_2d),1,129)
labels_test_2dT = np.array(labels_test_2d).reshape(len(labels_test_2d),1,129)

def seq2one(vocab, input_shape):
    lr_schedule = schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=60000, decay_rate=0.9, staircase=True)

    opt = Adam(learning_rate=0.0001)
    opt2 = RMSprop(learning_rate = lr_schedule,momentum=0.6)

    model = Sequential()

    model.add((LSTM(256,input_shape=input_shape)))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(vocab))
    model.add(Activation('sigmoid'))
    model.compile(loss="BinaryCrossentropy",
            optimizer=opt2)
    print("Learning rate inicial: ",model.optimizer._decayed_lr(np.float32).numpy())
    return model
    
def train_model(train_inputs, test_inputs, train_labels, test_labels):
  
  epochs = 50
  batch_size = 64

  vocab = 129
  model = seq2one(vocab, input_shape= (1,129))
  model.fit(np.array(train_inputs),np.array(train_labels),batch_size=batch_size,epochs=epochs)
  print("Learning rate final: ",model.optimizer._decayed_lr(np.float32).numpy())

  return model

model = train_model(data_train_2dT_norm,data_test_2dT_norm,labels_train_2d,labels_test_2d)
