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
from time import sleep
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print("Mascara Ideal")
MaskIdeal = np.load("MaskIdeal.npz",mmap_mode = 'r')
MaskIdeal = [MaskIdeal[k] for k in MaskIdeal]
print("FIM")

print("Espectograma Mix")
EspecMix = np.load("EspecMix.npz",mmap_mode = 'r')
EspecMix = [EspecMix[k] for k in EspecMix]
print("FIM")

print("Separando dataset")

data = EspecMix
labels = MaskIdeal

indices = np.arange(4300)
(
    data_train,
    data_test,
    labels_train,
    labels_test,
    indices_train,
    indices_test,
) = train_test_split(data, labels, indices, test_size=0.2, random_state=10)
