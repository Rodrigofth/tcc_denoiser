# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bdGcX7eXY3EOZP-8wHP-4Me2d0heTwPs
"""

from numpy.lib.twodim_base import mask_indices

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
from PIL import Image


 
def mix_sound(audio, ruido):
  
  if len(audio) >= len(ruido):
    while len(audio) >= len(ruido):
      ruido = np.append(ruido, ruido)
 
  ind = np.random.randint(0, ruido.size - audio.size)
  ruido = ruido[ind: ind + audio.size]
  
  mix = audio + ruido
  
  return mix,ruido

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#CALCULA O SNR DOS SINAIS 
def snr_sinal(audio, ruido):
  v1 = np.std(audio)
  v2 = np.std(ruido)
    
  snr =  20 * np.log10(v1/v2)
  return snr

#-------------------------------------------------------------------------------
#CALCULA O ESPECTOGRAMA DO SINAL
def spec_signal(s):
  pontosFFT = 256
  windowLen = 256
  Overlap = 64
  window = sci.signal.windows.hamming(256, sym=False)

  spec = librosa.stft(s, n_fft = pontosFFT, win_length = windowLen, hop_length = Overlap, 
                       window = window, center = True)
  return spec
#-------------------------------------------------------------------------------
#CALCULA O ESPECTOGRAMA INVERSO
def inv_spec(s):
  pontosFFT = 256
  windowLen = 256
  Overlap = 64
  window = sci.signal.windows.hamming(256, sym=False)
  specInv = librosa.istft(s, hop_length=Overlap, win_length=windowLen, 
                          window=window, center=True)
  return specInv


#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#Mascara Ideal Binaria
def Mask_Ideal_Bin(specV, specN, snr):
  FC = snr + 5
  a = []
  a = librosa.amplitude_to_db(np.abs(specV),ref=np.max)
  Mask = a
  for j in range(129):
    i = 0
    for i in range(len(specV[0])):
      if specN == 0:
        RSR = 0
        print('Divisao por Zero')
      
      RSR =  20*np.log10(specV[j,i] / specN[j, i])  
      if RSR >= FC:
        Mask[j][i] = 1
      if RSR < FC:
        Mask[j][i] = 0
  return Mask
#------------------------------------------------------------------------------



def Get_Features():
    

  print("Ruidos")
  target_url = 'https://drive.google.com/uc?id=1JW2HncltSnw2i0PsEnCt83KezvqpS5uS'#https://drive.google.com/file/d/1JW2HncltSnw2i0PsEnCt83KezvqpS5uS/view?usp=sharing
  gdown.download(target_url)
  Noise = np.load("Noises8k.npz",mmap_mode = 'r')
  Noise = [Noise[k] for k in Noise]
              
  print("Audios")
  target_url = 'https://drive.google.com/uc?id=1oI9TGTgt3IpI9P4auDmNCBMLKClG8sHZ'#https://drive.google.com/file/d/1oI9TGTgt3IpI9P4auDmNCBMLKClG8sHZ/view?usp=sharing
  gdown.download(target_url)
  Voice = np.load("Voice8k.npz",mmap_mode = 'r')
  Voice = [Voice[k] for k in Voice]

#Mixa audios limpos com os segmentos equivalentes dos ruidos
  Mix = []
  NoiseSegment = []
  for i in range(4300):
    v_mix, ruidoSegment = mix_sound(Voice[i], Noise[i])
    print("Mixando Audio com Ruido: ", i)
    Mix.append(v_mix)
    NoiseSegment.append(ruidoSegment)
#Calcula o espectograma dos audio, ruido e do mix
  specVoice = []
  for i in range(4300):
    v = spec_signal(Voice[i])
    print("Espectograma Voice: ",i)
    specVoice.append(v)
  
  specNoise = []
  for i in range(4300):
    r = spec_signal(NoiseSegment[i])
    print("Espectograma Noise: ",i)
    specNoise.append(r)

  specMix = []
  for i in range(4300):
    m = spec_signal(Mix[i])
    print("Espectograma Mixado: ",i)
    specMix.append(m)

#SNR do Mix
  SNR = []
  for i in range(4300):
    RSR = snr_sinal(Voice[i], NoiseSegment[i])
    print("SNR Voice/Noise: ",i)
    SNR.append(RSR)

#Mascara Binaria Ideal
  specMask = []
  for i in range(4300):
    Mask = Mask_Ideal_Bin(specVoice[i], specNoise[i], SNR[i])
    specMask.append(Mask)
    print("Mask ideal: ",i)

  return Mix,specVoice, specNoise, specMix, SNR, specMask


vM, spcV, spcN, spcM, snr, IBM = Get_Features()

print('Relação sinal ruido')
np.savez('SNR.npz', *snr)
print("FIM")

print('specVoice')
np.savez("EspecVoice.npz", *spcV)
print("FIM")

print('specNoise')
np.savez('EspecNoise',*spcN)
print("FIM")

print('specMix')
np.savez('EspecMix',*spcM)
print("FIM")

print('Mascara Binaria Ideal')
np.savez('MaskIdeal.npz', *IBM)
print("FIM")
print("FIM")
print("FIM")

for i in range(10):
  print(i)
