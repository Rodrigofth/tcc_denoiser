# -*- coding: utf-8 -*-
"""Denoiser

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1y8rF0T2EyEJqaKxTZnmVHq01GIjOmqyK
"""

import numpy as np 
import scipy as sci
import matplotlib.pyplot as plt  
import soundfile as sf
import gdown 
from gdown.download import download
from IPython.display import Audio
import os
import shutil
from zipfile import ZipFile
from time import sleep
import librosa
import zipfile
import numpy as np
import pandas as pd
import librosa.display
import IPython.display
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
from PIL import Image

#-------------------------------------------------------------------------------
#PLAY AUDIO
def play(y, sr):  
  return Audio(y, rate = sr)

#-------------------------------------------------------------------------------
#MIXA OS RUIDOS E OS AUDIOS(VOICE)  
def mix_sound(audio, ruido):
  
  if len(audio) >= len(ruido):
    while len(audio) >= len(ruido):
      ruido = np.append(ruido, ruido)
 
  ind = np.random.randint(0, ruido.size - audio.size)
  ruido = ruido[ind: ind + audio.size]
  
  mix = audio + ruido
  
  return mix,ruido

#-------------------------------------------------------------------------------
#PLOTA GRAFICO DO SINAL
def plota_sound_tempo(sinal, FS):
  time = np.arange(0, len(sinal) * 1/FS, 1/FS)
  plt.title('Sinal')
  plt.ylabel('Amplitude')
  plt.xlabel('Tempo (s)')
  plt.plot(time,sinal)
  plt.grid()
  plt.show()

#-------------------------------------------------------------------------------
#PLOTA O FFT DO SINAL
def plota_fft_sound(sinal, fs, pontos):
  time = np.arange(0, len(sinal) * (1/fs), 1/fs)
  fft = np.fft.fft(sinal)
  T = time[1] - time[0]
  N = sinal.size
  f = np.fft.fftfreq(len(sinal), T)
  freq = f[:N // 2]
  ampli= np.abs(fft)[:N // 2] * 1 / N
  plt.title('FFT do Sinal')
  plt.ylabel('Amplitude')
  plt.xlabel('Frequência (Hz)')
  if pontos == None:
    plt.plot(freq, ampli)
    plt.grid()
    plt.show()
  else:
    plt.plot(freq[:pontos], ampli[:pontos])
    plt.grid()
    plt.show()

#-------------------------------------------------------------------------------
#PLOTA O ESPECTOGRAMA DO SINAL 
def plota_spec(s):
  N = 1
  pontosFFT = 256 *N
  windowLen = 256*N
  Overlap = 64*N
  window = sci.signal.windows.hamming(256*N, sym=False)

  plt.title('Espectograma do Sinal')
  plt.ylabel('Frequência (Hz)')
  plt.xlabel('Tempo (s)')
  plt.specgram(s, NFFT = pontosFFT, Fs = 8000, window = window, noverlap = 0)
  #plt.specgram(s, NFFT = 256,Fs = 8000, noverlap = 0)
  plt.show()


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
#CALCULA E PLOTA O MEL-ESPECTOGRAMA DO SINAL
def specmel_signal(s):
  pontosFFT = 256
  windowLen = 256
  Overlap = 64
  window = sci.signal.windows.hamming(256, sym=False)

  specMel = librosa.feature.melspectrogram(y = s,sr=8000, S=s, n_fft=pontosFFT, hop_length=Overlap, win_length=windowLen, 
  window=window, pad_mode='reflect',center=True, power=2.0)

  fig, ax = plt.subplots()
  img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(specMel),ref=np.max), y_axis='log', x_axis='time', ax=ax, 
                  sr = 8000,hop_length = Overlap)
  ax.set_title('Power spectrogram')
  fig.colorbar(img, ax=ax, format="%+2.0f dB")
  plt.show()
  return specMel



#-------------------------------------------------------------------------------
#MODIFICA A FS DO SINAL PARA A FS DESEJADA(fs_output)
def resample(sinal, fs_input, fs_output):
  audioResample = []
  delay = 0.01
  n = 1
  if type(sinal) == list:    
    for i in range(len(sinal)):      
      audio_new = librosa.resample(sinal[i], orig_sr = fs_input, target_sr = fs_output)
      print('Arquivo nº',n,' resemplado')
      audioResample.append(audio_new)
      n = n + 1
  if type(sinal) == np.ndarray:
    audioResample = librosa.resample(sinal, orig_sr = fs_input, target_sr = fs_output)
    
    
  return audioResample

#-------------------------------------------------------------------------------
#GERA RUIDO GAUSSIANO COM A FUNÇÃO RANDN
def gera_ruido(amplitude, tamanho):
  return amplitude * np.random.randn(1,tamanho)

#-------------------------------------------------------------------------------
#Le Arquvios dos Audios Limpos
def Get_Audio():
  #baixa_arq('audios')
  nome_audios = []
  nome_audios = os.listdir("/home/rodrigo.fleith/downloads/DataSet_Audio")
  dataset_voice = []
  new_string = "/home/rodrigo.fleith/downloads/DataSet_Audio/"
  n = 1
  for i in range(len(nome_audios)):
    try:
      signal, samplerate = sf.read("{}{}".format(new_string, nome_audios[i]))
      a = resample(signal, 16000, 8000) 
      dataset_voice.append(a)
      print('Arquivo numero: ',i)
      
    except:
      #print("Audio Corrompido {}: {}".format(n, nome_audios[i]))
      print("Nº Audios Errado: ", n)
      n = n + 1
      pass

  return dataset_voice

#-------------------------------------------------------------------------------
#Le Arquivos de Ruidos
def Get_Noise():
  #baixa_arq('ruidos')
  nome_ruidos = []
  nome_ruidos = os.listdir("/home/rodrigo.fleith/downloads/DataSet_Ruido")
  dataset_noise = []
  new_string = "/home/rodrigo.fleith/downloads/DataSet_Ruido/"

  n = 1

  for i in range(len(nome_ruidos)):
    try:
      signal, FS = librosa.load(path = "{}{}".format(new_string, nome_ruidos[i]),sr = None, mono = True)
      a = resample(signal , FS, 8000)
      dataset_noise.append(a)
    except:
      #print("Audio Corrompido {}: {}".format(n, nome_ruidos[i]))
      print(n)
      n = n + 1
      pass
  return dataset_noise

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
      RSR =  20*np.log10(specV[j,i] / specN[j, i])  
      if RSR >= FC:
        Mask[j][i] = 1
      if RSR < FC:
        Mask[j][i] = 0
  return Mask
#-------------------------------------------------------------------------------

def Get_Features():
#Baxa os arquivos e downsample audios
  print("Audios")
  Voice = Get_Audio()
  print("Ruidos")
  Noise = Get_Noise()
#Mixa audios limpos com os segmentos equivalentes dos ruidos
  i = 0
  Mix = []
  NoiseSegment = []
  for i in range(4300):
    v_mix, ruidoSegment = mix_sound(Voice[i], Noise[i])
    print("Mixando Audio com Ruido: ", i)
    Mix.append(v_mix)
    NoiseSegment.append(ruidoSegment)
#Calcula o espectograma dos audio, ruido e do mix
  i = 0
  specVoice = []
  for i in range(4300):
    v = spec_signal(Voice[i])
    print("Espectograma Voice: ",i)
    specVoice.append(v)
  
  i = 0
  specNoise = []
  for i in range(4300):
    r = spec_signal(NoiseSegment[i])
    print("Espectograma Noise: ",i)
    specNoise.append(r)

  i = 0
  specMix = []
  for i in range(4300):
    m = spec_signal(Mix[i])
    print("Espectograma Mixado: ",i)
    specMix.append(m)

#SNR do Mix
  i = 0
  SNR_Mix = []
  for i in range(4300):
    RSR = snr_sinal(Voice[i], NoiseSegment[i])
    print("SNR Voice/Noise: ",i)
    SNR_Mix.append(RSR)

#Mascara Binaria Ideal

  for i in range(4300):
    specMask = Mask_Ideal_Bin(specVoice[i], specNoise[i], SNR_Mix[i])
    print("Mask ideal: ",i)
  

  return Voice, NoiseSegment, specVoice, specNoise, specMix, SNR_Mix, specMask



v, n, spV, spN, spM, snr, spMask = Get_Features()
