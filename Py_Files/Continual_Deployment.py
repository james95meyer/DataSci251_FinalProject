#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import random
import math
import shutil
import time
import warnings
import soundfile

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import torch.utils.data
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

torchaudio.set_audio_backend('soundfile')


# In[6]:


model = torch.jit.load('model_export.pt')
model.eval()


# In[7]:


while True:
    time.sleep(10)
    files = os.listdir()
    audio_files = []
    for file in files:
        if '.wav' in file:
            audio_files.append(file)
    
    with torch.no_grad():
        for file_name in audio_files:
        
            # Load an audiofile 
            data, sr = torchaudio.load(file_name)

            # Resample to two channels
            data = torch.cat([data, data])

            # Generate a Spectogram from the audio file
            data = transforms.MelSpectrogram(sr, n_fft=1024, hop_length=None, n_mels=64)(data)
            data = transforms.AmplitudeToDB(top_db=80)(data)
    
            # Normalize the values
            data_m, data_s = data.mean(), data.std()
            data = (data - data_m) / data_s
    
            # Add additional dimension to tensor in order to feed into the model
            data = data[None]
    
            # Receive a prediction from the model
            outputs = model(data)
            _, prediction = torch.max(outputs,1)
            prediction = int(prediction)
            if prediction != 10:
                os.remove(file_name)
                print("File deleted")

