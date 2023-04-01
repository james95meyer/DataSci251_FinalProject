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

torchaudio.set_audio_backend('soundfile')

model = torch.jit.load('model_export.pt')
model.eval()

while True:
    time.sleep(10)
    files = os.listdir("Audio_Files")  
    with torch.no_grad():
        for file_name in files:
        
            # Load an audiofile 
            data, sr = torchaudio.load("Audio_Files/" + file_name)

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
                os.remove("Audio_Files/" + file_name)
                print("File deleted")

