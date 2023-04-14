import os
import random
import math
import shutil
import time
import warnings
import soundfile
import csv
import torchvision

import librosa
import argparse
import numpy as np
import pickle as pkl 

from PIL import Image

#import torchvision.models as models
#import json
#import shutil
#from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import torch.utils.data
import lmdb
import torchaudio
from torchaudio import transforms

torchaudio.set_audio_backend('soundfile')

#Required transformation to each audio file to feed it into the model

def extract_spectrogram(audio_file):
    
    sampling_rate = 44100
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]
    centre_sec = 2.5

    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[i]*sampling_rate/1000))
        hop_length = int(round(hop_sizes[i]*sampling_rate/1000))

        audio_file = torch.Tensor(audio_file)
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_fft=4410, win_length=window_length, hop_length=hop_length, n_mels=128)(audio_file)
        eps = 1e-6
        spec = spec.numpy()
        spec = np.log(spec+ eps)
        spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
        specs.append(spec)
    return specs

#Create the predictions file
prediction_file = 'densenet_predictions.csv'
pred_list = []

#Load the model
model = torch.jit.load('pt_Files/myModel_export.pt')
model.eval()

#Generate a list of audio files and run inference on them
files = os.listdir("Audio_Files")  
with torch.no_grad():
    for file_name in files:
        file_name = "Audio_Files/" + file_name
        clip, sr = librosa.load(file_name, sr = 44100)
        clip = torch.Tensor(clip)
        data = extract_spectrogram(clip)
        data = torch.Tensor(np.array(data))
        #Add an additional dimension to feed it into the model
        data = data[None]
        outputs = model(data)
        _, prediction = torch.max(outputs,1)
        prediction = int(prediction)
        if prediction == 24:
            prediction = 1
        else:
            prediction = 0
        pred_list.append({'file':file_name, 'prediction':prediction})

#Write to the prediction file
with open(prediction_file, 'w') as f:
    writer = csv.DictWriter(f, fieldnames = ['file','prediction'])
    writer.writeheader()
    writer.writerows(pred_list)
