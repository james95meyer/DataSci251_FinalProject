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
import paho.mqtt.client as mqtt
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import torch.utils.data
import lmdb
import torchaudio
from torchaudio import transforms
from preprocessing import extract_spectrogram

torchaudio.set_audio_backend('soundfile')

#-----------------
#MQTT Setup
#-----------------

LOCAL_MQTT_HOST="localhost"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="audio_file"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

#-----------------
#Running inference
#-----------------

#Load the model
model = torch.jit.load('pt_Files/myModel_export.pt')
model.eval()

#Generate a list of audio files and run inference on them
while True:
    time.sleep(5)
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
                f = open(file_name, "rb")
                contents = f.read()
                msg = bytearray(contents)
                local_mqttclient.publish(LOCAL_MQTT_TOPIC,msg)
                print("Message sent")
                time.sleep(1)
                os.remove(file_name)
            else:
                os.remove(file_name)


