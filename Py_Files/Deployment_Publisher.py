import os
import random
import math
import shutil
import time
import warnings
import soundfile
import paho.mqtt.client as mqtt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import torch.utils.data
import torchaudio
from torchaudio import transforms

LOCAL_MQTT_HOST="localhost"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="test_topic"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

torchaudio.set_audio_backend('soundfile')

model = torch.jit.load('model_export.pt')
model.eval()

while True:
    time.sleep(1)
    files = os.listdir("Audio_Files")  
    with torch.no_grad():
        for file_name in files:
        
            file_name = "Audio_Files/" + file_name
        
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
                f = open(file_name, "rb")
                audiostring = f.read()
                f.close()
                byteArray = bytearray(audiostring)
                local_mqttclient.publish(LOCAL_MQTT_TOPIC, byteArray)
                print("Message published")
                time.sleep(3)
