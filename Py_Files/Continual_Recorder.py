#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyaudio
import wave
import datetime
import random


# In[2]:


chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = 5


# In[4]:


p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

# Store data in chunks for 5 seconds
while True:
    frames = [] # Initialize array to store frames
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print(len(frames))
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    filename = "audio_" + timestamp + ".wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Recording saved")

# Stop and close the stream 
#stream.stop_stream()
#stream.close()
# Terminate the PortAudio interface
#p.terminate()

#print('Finished recording')

# Save the recorded data as a WAV file

