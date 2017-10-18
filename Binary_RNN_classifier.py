
# coding: utf-8

# In[75]:

import wave
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import sklearn as sk
import scipy.io
import librosa
import librosa.display
get_ipython().magic('matplotlib inline')


# In[99]:

def load_sound_files(file_paths):
    raw_sounds = []
    
    
    items = os.listdir(file_paths)

    
    #searches through the input file for any files 
    #named .wav and adds them to the list
    
    newlist = []
    for names in items:
        if names.endswith(".wav"):
            newlist.append(names)
   
    #Loads the files found above in with librosa
    for fp in newlist:
        fp = os.path.join(path, fp)
        X,sr = librosa.load(fp,500)  
        raw_sounds.append(X)
    return raw_sounds


# In[100]:

path = '/home/tim/Documents/Masters/Data/93-001-2321.ch13/'
raw_sounds =  load_sound_files(path)


# In[ ]:

"""
For testing purposes at the moment, code is here to allow you to view the input data.


"""
def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=500)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    plt.show()
    
def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        f_ = librosa.stft(y = f, n_fft= 256, win_length= 256)
        librosa.display.specshow(librosa.amplitude_to_db(f_),
                                        sr = 500,
                                        y_axis = 'log',
                                        hop_length = 64 )
        #specgram(np.array(f), Fs=500, mode = 'psd')
        plt.title(n.title())
        #plt.colorbar(format='%+4.0f dB')
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 1200)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log', sr = 500)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()
 
plot_waves('minke',raw_sounds)
plot_specgram('minke',raw_sounds)
plot_log_power_specgram('minke',raw_sounds)


# In[157]:

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty(0), np.empty(0)
    print(parent_dir,sub_dirs)

   
    for label, sub_dir in enumerate(sub_dirs):
                
        items = os.listdir(os.path.join(parent_dir, sub_dir))

    
        #searches through the input file for any files 
        #named .wav and adds them to the list
    
        files = []
        for names in items:
            if names.endswith(".wav"):
                files.append(names)
        for fn in files:
            print(os.path.join(parent_dir, sub_dir, file_ext))
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = sub_dir
    return np.array(features), np.array(labels, dtype = np.int)



# In[158]:


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


# In[159]:

parent_dir = '/home/tim/Documents/Masters/Data'
tr_sub_dirs = ['Minke','Noise']
ts_sub_dirs = ['Test']
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)

print(tr_features)


# In[ ]:



