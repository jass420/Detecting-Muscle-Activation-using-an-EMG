# Backyard Brains Sep. 2019
# Made for python 3
# First install serial library
# Install numpy, pyserial, matplotlib
# pip3 install pyserial
#
# Code will read, parse and display data from BackyardBrains' serial devices
#
# Written by Jasnoor Matharu
# mjas0401@gmail.com

import threading
import serial
import time
import matplotlib.pyplot as plt 
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import tensorflow as tf
import os
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification


'''# Keep sf of the file
y, sr = librosa.load('BYB_Recording_2024-02-03_17.11.20.wav', sr=None)
# Automatically resample to a desired fs
y, sr = librosa.load('BYB_Recording_2024-02-03_17.11.20.wav', sr=44100)
# Load the Nutcracker example
#filename = librosa.example('nutcracker')
#y, sr = librosa.load(filename, sr=None) 

librosa.display.waveshow(y, sr)
plt.show()'''
#bicep_flex = 0, twist_arm = 1
directory = 'C:/Users/mjas0/OneDrive/Desktop/moscle'
audios = []
y_train = []
x_train = []
y_test = []
x_test = []
def process_audio(wav_files, c, typ):
    # Loop through each WAV file
    for filename in wav_files:
        if os.path.isfile(filename):
            try:
                data, fs = sf.read(filename, dtype='float32')
                #print("max length",max(0,len(data)))
                padded_data = np.pad(data, (0, 569552 - len(data)), mode='constant')
                #print(f"File '{filename}' read successfully")
                if typ == 'test':
                    x_test.append(padded_data)
                    y_test.append(c)
                elif typ == 'train':
                    x_train.append(padded_data)
                    y_train.append(c)
                
                print(data, fs)
            except Exception as e:
                print(f"Error reading file '{filename}': {e}")
        else:
            print(f"File '{filename}' does not exist")
    return x_train 
    #print(type(data))
    #print(data.shape, data.dtype, fs)

# List all files in the directory and get their full paths
full_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
print(full_paths)

if 'C:/Users/mjas0/OneDrive/Desktop/moscle\\bicep_flex' in full_paths:
    audio_paths = [os.path.join('C:/Users/mjas0/OneDrive/Desktop/moscle\\bicep_flex', f) for f in os.listdir('C:/Users/mjas0/OneDrive/Desktop/moscle\\bicep_flex')]
    wav_files = [f for f in audio_paths if f.endswith('.wav')]
    process_audio(wav_files, c=0, typ='train')

if 'C:/Users/mjas0/OneDrive/Desktop/moscle\\twist_arm' in full_paths:
    twist_audio_paths = [os.path.join('C:/Users/mjas0/OneDrive/Desktop/moscle\\twist_arm', f) for f in os.listdir('C:/Users/mjas0/OneDrive/Desktop/moscle\\twist_arm')]
    twist_wav_files = [f for f in twist_audio_paths if f.endswith('.wav')]
    process_audio(twist_wav_files, c=1, typ='train')

if 'C:/Users/mjas0/OneDrive/Desktop/moscle\\test_flex' in full_paths:
    bicep_test_audio_paths = [os.path.join('C:/Users/mjas0/OneDrive/Desktop/moscle\\test_flex', f) for f in os.listdir('C:/Users/mjas0/OneDrive/Desktop/moscle\\test_flex')]
    bicep_test_wav_files = [f for f in bicep_test_audio_paths if f.endswith('.wav')]
    process_audio(bicep_test_wav_files, c=0, typ='test')

if 'C:/Users/mjas0/OneDrive/Desktop/moscle\\test_arm_twist' in full_paths:
    twist_test_audio_paths = [os.path.join('C:/Users/mjas0/OneDrive/Desktop/moscle\\test_arm_twist', f) for f in os.listdir('C:/Users/mjas0/OneDrive/Desktop/moscle\\test_arm_twist')]
    twist_test_wav_files = [f for f in twist_test_audio_paths if f.endswith('.wav')]
    process_audio(twist_test_wav_files, c=1, typ='test')
#The code below initiates an MLP classifier with optimal parameters and subsequently applies bagging using bootstrapped data. 
#The ensemble of the mlp classifiers is then used to make predictions    
#1)Creating a mlp classifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(350,200,100,50), activation='logistic', solver='adam', tol=0.01, verbose=True, learning_rate='adaptive', warm_start=True, early_stopping=True)
#2) Bagging
bag_clf = BaggingClassifier(estimator=mlp_clf, bootstrap=True)
#3) Fitting the model to our data
bag_clf.fit(x_train, y_train)
train_pred = bag_clf.predict(x_test)
#Accuracy is calculated
accuracy = accuracy_score(y_test, train_pred)

print(f"Accuracy: {np.around(accuracy*100, 2)}%")
plt.plot(mlp_clf.loss_curve_)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
exit()
log_regr = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, multi_class='auto')
log_regr.fit(x_train, y_train)

#TODO: Make predictions on the test set
y_predict = log_regr.predict(x_test)

threshold = 0.75 
y_predict_binary = (log_regr.predict_proba(x_test)[:, 1] > threshold).astype(int)
conf2_matrix = confusion_matrix(y_true=y_test, y_pred=y_predict_binary)
print(conf2_matrix)
fig, ax = plt.subplots()
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf2_matrix, display_labels=log_regr.classes_)
cm_display.plot(ax=ax, cmap='Greens')
plt.xticks(ticks=[0, 1], labels=["Bicep flex", "Arm twist"])  # Assuming your classes are named Class 0 and Class 1
plt.yticks(ticks=[0, 1], labels=["Bicep flex", "Arm twist"])

plt.show()

accuracy = accuracy_score(y_pred=y_predict_binary, y_true=y_test)
print(f"Accuracy: {np.around(accuracy*100, 2)}%")

 
exit()
mlp_clf.fit(x_train, y_train)
#y_pred = mlp_clf.predict(test_x) 
train_pred = mlp_clf.predict(x_train)
accuracy = accuracy_score(x_train, train_pred)  # Assuming y_test contains the true labels for X_test 
print("Test set accuracy:", accuracy)
 

