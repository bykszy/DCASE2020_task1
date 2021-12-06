import os
import numpy as np
import scipy.io
import pandas as pd
from audiotoolkit import *
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool


file_path = ''
csv_file = 'evaluation_setup/fold1_train.csv'
output_path = 'features/logmel64_scaled'
feature_type = 'logmel'

sr = 44100
duration = 10
num_freq_bin = 64
num_fft = 2048
hop_length = int(num_fft / 2)
num_time_bin = int(np.ceil(duration * sr / hop_length))
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()


for i in range(len(wavpath)):
        #stereo, fs = sound.read(file_path + wavpath[i], stop=duration * sr)
        print(wavpath[i])
        #print(stereo)
        #logmel_data3 = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
        xx = AudioClip(wavpath[i], sr).cut(10, 1).get_audio()[0,0].T.flatten()
       # xx1 = AudioClip(wavpath[i], sr).cut(10, 0, 0).add_echo(100).add_noise().get_audio()[0, 0].T.flatten()
        #xx2 = AudioClip(wavpath[i], sr).cut(10, 0, 0).reverse().add_noise().get_audio()[0, 0].T.flatten()
       # xx3 = AudioClip(wavpath[i], sr).cut(10, 0, 0).specaug(1,1,10,10).get_melspectograms()
        #print(xx[0,0])
       # logmel_data3[:, :, 0] = xx3[0][:, :, 0]

        logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
        #logmel_data1 = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
        #logmel_data2 = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
       # logmel_data3 = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')

        logmel_data[:,:,0]= librosa.feature.melspectrogram(xx[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)
       # logmel_data1[:,:,0]= librosa.feature.melspectrogram(xx1[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)
        #logmel_data2[:,:,0]= librosa.feature.melspectrogram(xx2[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)
       # logmel_data3[:,:,0]= librosa.feature.melspectrogram(xx[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)
        #logmel_data3[:, :, 0]=
        logmel_data = np.log(logmel_data+1e-8)
       # logmel_data1 = np.log(logmel_data1 + 1e-8)
       # logmel_data2 = np.log(logmel_data2 + 1e-8)
       # logmel_data3 = np.log(logmel_data3 + 1e-8)

        feat_data = logmel_data
     #   feat_data1 = logmel_data1
        #feat_data2 = logmel_data2
    #    feat_data3 = logmel_data3

        feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
     #   feat_data1 = (feat_data1 - np.min(feat_data1)) / (np.max(feat_data1) - np.min(feat_data1))
     #   #feat_data2 = (feat_data2 - np.min(feat_data2)) / (np.max(feat_data2) - np.min(feat_data2))
     #   feat_data3 = (feat_data3 - np.min(feat_data3)) / (np.max(feat_data3) - np.min(feat_data3))

        feature_data = {'feat_data': feat_data,}
     #   feature_data1 = {'feat_data': feat_data1, }
        #feature_data2 = {'feat_data': feat_data2, }
      #  feature_data3 = {'feat_data': feat_data3, }

        cur_file_name = output_path + wavpath[i][5:-3] + feature_type
       # cur_file_name1 = output_path + wavpath[i][5:-4]+'-echo.' + feature_type
        #cur_file_name2 = output_path + wavpath[i][5:-4]+'-reverse.' + feature_type
      #  cur_file_name3 = output_path + wavpath[i][5:-4]+'-specaug.' + feature_type

        pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
      #  pickle.dump(feature_data1, open(cur_file_name1, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        #pickle.dump(feature_data2, open(cur_file_name2, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
       # pickle.dump(feature_data3, open(cur_file_name3, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        

