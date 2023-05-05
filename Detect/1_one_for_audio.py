# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import wave
import librosa
import sys
sys.path.append('/')

import tensorflow._api.v2.compat.v1 as tf
from vggish import loupe_keras as lpk

user = sys.argv[1]
print(user)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

cluster_size = 16
min_len = 100
max_len = -1

def wav2vlad(wave_data, sr):
    global cluster_size
    signal = wave_data
    melspec = librosa.feature.melspectrogram(signal, n_mels=80,sr=sr).astype(np.float32).T
    melspec = np.log(np.maximum(1e-6, melspec))
    feature_size = melspec.shape[1]
    max_samples = melspec.shape[0]
    output_dim = cluster_size * 16
    feat = lpk.NetVLAD(feature_size=feature_size, max_samples=max_samples, cluster_size=cluster_size, output_dim=output_dim)(tf.convert_to_tensor(melspec))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        r = feat.numpy()
    return r

def extract_features(number, audio_features, name):
    global max_len, min_len

    if not os.path.exists("EATD-Corpus/" + name + "/"+ str(number)):
        print("error!!!")
        return
    else:
        print("not error!!!")

    pre = "EATD-Corpus/" + name + "/" + str(number)
    pre = os.path.join(os.getcwd(), pre)
    pos = os.path.join(pre, "positive.wav")
    positive_file = wave.open(pos)
    print("success open positive.wav!")
    sr1 = positive_file.getframerate()
    nframes1 = positive_file.getnframes()
    wave_data1 = np.frombuffer(positive_file.readframes(nframes1), dtype=np.short).astype(np.float)
    print(wave_data1)
    len1 = nframes1 / sr1

    neutral_file = wave.open( pre + "/neutral.wav")
    sr2 = neutral_file.getframerate()
    nframes2 = neutral_file.getnframes()
    wave_data2 = np.frombuffer(neutral_file.readframes(nframes2), dtype=np.short).astype(np.float)
    len2 = nframes2 / sr2

    negative_file = wave.open( pre + "/negative.wav")
    sr3 = negative_file.getframerate()
    nframes3 = negative_file.getnframes()
    wave_data3 = np.frombuffer(negative_file.readframes(nframes3), dtype=np.short).astype(np.float)
    len3 = nframes3 / sr3

    for l in [len1, len2, len3]:
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l

    if wave_data1.shape[0] < 1:
        wave_data1 = np.array([1e-4]*sr1*5)
    if wave_data2.shape[0] < 1:
        wave_data2 = np.array([1e-4]*sr2*5)
    if wave_data3.shape[0] < 1:
        wave_data3 = np.array([1e-4]*sr3*5)
    audio_features.append([wav2vlad(wave_data1, sr1), wav2vlad(wave_data2, sr2),wav2vlad(wave_data3, sr3)])


audio_features = []

# print(len(os.listdir("EATD-Corpus/"+user)))
# print(os.listdir("EATD-Corpus/"+user))

fold_num = sum([os.path.isdir("/Users/yangqingyi/Desktop/JS-AI/心灵侦探/EATD-Corpus/"+user+"/"+listx) for listx in os.listdir("/Users/yangqingyi/Desktop/JS-AI/心灵侦探/EATD-Corpus/"+user)])
print(fold_num)

for index in range(fold_num,fold_num+1):
    extract_features(index, audio_features, user)

print("Saving npz file locally...")

savepath = os.path.join(os.path.join("EATD-Corpus", user), 'feature.npz')
np.savez(savepath, audio_features)

print(max_len, min_len)
