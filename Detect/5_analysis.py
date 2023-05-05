from sklearn import manifold, datasets
import numpy as np
import joblib
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")
import sys
user = sys.argv[1]

"""
这里的train_features表示我们的历史训练集，也就是历史上服务过的用户（所谓的EATD数据集）,与train_labels对应,用new function压缩包中的
fuse_features和fuse_labels.npy
"""
train_features=["Detect/fuse_features.npy"]
train_labels=["Detect/fuse_labels.npy"]

"""
这里的x_test表示现在进行抑郁识别的新用户。和抑郁评分中的test_features一致,为方便起见暂时取前n个被试者为新用户
"""
X_train = np.load(train_features[0])
y = np.load(train_labels[0])
import os
fuse_path = os.path.join(os.path.join("EATD-Corpus", user), 'fuse_features.npy')
x_test = np.load(fuse_path)

# In[]
tsne = manifold.TSNE() 
tsne = joblib.load("Detect/Model/tsne.dat")
X_train=np.concatenate([X_train,x_test])
y=np.concatenate([y,np.ones(x_test.shape[0])*2])

X_tsne = tsne.fit_transform(X_train) # 对训练X进行降维
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

x_test = X_norm[-x_test.shape[0]:,:]
X0=X_norm[y==0]
X1=X_norm[y==1]

import pandas as pd
data=pd.read_csv("EATD-Corpus/"+str(user)+"/"+"record.csv")
if data.shape[0]==1:
    data["sds_x"]=x_test[:,0]
    data["sds_y"]=x_test[:,1]
    data.to_csv("EATD-Corpus/"+str(user)+"/"+"record.csv",index=False)
else:
    flist = list(data["sds_x"].dropna().values) + list(x_test[:, 0])
    plist = list(data["sds_y"].dropna().values) + list(x_test[:, 1])
    data["sds_x"] = flist
    data["sds_y"] = plist
    data.to_csv("EATD-Corpus/" + str(user) + "/" + "record.csv", index=False)

# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import os
import re
import jieba
import itertools
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import pandas as pd
import jiagu
import sys
user = sys.argv[1]
topics = ['positive', 'neutral', 'negative']
def find_chinese(file):
    pattern = re.compile(r'[^\u4E00-\u9FD5]')
    chinese = re.sub(pattern, '', file)
    return chinese

with open('Detect/stopwordlist.txt', 'r', encoding='utf-8') as f:#停用词的文本
    stop = f.read()
stop = stop.split()
stop = [' ', '\n', '这部'] + stop #停用词

# 统计词频
def word_show(data,topic,i):
    num = pd.Series(list(itertools.chain(*list(data)))).value_counts() # 统计词频
    wc = WordCloud(font_path='Detect/simhei.ttf', background_color='White',scale=32)
    wc2 = wc.fit_words(num)
    plt.figure()
    plt.imshow(wc2)
    plt.axis('off')
    plt.title("第"+str(i)+"次用户文本关键词记录")
    plt.show()
    return num

# In[]
data=pd.read_csv("EATD-Corpus/"+str(user)+"/"+"record.csv")
summarylist=[]
def extract_features():
    for index in range(len(os.listdir("EATD-Corpus/"+user))):#这里调整第几个到第几个被试者
        """
        这里是存放被试者的信息的目录，和EATD一样,path就是子目录的一个前缀
        """
        if os.path.isdir("EATD-Corpus/"+str(user)+"/"+str(index+1)):
            sentences=""
            for topic in topics:
                with open("EATD-Corpus/"+str(user)+"/"+str(index+1)+"/"+'%s.txt'%(topic) ,'r',encoding="utf-8") as f:
                    lines = f.readlines()[0]
                    sentences+=lines

            lines=pd.Series(sentences)
            data_pos = lines.apply(jieba.lcut)
            data_pos = data_pos.apply( lambda x: [i for i in x if i not in stop and len(i)>1] )
            data_pos.index = [ i for i in range(data_pos.shape[0]) ]
            summarize = jiagu.summarize(sentences, 1) # 摘要
            summarylist.append(summarize)
            print("个人摘要:"+summarize[0])

extract_features()
data=pd.read_csv("EATD-Corpus/"+str(user)+"/"+"record.csv")
data["个人摘要"]=summarylist
data.to_csv("EATD-Corpus/"+str(user)+"/"+"record.csv",index=False)