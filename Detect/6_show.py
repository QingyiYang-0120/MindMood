# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import joblib
warnings.filterwarnings("ignore")
import numpy as np

import jieba
import itertools
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
name= sys.argv[1]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


data = pd.read_csv("EATD-Corpus/" + str(name) + "/" + "record.csv")
with open('Detect/stopwordlist.txt', 'r', encoding='utf-8') as f:  # 停用词的文本
    stop = f.read()
stop = stop.split()
stop = [' ', '\n', '这部'] + stop

fig = plt.figure(figsize=(15, 12), dpi=700)
ax = fig.add_subplot(221)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.plot(np.arange(0, data.shape[0])+1, data["SDS分数上限"], linestyle="--", label="SDS分数上限")
ax.plot(np.arange(0, data.shape[0])+1, data["SDS分数下限"], linestyle="--", label="SDS分数下限")
ax.plot(np.arange(0, data.shape[0])+1, data["预测SDS分数"], label="预测SDS分数")
ax.scatter(np.arange(0, data.shape[0])+1, data["SDS分数上限"])
ax.scatter(np.arange(0, data.shape[0])+1, data["SDS分数下限"])
ax.scatter(np.arange(0, data.shape[0])+1, data["预测SDS分数"], marker="*")
# plt.grid()
plt.legend()
ax.set_xlabel("记录")
ax.set_ylabel("分数")
ax.set_title(name + "SDS分数变化图", fontsize=20)

train_features = ["Detect/fuse_features.npy"]
train_labels = ["Detect/fuse_labels.npy"]
# from sklearn import manifold
# tsne = manifold.TSNE()
tsne = joblib.load("Detect/Model/tsne.dat")

X_train = np.load(train_features[0])
y = np.load(train_labels[0])

x_test = data[["sds_x", "sds_y"]].values
X_tsne = tsne.fit_transform(X_train)  # 对训练X进行降维
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
X0 = X_norm[y == 0]
X1 = X_norm[y == 1]

ax1 = fig.add_subplot(222)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.title("被试者情绪倾向二维可视化图", fontsize=20)
plt.scatter(X0[:, 0], X0[:, 1], label="历史正常", marker="o", color="green")
plt.scatter(X1[:, 0], X1[:, 1], label="历史抑郁", marker="x", color="red")
plt.scatter(x_test[:, 0], x_test[:, 1], label=name, marker="d", color="blue")
for i in range(x_test.shape[0]):
    if (data["结果"][i] == 1):
        plt.text(x_test[i, 0], x_test[i, 1], str(i + 1) + "X")
    else:
        plt.text(x_test[i, 0], x_test[i, 1], str(i + 1))
plt.grid()
plt.legend(bbox_to_anchor=(0.5, -0.1), loc=8, ncol=10)
plt.xticks([])
plt.yticks([])

ax2 = fig.add_subplot(223)
x0 = np.mean(data["正常概率"].values)
x1 = np.mean(data["抑郁概率"].values)
piex = np.array([x0, x1])
patches, l_text, p_text = plt.pie(piex, labels=['正常', '抑郁'], colors=["lightgreen", "mediumorchid"],
                                  explode=(0, 0.2), autopct='%.2f%%')
for t in p_text:
    t.set_size(20)
for t in l_text:
    t.set_size(12)
plt.title(name + "情绪概率可视化展示", fontsize=20)  # 设置标题
plt.grid()

# In[]
ax3 = fig.add_subplot(224)
sens = ""
for i in data["个人摘要"]:
    sens += i[1:-1]
lines = pd.Series(sens)
data_pos = lines.apply(jieba.lcut)
data_pos = data_pos.apply(lambda x: [i for i in x if i not in stop and len(i) > 1])
data_pos.index = [i for i in range(data_pos.shape[0])]
num = pd.Series(list(itertools.chain(*list(data_pos)))).value_counts()  # 统计词频
# scale=32这个属性被我注释掉了
wc = WordCloud(font_path='Detect/Simhei.ttf', background_color='White', height=350, colormap="RdPu")
wc2 = wc.fit_words(num)
plt.imshow(wc2)
plt.axis('off')
plt.title(name + "用户文本关键词记录", fontsize=20)
#
# filename = name+".png"
# pre = os.path.join(os.path.join("/static/detailedresult",filename))
# print(pre)

plt.savefig('static/detailedresult.png', transparent=True, bbox_inches='tight', pad_inches=0.0)

print("success!!")