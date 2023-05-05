from transformers import AutoModelForSequenceClassification , AutoTokenizer, pipeline
from cnsenti import Emotion
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import numpy as np
colors = sns.color_palette('bright')

emotion = Emotion()
class_num = 2

# model_name = "liam168/c2-roberta-base-finetuned-dianping-chinese"
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=class_num)
# model.save_pretrained('./Auto/')
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained('./Auto/')

tokenizer = AutoTokenizer.from_pretrained('Detect/Auto')
model = AutoModelForSequenceClassification.from_pretrained('Detect/Auto', num_labels=class_num)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# print("请输入您的感情状态文本:")
# msg = input(" ")
import sys
user = sys.argv[1]
msg = sys.argv[2]
# print(msg)

if classifier(msg)[0]['label']=="negative":
    print("您有点抑郁倾向！请注意!")
    result = emotion.emotion_count(msg)
    print("您的文段状态分析(单词,句子,情感词统计分布):",result)

    # plt.figure(dpi=600)
    # words=[] # 记录对应的情感类别
    # counts=[]  # 情感词的出现频率
    # for (i,y) in zip(result.keys(),result.values()):
    #     if i=="words" or i=="sentences":
    #         continue
    #     else:
    #         words.append(i)
    #         counts.append(y)
    # counts=np.abs(counts)+0.01
    # counts=counts/counts.sum()
    # c=tuple([0 for i in range(len(words))])
    # c=list(c)
    # c[np.argmax(counts)]=0.1
    # c=tuple(c)
    # plt.pie(counts, labels=words,colors = colors, explode=c, shadow=True, autopct = '%0.0f%%')
    # plt.title("情绪倾向分布")
    # plt.savefig("EATD-Corpus/"+str(user)+"/"+"quicktest.png", transparent=True, bbox_inches='tight', pad_inches=0.0)
    # plt.show()

else:
    print("正常!")
    result = emotion.emotion_count(msg)
    print("您的文段状态分析(单词,句子,情感词统计分布):", result)
