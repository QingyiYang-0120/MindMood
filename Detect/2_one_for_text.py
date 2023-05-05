"""
被试者文本特征提取
"""
import datetime
start=datetime.datetime.now()
import warnings

from tqdm import tqdm
warnings.filterwarnings("ignore")

from transformers import logging

logging.set_verbosity_warning()

import torch
from transformers import BertTokenizer
from transformers import BertModel

topics = ['positive', 'neutral', 'negative']
answers = {}
text_features = []
text_targets = []
ss = []
import numpy as np
import os

import sys
user = sys.argv[1]
print(user)

def extract_features(text_features, user):
    fold_num = sum([os.path.isdir("EATD-Corpus/" + user + "/" + listx) for listx in os.listdir("EATD-Corpus/" + user)])
    print(fold_num)

    for index in tqdm(range(fold_num, fold_num + 1)):
        if os.path.isdir("EATD-Corpus/" + user + "/" + str(index)):
            answers[index] = []
            for topic in topics:
                with open("EATD-Corpus/" + user + "/" + str(index) + "/" + '%s.txt' % (topic), 'r', encoding="utf-8") as f:

                    lines = f.readlines()[0]
                    marked_text = "[CLS] " + lines + " [SEP]"
                    print(marked_text)

                    tokenized_text = tokenizer.tokenize(marked_text)

                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                    segments_ids = [1] * len(tokenized_text)
                    tokens_tensor = torch.tensor([indexed_tokens])
                    segments_tensors = torch.tensor([segments_ids])

                    model.eval()
                    with torch.no_grad():
                        outputs = model(tokens_tensor, segments_tensors)
                        hidden_states = outputs[2]
                    token_vecs = hidden_states[-2][0]
                    sentence_embedding = torch.mean(token_vecs, dim=0)
                    answers[index].append(sentence_embedding)

            temp = []
            for i in range(3):
                temp.append(np.array(answers[index][i]))
            text_features.append(temp)
        else:
            print("error!")

# print("------current:")
# print(os.getcwd())

tokenizer = BertTokenizer.from_pretrained('Detect/bert')
model = BertModel.from_pretrained('Detect/bert', output_hidden_states=True)

extract_features(text_features, user)

print("Saving npz file locally...")
# predict_text.npz存放被试者提取的三类文本向量
savepath = os.path.join(os.path.join("EATD-Corpus", user), 'predict_text.npz')
np.savez(savepath, text_features)
print("text_features:")
print(len(text_features))

end=datetime.datetime.now()
print('Running time: %s Seconds' %(end-start))