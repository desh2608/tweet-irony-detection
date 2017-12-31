
# coding: utf-8

# In[74]:


import numpy as np
import io
import operator


# In[75]:


fp_train = io.open("./datasets/train/SemEval2018-T3-train-taskA.txt",'r',encoding="utf-8")


# In[76]:


def getEmojisFromTweet(sample):
    emoji = []
    idx, label, tweet = sample.split('\t')
    for i in tweet.split():
        if i.startswith(":"):
            emoji.extend(list(filter(None, i.split(":"))))
    emoji = list(set(emoji))
    emoji = [i for i in emoji if i[0].isalpha()]
    return emoji, label


# In[77]:


def readData(fp):
    samples = fp.read().strip().split('\n')
    samples = samples[1:]
    emojis = []
    labels = []

    for sample in samples:
        emoji, label = getEmojisFromTweet(sample)
        labels.append(label)
        emojis.append(emoji)

    return emojis, labels


# In[78]:


emojis, labels = readData(fp_train)
print (len(labels))


# In[79]:


def makeEmojiList(emojis):
    ef = {}
    for sent in emojis:
        for e in sent:
            if e in ef:
                ef[e] += 1
            else:
                ef[e] = 0
    el = {}
    i = 0
    for e,f in ef.items():
        el[e] = i
        i += 1
    el['UNK'] = i
    return el


# In[80]:


emoji_list = makeEmojiList(emojis)


# In[87]:


freq_irony = {}
freq_non = {}
freq_norm = {}
for emoji,label in zip(emojis,labels):
    label = int(label)
    if len(emoji) != 0:
        for x in emoji:
            if label == 1:
                if x in freq_irony:
                    freq_irony[x] += 1
                    freq_norm[x] += 1
                else:
                    freq_irony[x] = 1
                    freq_norm[x] = 1
            else:
                if x in freq_non:
                    freq_non[x] += 1
                    freq_norm[x] -= 1
                else:
                    freq_non[x] = 1
                    freq_norm[x] = -1
freq_irony = sorted(freq_irony.items(), key=operator.itemgetter(1), reverse=True)
freq_non = sorted(freq_non.items(), key=operator.itemgetter(1), reverse=True)
freq_norm = sorted(freq_norm.items(), key=operator.itemgetter(1), reverse=True)


# In[90]:


emoji_imp = [i[0] for i in freq_norm[:5]]
emoji_imp.extend([i[0] for i in freq_norm[-5:]])


# In[94]:


W_emoji = np.zeros((len(emojis),len(emoji_imp)),dtype=np.int)
i = 0
for emoji in emojis:
    for x in emoji:
        if x in emoji_imp:
            W_emoji[i][emoji_imp.index(x)] += 1
    i += 1


# In[101]:


np.save('./datasets/taskA-emoji',W_emoji)

