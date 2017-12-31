
# coding: utf-8

# In[1]:


import numpy as np
import wordsegment as ws
import pickle
import re
import io

# In[19]:


import time, sys


# In[20]:


def update_progress(progress):
	barLength = 100 # Modify this to change the length of the progress bar
	status = ""
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
		status = "error: progress var must be float\r\n"
	if progress < 0:
		progress = 0
		status = "Halt...\r\n"
	if progress >= 1:
		progress = 1
		status = "Done...\r\n"
	block = int(round(barLength*progress))
	text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
	sys.stdout.write(text)
	sys.stdout.flush()


# Loading data for processing

# In[11]:


fp_train = io.open("./datasets/train/SemEval2018-T3-train-taskA_emoji.txt",'r',encoding="utf-8")
ws.load()


# The code in below cell is function for obtaining text from a single tweet.

# In[12]:


def getDataFromTweet(sample):
	idx, label, tweet = sample.split('\t')
	hashtags = [i[1:] for i in tweet.split() if i.startswith("#")]
	hashtag_words = []
	text_words = [re.sub(r'[^\w\s]','',i).lower() for i in tweet.split() if not i.startswith("#")]
	for hashtag in hashtags:
		hashtag_words.extend(ws.segment(hashtag))
	return idx, label, tweet, text_words, hashtag_words


# This function reads all data from file and stores in lists

# In[13]:

def padData(data,pad_symbol):
	maxl = max([len(sent) for sent in data])
	data_padded = []

	for sent in data:
		sent_new = []
		length = len(sent)
		for i in range(length):
			sent_new.append(sent[i])
		for i in range(length,maxl):
			sent_new.append(pad_symbol)
		data_padded.append(sent_new)

	return data_padded, maxl


def readData(fp):
	samples = fp.read().strip().split('\n')
	samples = samples[1:]
	idxs = []
	tweets = []
	tweet_text = []
	tweet_hashtags = []
	labels = []

	for sample in samples:
		idx, label, tweet, text_words, hashtag_words = getDataFromTweet(sample)
		idxs.append(idx)
		labels.append(label)
		tweets.append(tweet)
		tweet_text.append(text_words)
		tweet_hashtags.append(hashtag_words)

	tweet_text, text_len = padData(tweet_text,'<pad>')
	tweet_hashtags, hashtags_len = padData(tweet_hashtags,'<pad>')

	return idxs, tweets, tweet_text, text_len, tweet_hashtags, hashtags_len, labels


# In[14]:


idxs, tweets, tweet_text, text_len, tweet_hashtags, hashtags_len, labels = readData(fp_train)


# In[15]:


print(len(tweets))
print(text_len, hashtags_len)

# In[16]:


def makeWordList(sent_list):
	wf = {}
	for sent in sent_list:
		for w in sent:
			if w in wf:
				wf[w] += 1
			else:
				wf[w] = 0
	wl = {}
	rwl = {}
	i = 0
	for w,f in wf.items():
		wl[w] = i
		rwl[i] = w
		i += 1
	wl['UNK'] = i
	return wl,rwl


# In[17]:


word_list, rev_word_list = makeWordList(tweet_text+tweet_hashtags)


# In[18]:


print(len(word_list))


# In[8]:


def mapWordToId(sent_contents,word_dict):
	T = []
	for sent in sent_contents:
		t = []
		for w in sent:
			if w in word_dict:
				t.append(word_dict[w])
			else:
				t.append(word_dict['UNK'])
		T.append(t)
	return T


# In[9]:


tweet_text_id = mapWordToId(tweet_text,word_list)
tweet_hashtags_id = mapWordToId(tweet_hashtags,word_list)


# In[10]:


# print (tweet_text_id[0])
# print (tweet_hashtags_id[0])


# In[23]:


def readWordEmb(word_dict, fname, embSize=100):
	print ("Reading word vectors from file")
	wv = []
	wl = []
	num_lines = sum(1 for line in io.open(fname, encoding="utf-8"))
	with io.open(fname, 'r',encoding="utf-8") as f:
		i = 1
		for line in f :
			vs = line.split()
			if len(vs) < 50 :
				continue
			vect = map(float, vs[1:])
			wv.append(vect)
			wl.append(vs[0])
			if (i%1000==0):
				print (i,"of",num_lines)
			i += 1
	wordemb = []
	count = 0
	print ("Reading words from word list")
	i = 1
	for word, id in word_dict.items():
		if str(word) in wl:
			wordemb.append(wv[wl.index(str(word))])
		else:
			count += 1
			wordemb.append(np.random.rand(embSize))
		if (i%100==0):
			print (i,"of",len(word_dict))
		i += 1
	wordemb = np.asarray(wordemb, dtype='float32')
	print ("Number of unknown word in word embedding", count)
	return wordemb


# In[ ]:


emb_file = "./glove/glove.twitter.27B.100d.txt"
wv = readWordEmb(word_list,emb_file)


# Save all data

# In[ ]:


W_text = np.array(tweet_text_id)
W_hashtags = np.array(tweet_hashtags_id)
Y = np.zeros([len(labels), 2],dtype=np.int)
for i in range(len(labels)):
	Y[i][labels[i]] = 1

with io.open('./datasets/taskA-train.pickle', 'wb') as handle:
	pickle.dump(text_len, handle)
	pickle.dump(hashtags_len, handle)
	pickle.dump(word_list, handle)

np.savez('./datasets/taskA-train',W_text,W_hashtags,Y,wv)