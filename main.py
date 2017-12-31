import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
import pickle
import model as ModelSource
import io

fp = open('./results/taskA.txt','w')

with io.open('./datasets/taskA-train.pickle', 'rb') as handle:
	text_len = pickle.load(handle)
	hashtags_len = pickle.load(handle)
	word_dict = pickle.load(handle)

with np.load('./datasets/taskA-train.npz') as data:
	W_text = data['arr_0']
	W_hashtags = data['arr_1']
	Y = data['arr_2']
	wv = data['arr_3']

W_emoji = np.load('./datasets/taskA-emoji.npy')

per = 0.9

num_total = len(W_text)
word_dict_size = len(word_dict)
label_dict_size = 2
emoji_dict_size = len(W_emoji[0])

print ("Total number of samples =",num_total)
print ("Dictionary size =",word_dict_size)

# Splitting training set into train and validation set
W_text_tr = W_text[:int(per*num_total)+1]
W_text_dev = W_text[int(per*num_total)+1:]
W_hashtags_tr = W_hashtags[:int(per*num_total)+1]
W_hashtags_dev = W_hashtags[int(per*num_total)+1:]
W_emoji_tr = W_emoji[:int(per*num_total)+1]
W_emoji_dev = W_emoji[int(per*num_total)+1:]
Y_tr = Y[:int(per*num_total)+1]
Y_dev = Y[int(per*num_total)+1:]


model = ModelSource.Model(label_dict_size,wv,text_len,hashtags_len,emoji_dict_size)

## Training the model
num_train = len(W_text_tr)
y_true_list = []
y_pred_list = []
num_epochs = 50
N = 5
batch_size = 256
num_batches_per_epoch = int(num_train/batch_size)


def test_step(W_text_te, W_hashtags_te, W_emoji_te, Y_te):
	n = len(W_text_te)
	num = int(n/batch_size) + 1
	sample = []
	for batch_num in range(num):
		start_index = batch_num*batch_size
		end_index = min((batch_num + 1) * batch_size, n)
		sample.append(range(start_index, end_index))
	pred = []
	for i in sample:
		p = model.test_step(W_text_te[i], W_hashtags_te[i], W_emoji_te[i], Y_te[i])
		pred.extend(p)
	return pred


for j in range(num_epochs):
	acc = []
	step = 0
	sam=[]
	for batch_num in range(num_batches_per_epoch):
		start_index = batch_num*batch_size
		end_index = (batch_num + 1) * batch_size
		sam.append(range(start_index, end_index))

	for rang in sam:
		step,acc_cur  = model.train_step(W_text_tr[rang], W_hashtags_tr[rang], W_emoji_tr[rang], Y_tr[rang])
		acc.append(acc_cur)

	acc = np.array(acc)
	print ("Average accuracy for epoch",j+1,"=",np.mean(acc))
	if ((j+1)%N==0):
		fp.write('Epoch: '+str(j+1)+'\n')
		pred = test_step(W_text_dev, W_hashtags_dev, W_emoji_dev, Y_dev)
		print ("test data size ", len(pred))
		y_true = np.argmax(Y_dev, 1)
		y_pred = pred
		fp.write(classification_report(y_true, y_pred, digits=4))
		print(classification_report(y_true, y_pred, digits=4))
print ("Training finished.")
fp.close()
##------------------------------------------------------------------------------------##
"""
##TESTING

with open('./ddi/ddi-test.pickle', 'rb') as handle:
	# sent_names = pickle.load(handle)
	sentences = pickle.load(handle)
	sent_lengths = pickle.load(handle)
	W_te = pickle.load(handle)
	Y_onehot = pickle.load(handle)
	wv = pickle.load(handle)
	word_list = pickle.load(handle)
	rev_word_list = pickle.load(handle)
	label_dict = pickle.load(handle)
	rev_label_dict = pickle.load(handle)

print "Test data loaded"

num_total = len(W_te)
seq_len = len(W_te[0])
# word_dict_size = len(word_list)
# label_dict_size = len(label_dict)

pred = test_step(W_te,Y_onehot)

y_true = np.argmax(Y_onehot, 1)
y_pred = pred
print(classification_report(y_true, y_pred,[1,2,3,4,5,6,7,8],digits=4))
"""
