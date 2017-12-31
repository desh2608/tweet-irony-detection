import numpy as np
import tensorflow as tf

class Model(object):

	def __init__(self, num_classes, wv, text_len, hashtags_len, emoji_len, num_filters_text = 75, num_filters_hashtags = 80, l2_reg_lambda = 0.01):

		tf.reset_default_graph()

		self.w_text  = tf.placeholder(tf.int32, [None, text_len], name="w_text")
		self.w_hashtags = tf.placeholder(tf.int32, [None, hashtags_len], name="w_hashtags")
		self.w_emoji = tf.placeholder(tf.float32, [None,emoji_len], name="w_emoji")
		self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Initialization
		W_emb = tf.Variable(wv,name='W_emb',trainable=True)

		# Embedding layer
		X_text = tf.nn.embedding_lookup(W_emb, self.w_text)
		X_hashtags = tf.nn.embedding_lookup(W_emb, self.w_hashtags)

		# LSTM layer for text
		with tf.variable_scope('lstm1'):
			lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_filters_text, state_is_tuple=True)
			lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_filters_text, state_is_tuple=True)

			# _X_text = tf.unstack(X_text, num=text_len, axis=1)
			outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs = X_text, dtype=tf.float32)
			# outputs_fw = tf.unstack(outputs[0],axis=1)
			# outputs_bw = tf.unstack(outputs[1],axis=1)
			# output_fw = outputs_fw[len(outputs_fw)-1]
			# output_bw = outputs_bw[0]
			# h1_text = tf.concat([output_fw,output_bw],axis=1)
			outputs = tf.expand_dims(tf.concat(outputs,2),-1)
			h1_text = tf.nn.max_pool(outputs,ksize=[1, text_len, 1, 1],strides=[1, 1, 1, 1], padding='VALID')
			h1_text = tf.squeeze(h1_text,axis=[1,3])

		# LSTM layer for hashtags
		with tf.variable_scope('lstm2'):
			lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_filters_hashtags, state_is_tuple=True)
			lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_filters_hashtags, state_is_tuple=True)

			# _X_hashtags = tf.unstack(X_hashtags, num=hashtags_len, axis=1)
			outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs = X_hashtags, dtype=tf.float32)
			# outputs_fw = tf.unstack(outputs[0],axis=1)
			# outputs_bw = tf.unstack(outputs[1],axis=1)
			# output_fw = outputs_fw[len(outputs_fw)-1]
			# output_bw = outputs_bw[0]
			# h1_hashtags = tf.concat([output_fw,output_bw],axis=1)
			outputs = tf.expand_dims(tf.concat(outputs,2),-1)
			h1_hashtags = tf.nn.max_pool(outputs,ksize=[1, hashtags_len, 1, 1],strides=[1, 1, 1, 1], padding='VALID')
			h1_hashtags = tf.squeeze(h1_hashtags,axis=[1,3])


		print ("h1_text",h1_text.get_shape())
		print ("h1_hashtags",h1_hashtags.get_shape())
		print ("w_emoji",self.w_emoji.get_shape())

		h1_text_emoji = tf.concat([h1_text,self.w_emoji],axis=1)
		print ("h1_text",h1_text_emoji.get_shape())
		# Circular correlation
		h_circ = self.holographic_merge(h1_text_emoji,h1_hashtags)
		print ("h_circ",h_circ.get_shape())

		#Dropout
		h_drop = tf.nn.dropout(h_circ,self.dropout_keep_prob)

		# Fully connetected layer
		W = tf.Variable(tf.truncated_normal([2*num_filters_hashtags, num_classes], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
		scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")


		l2_loss = tf.constant(0.0)
		l2 = l2_reg_lambda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() 
			if not ("noreg" in tf_var.name or "Bias" in tf_var.name))
		l2_loss += l2


		# prediction and loss function
		self.predictions = tf.argmax(scores, 1)
		self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
		self.loss = tf.reduce_mean(self.losses) + l2_loss

		# Accuracy
		self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"))
		self.global_step = tf.Variable(0,name='global_step',trainable=False)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.loss,global_step=self.global_step)

		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(config=session_conf)

		self.sess.run(tf.global_variables_initializer())

	
	def holographic_merge(self,a,b):
		a_fft = tf.fft(tf.complex(a, 0.0))
		b_fft = tf.fft(tf.complex(b, 0.0))
		ifft = tf.ifft(tf.conj(a_fft) * b_fft)
		return tf.cast(tf.real(ifft), 'float32')

	def train_step(self, W_text_batch, W_hashtags_batch, W_emoji_batch, y_batch):
			feed_dict = {
				self.w_text		:W_text_batch,
				self.w_hashtags: W_hashtags_batch,
				self.w_emoji 	:W_emoji_batch,
				self.dropout_keep_prob: 0.5,
				self.input_y 	:y_batch
					}
			_, step, loss, accuracy, predictions = self.sess.run([self.optimizer, self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
			print ("step "+str(step) + " loss "+str(loss) +" accuracy "+str(accuracy))
			return step,accuracy

	def test_step(self, W_text_batch, W_hashtags_batch, W_emoji_batch, y_batch):
			feed_dict = {
				self.w_text 		:W_text_batch,
				self.w_hashtags: W_hashtags_batch,
				self.w_emoji 	:W_emoji_batch,
				self.dropout_keep_prob:1.0,
				self.input_y :y_batch
				}
			loss, accuracy, predictions = self.sess.run([self.loss, self.accuracy, self.predictions], feed_dict)
			return predictions
