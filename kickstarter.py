import numpy as np
import pandas as pd
import csv, json
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from collections import Counter
import tensorflow as tf


# utils
COUNTRY_MAP = {	 'AU': '6',
				 'CA': '7',
				 'DE': '0',
				 'DK': '4',
				 'GB': '2',
				 'IE': '8',
				 'NL': '1',
				 'NO': '9',
				 'NZ': '3',
				 'SE': '5',
				 'US': '10',
				 'others': '11'}

CURRENCY_MAP = {	 'AUD': '7',
					 'CAD': '2',
					 'DKK': '8',
					 'EUR': '6',
					 'GBP': '5',
					 'NOK': '3',
					 'NZD': '1',
					 'SEK': '4',
					 'USD': '0',
					 'others':'9'
				}

def csv_to_list(filename):
	with open(filename, 'r') as csvfile:
		file_reader = csv.reader(csvfile, delimiter=',')
		# Converting the csv file reader to a lists 
		data_list = list(file_reader)

	return data_list

def get_headers_and_data(data_list):
	header = data_list[0] 
	data_list = data_list[1:]
	return header,data_list

def split_in_test_and_train(X,Y):
	test_size=0.2

	X = np.array(X)
	Y = np.array(Y)
	testing_size = int(test_size * len(X))

	train_x = list(X[:-testing_size])
	train_y = list(Y[:-testing_size])
	print ('here')
	test_x = list(X[::-1][:testing_size])
	test_y = list(Y[:testing_size])
	return train_x, train_y, test_x, test_y


def process_train():
	filename = "datasets/train.csv"
	data_list = csv_to_list(filename)

	header, data_list = get_headers_and_data(data_list) 
	data_list = np.asarray(data_list)

	selected_data = data_list[:, [0,3, 5, 6, 7, 8, 9, 10, 11, 13]]
	data_list = None
	df = pd.DataFrame(data=selected_data[0:,1:],
				 index=selected_data[0:,0],
					columns=['goal','disable_communication',
					'country','currency','deadline','state_changed_at','created_at',
					'launched_at','final_status'],
							dtype='str')
	df['disable_communication'] = df['disable_communication'].map({'false':'0','true':'1'})
	df['country'] = df['country'].map(COUNTRY_MAP)
	df['currency'] = df['currency'].map(CURRENCY_MAP)
	df['final_status'] = df['final_status'].map({'1':[0,1],'0':[1,0]})
	return df

def process_test():
	filename = "datasets/test.csv"
	data_list = csv_to_list(filename)

	header, data_list = get_headers_and_data(data_list) 
	data_list = np.asarray(data_list)

	selected_data = data_list[:, [0,3, 5, 6, 7, 8, 9, 10,11]]
	data_list = None
	df = pd.DataFrame(data=selected_data[0:,1:],
				 index=selected_data[0:,0],
					columns=['goal','disable_communication',
					'country','currency','deadline','state_changed_at','created_at',
					'launched_at'],
							dtype='str')
	df['disable_communication'] = df['disable_communication'].map({'false':'0','true':'1'})
	df['country'] = df['country'].map(COUNTRY_MAP)
	df['currency'] = df['currency'].map(CURRENCY_MAP)
	df=df.fillna(0)
	return df

def pre_process_train(df):
	from sklearn.utils import shuffle
	df = shuffle(df)
	X = df.iloc[:,[0,1,2,3,4,5,6,7,]].values
	Y = df.iloc[:,8].values

	# one hot encoding 
	one_hot_encoder = OneHotEncoder(categorical_features=[1,2,3])
	X = one_hot_encoder.fit_transform(X).toarray()

	# Y_one_hot_encoder = OneHotEncoder(categorical_features =[8] )
	# Y = Y_one_hot_encoder.fit_transform(Y).toarray()

	# # scaling
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	# Y = scaler.fit_transform(Y)
	return X,Y

def pre_process_test(df):
	X = df.iloc[:,[0,1,2,3,4,5,6,7,]].values

	one_hot_encoder = OneHotEncoder(categorical_features=[1,2,3])
	X = one_hot_encoder.fit_transform(X).toarray()

	# # scaling
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	return X


def make_model(data,train_x):

	n_nodes_hl1 = 1000
	n_nodes_hl2 = 1000
	n_nodes_hl3 = 1000

	n_classes = 2 # No of classification
	num_classes = 2
	num_hidden = 128


	hidden_1_layer = {'weights': tf.Variable(tf.truncated_normal([len(train_x[0]), n_nodes_hl1], stddev=0.1),name= 'weights'),
					  'biases': tf.Variable(tf.constant(0.01, shape=[n_nodes_hl1]),name = 'biases')}

	hidden_2_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1),name= 'weights'),
					  'biases': tf.Variable(tf.constant(0.01, shape=[n_nodes_hl2]),name = 'biases')}

	hidden_3_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1,),name= 'weights'),
					  'biases': tf.Variable(tf.constant(0.01, shape=[n_nodes_hl3]),name = 'biases')}

	output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev=0.1),name= 'weights'),
					'biases': tf.Variable(tf.constant(0.01, shape=[n_classes]),name = 'biases'), }


	layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	# now goes through an activation function - sigmoid function
	layer_1 = tf.nn.relu(layer_1)
	layer_1 = tf.layers.dropout(layer_1, rate=0.1,name = "dropout")

	print ("Layer 1 done!!")
	# input for layer 2 = result of activ_func for layer 1
	layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	layer_2 = tf.nn.relu(layer_2)
	layer_2 = tf.layers.dropout(layer_2, rate=0.1,name = "dropout")

	print ("Layer 2 done!!")

	layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	layer_3 = tf.nn.relu(layer_3)
	layer_3 = tf.layers.dropout(layer_3, rate=0.1,name = "dropout")

	print ("Layer 3 done!!")

	output = tf.matmul(layer_3, output_layer['weights'],name = "output") + output_layer['biases']

	return output


def train_neural_network(train_x,train_y,test_x,test_y,predict_x=None):
	prediction_list = []
	tf.reset_default_graph()
	with tf.name_scope('input'):
		x = tf.placeholder('float', [None, len(train_x[0])],name= 'x_input')
		y = tf.placeholder('float',name = 'y-input')
	# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
	

	logits = make_model(x,train_x)
	prediction = tf.nn.softmax(logits,name="prediction")
	print ('model ready!!')
	with tf.name_scope('pred'):
		pred = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y,name="softmax")
	with tf.name_scope('cost'):
		cost = tf.reduce_mean(pred)
	with tf.name_scope('train'):
		optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(cost)

	n_epochs = 10
	batch_size = 100


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())  # initializes our variables. Session has now begun.
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter('hacker/train/1/',
											  sess.graph)
		test_writer = tf.summary.FileWriter('hacker/test/1/')

		for epoch in range(n_epochs):
			epoch_loss = 0  # we'll calculate the loss as we go

			i = 0
			while i < len(train_x):
				#we want to take batches(chunks); take a slice, then another size)
				start = i
				end = i+batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				if i%200 == 0:
					train_writer.add_summary(_, i)
				epoch_loss += c
				i+=batch_size
			print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)
			with tf.name_scope('accuracy'):
				correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
				accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
				tf.summary.scalar("accuracy", accuracy)
					   

			print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
			saver = tf.train.Saver()
			tf_log = 'tf.log'
			saver.save(sess, "hacker3.ckpt")

	return prediction_list



def test_neural_network(test_x):
	
	batch_size = len(test_x)/5
	i = 0
	tf.reset_default_graph()
	x = tf.placeholder('float', [batch_size,len(test_x[0])])
	y = tf.placeholder('float',[2])
	prediction = make_model(x,test_x)
	pred1 = tf.nn.softmax(logits=prediction)
	prediction_list = []
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, "hacker3.ckpt")

		i = 0
		while i < len(test_x):
			#we want to take batches(chunks); take a slice, then another size)
			start = int(i)
			end = int(i+batch_size)

			batch_x = np.array(test_x[start:end])

			pre = sess.run(tf.argmax(pred1.eval(feed_dict={x:batch_x}),1))
			prediction_list.extend(pre)
			i+=batch_size
	return prediction_list





