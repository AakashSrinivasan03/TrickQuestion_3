

# -*- coding: utf-8 -*-
import pickle
import argparse
import sys
from datetime import datetime
import os
import pandas as pd
import numpy as np
import math
import scipy as sp
import scipy.special
import sklearn
import sklearn.metrics
import sklearn.preprocessing
import tensorflow as tf
def main():
	global args 
	args = parse_arguments()
	subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
	log_dir = os.path.join(args.expt_dir, subdir)
	if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
		os.makedirs(log_dir)
	model_dir = os.path.join(args.save_dir, subdir)
	if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist	os.makedirs(model_dir)
		os.makedirs(model_dir)

	#read data
	data_train = pd.read_csv(args.train).iloc[:,:].as_matrix()
	train_X = data_train[:,1:-1]

	train_X=sklearn.preprocessing.normalize(train_X)
	train_Y = data_train[:,-1]

	data_val = pd.read_csv(args.val).iloc[:,:].as_matrix()
	val_X = data_val[:,1:-1]
	val_X=sklearn.preprocessing.normalize(val_X)
	val_Y = data_val[:,-1]

	data_test = pd.read_csv(args.test).iloc[:,:].as_matrix()
	test_X = data_test[:,1:]
	test_X=sklearn.preprocessing.normalize(test_X)


	m_train = train_X.shape[0]
	m_val  = val_X.shape[0]
	features = 784
	classes = 10
	MAX_EPOCHS = 25
	alpha = 0.0001
	gamma = 0.001
	eps = 0.000001


	# One hot encoded form
	train_Y1 = np.zeros((m_train,classes), dtype=np.int8)
	for i in range(m_train):
		train_Y1[i][int(train_Y[i])] = 1

	val_Y1 = np.zeros((m_val,classes) , dtype=np.int8)
	for i in range(m_val):
		val_Y1[i][int(val_Y[i])] = 1
	#print(train_Y1.shape)	
	train = np.concatenate((train_X, train_Y1), axis=1)
	val = np.concatenate((val_X, val_Y1), axis=1)

	np.random.shuffle(train)
	np.random.shuffle(val)

	train_X = train[:,0:features].reshape(train.shape[0],28,28)
	train_Y = train[:,features:]
	val_X  = val[:,0:features].reshape(val.shape[0],28,28)
	val_Y  = val[:,features:]
	print(train_X.shape)
	#scalar=sklearn.preprocessing.StandardScaler()
	#scalar=scalar.fit(train_X)
	#train_X=scalar.transform(train_X)
	#print(train_Y.shape)
	
	layers_size=args.sizes.split(',')
	layers_size.append('10')
	
	num_layers=len(layers_size)

	test_prediction=[]
	
	'''if (args.pretrain):
		with open('model/W.pickle', 'rb') as handle:
			W_dict = pickle.load(handle)
		with open('model/b.pickle', 'rb') as handle:
    			b_dict = pickle.load(handle)
	else:
		W_dict,b_dict=minibatch_gradient_descent(W_dict,b_dict,args.num_hidden,train_X,train_Y,args.batch_size,val_X,val_Y)		
'''




	#sub.to_csv("sub_30.csv", index=False)
	'''with open(model_dir+'/W.pickle', 'wb') as handle:
		pickle.dump(W_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)	
	with open(model_dir+'/b.pickle', 'wb') as handle:
		pickle.dump(b_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)'''
	X=tf.placeholder(tf.float32,shape=(None, 28,28,1))
	y=tf.placeholder(tf.float32,shape=(None,10))
	conv1 = tf.layers.conv2d(X, 64, 3, (1, 1), 'same', activation=tf.nn.relu)    
	pool1 = tf.layers.max_pooling2d(conv1, 2, 2) 

	conv2 = tf.layers.conv2d(pool1, 128, 3, 1, 'same', activation=tf.nn.relu)    
	pool2 = tf.layers.max_pooling2d(conv2, 2, 2) 

	conv3 = tf.layers.conv2d(pool2, 256, 3, 1, 'same', activation=tf.nn.relu)    
	conv4 = tf.layers.conv2d(conv3, 256, 3, 1, 'same', activation=tf.nn.relu)    


	pool3 = tf.layers.max_pooling2d(conv4, 2, 2) 
	op=tf.shape(pool3)

	flat = tf.reshape(pool3, [tf.shape(X)[0],256*3*3 ])      
	fc1 = tf.layers.dense(flat, 1024)
	fc2 = tf.layers.dense(fc1, 1024)

	fc3 = tf.layers.dense(fc2, 10)
	#fc3_bn=tf.layers.batch_normalization(fc3,axis=1)
	#tf.nn.batch_normalization(fc3, )
	y_pred = tf.nn.softmax(fc3)





	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,
	                                                        labels=y)
	loss = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
	_,accuracy=tf.metrics.accuracy(tf.argmax(y,axis=1),tf.argmax(y_pred,axis=1))
	#correct_prediction = tf.equal(y_pred, y)
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



	#session = tf.Session()
	#train_batch_size = args.batch_size
	#args.batch_size=100
	sess=tf.Session()
	init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init)
	#sess.run(tf.global_variables_initializer())
	#print(sess.run(tf.trainable_variables()))
	for epochs in range(0,MAX_EPOCHS):
		for i in range(0,int(train_X.shape[0]/args.batch_size)):
			X_batch=train_X[i*args.batch_size:np.minimum((i+1)*args.batch_size,55000),:,:]
			#print(train_Y.shape)
			y_batch=train_Y[i*args.batch_size:np.minimum((i+1)*args.batch_size,55000),:]
			#print(sess.run(tf.shape(np.expand_dims(X_batch,3))))
			#print(sess.run([tf.argmax(y_pred,axis=1),tf.argmax(y,axis=1)],feed_dict={X:np.expand_dims(X_batch,3),y:y_batch}))
			#print(sess.run([tf.argmax(fc3,axis=1),tf.argmax(y,axis=1)],feed_dict={X:np.expand_dims(X_batch,3),y:y_batch}))
			_,val=sess.run([optimizer,accuracy],feed_dict={X:np.expand_dims(X_batch,3),y:y_batch})
			print(val)
			#print(sess.run(accuracy,feed_dict={X:np.expand_dims(val_X,3),y:val_Y}))
			



		
	MAX_EPOCHS = 25
	alpha = 0.0001
	gamma = 0.001
	eps = 0.000001











def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", default=5, type=float, help="initial learning rate for gradient descent based algorithms")
	parser.add_argument("--momentum" ,default=0.9, type=float, help="momentum to be used by momentum based algorithms")
	parser.add_argument("--num_hidden",default=1,type=int, help="number of hidden layers - this does not include the 784 dimensional input layer and the 10 dimensional output layer")
	parser.add_argument("--sizes",default='256' ,type=str, help="a comma separated list for the size of each hidden layer")
	parser.add_argument("--activation",default='sigmoid',type=str, help="the choice of activation function - valid values are tanh/sigmoid")
	parser.add_argument("--loss",default='ce' ,type=str, help="possible choices are squared error[sq] or cross entropy loss[ce]")
	parser.add_argument("--opt", default='adam',type=str, help="the optimization algorithm to be used: gd, momentum, nag, adam - you will be implementing the mini-batch version of these algorithms")
	parser.add_argument("--batch_size",default=100 ,type=int, help="the batch size to be used - valid values are 1 and multiples of 5")
	parser.add_argument("--anneal" ,default=True,type=bool,help="if true the algorithm should halve the learning rate if at any epoch the validation loss decreases and then restart that epoch")
	parser.add_argument("--save_dir",default='Save_Dir',type=str, help="the directory in which the pickled model should be saved - by model we mean all the weights and biases of the network")
	parser.add_argument("--expt_dir",default='Expt_Dir' ,type=str, help= "the directory in which the log files will be saved - see below for a detailed description of which log files should be generated")
	parser.add_argument("--train",default="../fashionmnist/train.csv",type=str, help="path to the Training dataset")
	parser.add_argument("--test",default="../fashionmnist/test.csv" ,type=str, help = "path to the Test dataset")
	parser.add_argument("--val",default="../fashionmnist/val.csv" ,type=str, help = "path to the val dataset")
	parser.add_argument("--pretrain",default=False ,type=str, help ="pretrain")
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	main()
