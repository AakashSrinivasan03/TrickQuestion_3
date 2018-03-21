
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
	test_X = test_X.reshape(test_X.shape[0],28,28)


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
	with tf.device('/cpu:0'):	
		X=tf.placeholder(tf.float32,shape=(None, 28,28,1))
		y=tf.placeholder(tf.float32,shape=(None,10))
		mode_placeholder = tf.placeholder(tf.bool)
	with tf.device('/gpu:0'):	
		conv1 = tf.layers.conv2d(X, 64, kernel_size=[3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu)    
		pool1 = tf.layers.max_pooling2d(conv1, 2, 2) 

		conv2 = tf.layers.conv2d(pool1, 128, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu)    
		pool2 = tf.layers.max_pooling2d(conv2, 2, 2) 

		conv3 = tf.layers.conv2d(pool2, 256, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu)    
		conv4 = tf.layers.conv2d(conv3, 256, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu)    


		pool3 = tf.layers.max_pooling2d(conv4, 2, 2) 
		op=tf.shape(pool3)

		flat = tf.reshape(pool3, [tf.shape(X)[0],256*3*3 ])      
		fc1 = tf.layers.dense(flat, 1024,activation=tf.nn.relu)
		fc2 = tf.layers.dense(fc1, 1024,activation=tf.nn.relu)

		fc3 = tf.layers.dense(fc2, 10)
		bn = tf.layers.batch_normalization(fc3, axis=1, center=True, scale=False, training=mode_placeholder)

		##fc3_bn=tf.layers.batch_normalization(fc3,axis=1)
		#tf.nn.batch_normalization(fc3, )
		y_pred = tf.nn.softmax(fc3)





		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc3,
		                                                        labels=y)
		loss = tf.reduce_mean(cross_entropy)
		if(args.opt=='adam'):
			optimizer = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)
		if(args.opt=='gd'):
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr).minimize(loss)	
		_,accuracy=tf.metrics.accuracy(tf.argmax(y,axis=1),tf.argmax(y_pred,axis=1))
		#correct_prediction = tf.equal(y_pred, y)
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



		#session = tf.Session()
		#train_batch_size = args.batch_size
		#args.batch_size=100
		sess=tf.Session()
		init = tf.local_variables_initializer()
		#if(args.init==1):
		#	a
		#else:
		#	b	
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		sess.run(init)
		sess.run(tf.global_variables_initializer())
	with tf.device('/cpu:0'):
		saver = tf.train.Saver()
		#sess.run(tf.global_variables_initializer())
		#print(sess.run(tf.trainable_variables()))
		
		#sess.run(tf.global_variables_initializer())
		#print(sess.run(tf.trainable_variables()))
	with tf.device('/cpu:0'):	
		steps=0
		count=0
		validation_list=[]
		for epochs in range(0,MAX_EPOCHS):
			for i in range(0,int(train_X.shape[0]/args.batch_size)):
				X_batch=train_X[i*args.batch_size:np.minimum((i+1)*args.batch_size,55000),:,:]
				#print(train_Y.shape)
				y_batch=train_Y[i*args.batch_size:np.minimum((i+1)*args.batch_size,55000),:]
				#print(sess.run(tf.shape(np.expand_dims(X_batch,3))))
				#print(sess.run([tf.argmax(y_pred,axis=1),tf.argmax(y,axis=1)],feed_dict={X:np.expand_dims(X_batch,3),y:y_batch}))
				#print(sess.run([tf.argmax(fc3,axis=1),tf.argmax(y,axis=1)],feed_dict={X:np.expand_dims(X_batch,3),y:y_batch}))
				#print("aaa",sess.run(op,feed_dict={X:np.expand_dims(X_batch,3),y:y_batch}))
				##print(sess.run([tf.shape(logits),tf.shape(y)],feed_dict={X:np.expand_dims(X_batch,3),y:y_batch}))
				_,val=sess.run([optimizer,accuracy],feed_dict={X:np.expand_dims(X_batch,3),y:y_batch, mode_placeholder:True})
				print(steps,":",val)
				steps += 1
				##print(val)
			count += 1
			if(count<=5):
				validation_list.append(sess.run(accuracy,feed_dict={X:np.expand_dims(val_X,3),y:val_Y, mode_placeholder:False}))
			else:
				curr_val_1=sess.run(accuracy,feed_dict={X:np.expand_dims(val_X,3),y:val_Y})
				curr_val_2=sess.run(accuracy,feed_dict={X:np.expand_dims(val_X,3),y:val_Y})
				curr_val=curr_val_1+curr_val_2
				print("VAL_LOSS",curr_val)	
				if(curr_val<min(validation_list) and args.early_stop):
					save_path = saver.save(sess, "/output/model.ckpt")
					exit()
				validation_list[count%5]=curr_val	

			
			if(count%25==0):
				test_prediction=sess.run([tf.argmax(y_pred,axis=1)],feed_dict={X:np.expand_dims(test_X,3), mode_placeholder:False})[0]
				test_id = np.array([i for i in range(len(test_prediction))])
				print (test_prediction)
				print(len(test_prediction))
				#print(test_id)
				columns = ['label']
				sub = pd.DataFrame(data=test_prediction, columns=columns)
				sub['id'] = test_id
				sub = sub[['id','label']]
				save_path = saver.save(sess, "/output/model.ckpt")




				sub.to_csv("/output/sub_30_"+str(count)+".csv", index=False)

		
	MAX_EPOCHS = 25
	alpha = 0.0001
	gamma = 0.001
	eps = 0.000001
	save_path = saver.save(sess, "/output/model.ckpt")










def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", default=0.005, type=float, help="initial learning rate for gradient descent based algorithms")
	parser.add_argument("--init", default=1, type=float, help="initialisation for weights 1-for Xavier,2-He")
	parser.add_argument("--early_stop", default=False, type=bool, help="Early stopping")
	parser.add_argument("--opt", default='gd',type=str, help="the optimization algorithm to be used: gd, momentum, nag, adam - you will be implementing the mini-batch version of these algorithms")
	parser.add_argument("--batch_size",default=50 ,type=int, help="the batch size to be used - valid values are 1 and multiples of 5")
	
	parser.add_argument("--save_dir",default='Save_Dir',type=str, help="the directory in which the pickled model should be saved - by model we mean all the weights and biases of the network")
	parser.add_argument("--expt_dir",default='Expt_Dir' ,type=str, help= "the directory in which the log files will be saved - see below for a detailed description of which log files should be generated")
	parser.add_argument("--train",default="/fashionmnist/train.csv",type=str, help="path to the Training dataset")
	parser.add_argument("--test",default="/fashionmnist/test.csv" ,type=str, help = "path to the Test dataset")
	parser.add_argument("--val",default="/fashionmnist/val.csv" ,type=str, help = "path to the val dataset")
	parser.add_argument("--pretrain",default=False ,type=str, help ="pretrain")
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	main()