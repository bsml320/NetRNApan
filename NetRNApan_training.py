# -*- coding: utf-8 -*-
"""
@author: hxu8
"""
import os
import os,sys
import scipy.io
import numpy as np
from numpy import interp
import argparse
import h5py
import json
import uuid
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.constraints import maxnorm
from keras.initializers import RandomUniform, RandomNormal, glorot_uniform, glorot_normal
from keras.layers import Bidirectional, Input
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import  Dense, Dropout, Permute, Lambda
from keras.layers.merge import multiply
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.models import load_model
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from utils import *

np.random.seed(12345)

#You can use the Hyperopt tool to search for the best parameter in a set of parameters as follow:
params = {'batch_size': 64.0, 'dense_unit': 128.0, 'drop_out_cnn': 0.2738070724985381,
        'drop_out_lstm': 0.16261503928101084, 'filter': 256.0,
        'kernel_initializer': 'random_uniform', 'l2_reg': 1.0960198460047699e-05,
        'learning_rate': 0.00028511592517082153, 'lstm_unit': 128.0, 'pool_size': 3.0,
        'window_size': 10.0}

def seq_matrix(seq_list):
	tensor = np.zeros((len(seq_list),41,4))
	for i in range(len(seq_list)):
		seq = seq_list[i]
		j = 0
		for s in seq:
			if s == 'A' or s == 'a':
				tensor[i][j] = [1,0,0,0]
			if s == 'U' or s == 'u':
				tensor[i][j] = [0,1,0,0]
			if s == 'C' or s == 'c':
				tensor[i][j] = [0,0,1,0]
			if s == 'G' or s == 'g':
				tensor[i][j] = [0,0,0,1]
			if s == 'N' or s == 'n':
				tensor[i][j] = [0,0,0,0]
			j += 1
	
	return tensor
    
def fasta_to_matrix(file_path): 
	os.chdir(file_path)]
    seq_name = [name for name in os.listdir(file_path)]
    
	for name in seq_name:
		if 'pos' in name:
			print (name)
			y = []
			seq = []
			positive_seq_file = open(name)
			lines = positive_seq_file.readlines()
			positive_seq_file.close()
			for line in lines:
				line = line.strip()
				if line[0] == '>':
					y.append(1)
				else:
					seq.append(line)

			X1 = seq_matrix(seq)
			print ('pos_ending!')

		if 'neg' in name:
			print (name)
			y = []
			seq = []
			negative_seq_file = open(name)
			lines = negative_seq_file.readlines()
			negative_seq_file.close()
			for line in lines:
				line = line.strip()
				if line[0] == '>':
					y.append(0)
				else:
					seq.append(line)

			X_1 = seq_matrix(seq)

	X_train = np.concatenate([X1,X_1])    
	y_train = np.concatenate([np.ones(len(X1)), np.zeros(len(X_1))])

	return X_train, y_train
    
def get_model(params):
    inputs = Input(shape = (41, 4,))
    cnn_out = Convolution1D(int(params['filter']), int(params['window_size']),
               	kernel_initializer=params['kernel_initializer'], 
               	kernel_regularizer=regularizers.l2(params['l2_reg']), 
              	activation="relu")(inputs)  
    pooling_out = MaxPooling1D(pool_size=int(params['pool_size']), 
    	strides=int(params['pool_size']))(cnn_out) 
    dropout1 = Dropout(params['drop_out_cnn'])(pooling_out) 
    lstm_out = Bidirectional(LSTM(int(params['lstm_unit']), return_sequences=True, 
    	kernel_initializer=params['kernel_initializer'], 
    	kernel_regularizer=regularizers.l2(params['l2_reg'])), merge_mode = 'concat')(dropout1)
    a = Permute((2, 1))(lstm_out)
    a = Dense(lstm_out._keras_shape[1], activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    attention_out = multiply([lstm_out, a_probs])
    attention_out = Lambda(lambda x: K.sum(x, axis=1))(attention_out)
    dropout2 = Dropout(params['drop_out_lstm'])(attention_out) 
    dense_out = Dense(int(params['dense_unit']), activation='relu', 
    	kernel_initializer=params['kernel_initializer'], 
    	kernel_regularizer=regularizers.l2(params['l2_reg']))(dropout2) 
    output = Dense(1, activation='sigmoid')(dense_out)
    model = Model(input=[inputs], output=output)
    adam = Adam(lr=params['learning_rate'],epsilon=10**-8)  
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[roc_auc])
    return model 


def model_training_cross_validation(X,y,filename):

    fileroc = filename + '/' + 'CV'
    if not os.path.isdir(fileroc):
        os.makedirs(fileroc)
        
    fold=[5] #You can set a set of K values for multiple-fold cross validation
    for k in fold:
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)   
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        es = EarlyStopping(monitor='val_roc_auc', mode='max', verbose=1, patience=10)
    
        font1 = {'family' : 'Times New Roman',
                'weight' : 'normal',
                'size'   : 16}      
        figsize=6.2, 6.2
        figure, ax = plt.subplots(figsize=figsize)
        
        plt.tick_params(labelsize=18)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]  
    
        i = 0
        for train, test in kfold.split(X, y):
            classifier = get_model(params)

            fileroc_CV = fileroc + '/%s' % (str(i))
            if not os.path.isdir(fileroc_CV):
                os.makedirs(fileroc_CV)  
            #Save model and weights to file
            mc = ModelCheckpoint(fileroc_CV + '/best_model.h5', monitor='val_roc_auc', mode='max', verbose=1, save_best_only=True, save_weights_only=True)
            classifier.fit(X[train], 
                            y[train],
                            batch_size=64,
                            epochs = 100,
                            shuffle=True,
                            callbacks=[es, mc],
                            validation_data=(X[test], y[test]),
                            verbose=1)
       
            saved_model = load_model(fileroc_CV + '/best_model.h5')
            probas_ = saved_model.predict(X[test])       
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_)
            tprs.append(interp(mean_fpr, fpr, tpr))
    
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))
    
            i += 1
       
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)
    
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
    
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
    
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', font1)
        plt.ylabel('True Positive Rate', font1)
        title='ROC Curve'
        plt.title(title, font1)
        plt.legend(loc="lower right")
        plt.savefig(fileroc + '/' + 'All_feature_CV5_roc.png', dpi=300, bbox_inches = 'tight')

def Main(): 
    parser=argparse.ArgumentParser(description='model_training')
    parser.add_argument("-f", "--fasta", type=str, help="Filepath for training data")
    parser.add_argument("-o", "--out", type=str, help="Training output")
    args = parser.parse_args()
    infile = args.fasta
    filename = args.out
    X,y = fasta_to_matrix(infile)
    model_training_cross_validation(X,y,filename)
   
if __name__ == "__main__":
    Main()