# -*- coding: utf-8 -*-
"""
@author: HXu8
"""
import os,sys
import argparse
import numpy as np
from keras.models import Model, load_model, model_from_json
import argparse
import keras
print('Ddeveloped with keras version 2.3.1, and current keras version: ',keras.__version__)

#binding model
model="./models/best_model.json"
weight="./models/best_model.h5"

#function to import keras NN model
def get_model(model_name0,weight_name0):
    model_name0 = model_name0
    weight_name0 = weight_name0
    model0 = model_from_json(open(model_name0).read())
    model0.load_weights(weight_name0)
    return model0

#function to binay encoding
def seq_matrix(infile):
    header,seq_list = [], []
    seq_file = open(infile)
    lines = seq_file.readlines()
    seq_file.close()
    for line in lines:
    	line = line.strip()
    	if line[0] == '>':
    		header.append(line[1:])
    	else:
    		seq_list.append(line)   
    	
    tensor = np.zeros((len(seq_list),41,4))
    for i in range(len(seq_list)):
        seq = seq_list[i]
        j = 0
        for s in seq:
            if s == 'A' or s == 'a':
                tensor[i][j] = [1,0,0,0]
            if s == 'U' or s == 't':
                tensor[i][j] = [0,1,0,0]
            if s == 'C' or s == 'c':
                tensor[i][j] = [0,0,1,0]
            if s == 'G' or s == 'g':
                tensor[i][j] = [0,0,0,1]
            if s == 'N' or s == 'n':
                tensor[i][j] = [0,0,0,0]
            j += 1
    return header, tensor

#function to prediction task
def prediction(infile,outfile):
    header, tensor = seq_matrix(infile)
    dnn_model = get_model(model,weight)
    predictions = dnn_model.predict(tensor) 
    pred_train_prob = predictions
    rounded = [round(x[0]) for x in predictions]
    
    fileout = open(outfile, "w")
    fileout.write("Prediction:\n")
    for i in range(len(header)):
        fileout.write(str(header[i]))
        fileout.write("\t")
        fileout.write(str(rounded[i]))
        fileout.write("\t")
        fileout.write(str(pred_train_prob[i][0]))
        fileout.write("\n")
    fileout.close()

def main():
    parser = argparse.ArgumentParser(description="progrom usage")
    parser.add_argument("-f", "--fasta", type=str, help="prediction instances")
    parser.add_argument("-o", "--out", type=str, help="prediction output")
    args = parser.parse_args()
    infile = args.fasta
    outfile = args.out
    prediction(infile,outfile)

if __name__ == '__main__':
        main()
