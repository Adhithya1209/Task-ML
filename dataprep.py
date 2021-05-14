# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import numpy as np
import pickle
num_of_samples = 100
num_of_features= 5
dataset = np.random.randn(num_of_samples, num_of_features+1)

outfile = open('Dataset.pkl','wb')
pickle.dump(dataset, outfile)
outfile.close()

