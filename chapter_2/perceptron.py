# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 19:19:18 2018

@author: Jianbing_Dong
"""

#%%
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt

import argparse


#%%
class Instance2_1(object):
    
    def __init__(self, keep_oneline, learning_rate=1e-3, max_step = 1000):
        
        self.learning_rate = learning_rate
        self.max_step = max_step
        self.keep_oneline = keep_oneline
        
        self.initialize()
    
    def get_data(self):
        
        data = np.array([[3, 3],
                         [4, 3],
                         [4.5, 4.5],
                         [5, 4],
                         [3.5, 4.5],                   
                         [1, 1],
                         [1, 4],
                         [1, 0],
                         [2.5, 1.5],
                         [0, 1.5]])
        
        label = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
        
        fig = plt.figure()
    
        self.subfig = fig.add_subplot(1, 1, 1)
        self.subfig.scatter(data[:5, 0], data[:5, 1], marker='o', s=50)
        self.subfig.scatter(data[5:, 0], data[5:, 1], marker='x', s=80)
        self.subfig.set_xlabel('x1')
        self.subfig.set_ylabel('x2')
        self.subfig.axis([-1,6,-1,6])    
        
        plt.ion()
        
        return data, label
        
    def initialize(self):
        self.w = np.random.randn(1, 2)
        self.b = np.random.randn()
        
        
    def data_batch(self):
        data, label = self.get_data()
        while True:
            index = np.random.randint(0, data.shape[0])
            data_i = data[index]
            label_i = label[index]
             
            yield data_i, label_i        
        
    def inference(self, x, y):
        logit = np.matmul(self.w, x) + self.b
        if logit >=0:
            y_hat = 1
        else:
            y_hat = -1
            
        return y_hat
    
    
    def update(self, label_i, data_i):
        if label_i * (np.matmul(self.w, data_i) + self.b) <= 0:
            self.w += self.learning_rate*label_i*data_i
            self.b += self.learning_rate*label_i
        
    
    def train(self, show_n):
        
        i = 0
        for data_batch, label_batch in self.data_batch():
            self.inference(data_batch, label_batch)
            self.update(label_batch, data_batch)
            
            if i>self.max_step:
                break
            
            if i % show_n == 0:
                
                if self.keep_oneline and len(self.subfig.lines) > 1:
                    self.subfig.lines.pop(1)
                    
                show_x1 = np.linspace(0, 6, 100)
                show_x2 = -(self.b + self.w[0, 0]*show_x1) / self.w[0, 1]
                
                self.subfig.plot(show_x1, show_x2, label=str(i))
                self.subfig.legend(loc='upper left')
                
                plt.pause(0.1)
                plt.show()
                
            i += 1
            
        plt.waitforbuttonpress()

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_oneline', type=eval,
                        default=True, choices=[True, False])
    flags = parser.parse_args()
    
    theinstance = Instance2_1(keep_oneline=flags.keep_oneline,
                              max_step=500, learning_rate=0.5)

    theinstance.train(show_n=1)
    


