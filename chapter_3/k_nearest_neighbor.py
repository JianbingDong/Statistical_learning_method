# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:59:42 2018

@author: Jianbing_Dong
"""

#%%
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt



#%%

class K_NN(object):
    
    def __init__(self):
        pass
    
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
                         [0, 1.5],
                         [3.5, 1],
                         [4, 2],
                         [4.5, 0.5],
                         [5.5, 1.5],
                         [2.5, 0]])
        
        label = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0])
        label = ['blue']*5 + ['red']*5 + ['green']*5
        
        fig = plt.figure()
    
        self.subfig = fig.add_subplot(1, 1, 1)
        self.subfig.scatter(data[:5, 0], data[:5, 1], marker='o', s=50)
        self.subfig.scatter(data[5:10, 0], data[5:10, 1], marker='v', s=50, color='r')
        self.subfig.scatter(data[10:, 0], data[10:, 1], marker='^', s=50, color='g')
        self.subfig.set_xlabel('x1')
        self.subfig.set_ylabel('x2')
        self.subfig.axis([-1,6,-1,6])    
        
        return data, label    
        
        
    def distance(self, data1, data2):
        dis = np.sqrt(np.sum((data1 - data2)**2))
        return dis
        
        
    def create_kd_tree(self, data, axis, father_node):
        
        if len(data) == 0:
            return None
            
        median = np.median(data[:, axis])
        
        median_indexes = np.where(data[:,axis]==median)[0]
        if len(median_indexes) != 0:                 
            median_index = np.random.choice(median_indexes, 1)
        else:
            median_index = data.shape[0] // 2

        median_data = data[median_index]
        thenode = Node(median_data, father_node=father_node)
        
        left_index = np.where(data[:,axis]<=median)[0]
        l_median_index = np.where(left_index==median_index)
        left_index = np.delete(left_index, l_median_index) 
        
        left_data = data[left_index]
    
        right_index = np.where(data[:,axis]>median)[0]
        
        right_data = data[right_index]
    
        axis = 1 - axis        

        thenode.l_node = self.create_kd_tree(left_data, axis=axis, father_node=thenode)
        thenode.r_node = self.create_kd_tree(right_data, axis=axis, father_node=thenode)
        
        return thenode
        
    
    def get_leaf(self, input_, node, axis):
        
        if node.l_node == None and node.r_node == None:
            return node

        node.data = np.reshape(node.data, newshape=(1, 2))
        
        if input_[axis] <= node.data[0][axis] and node.l_node != None:
            axis = 1 - axis
            leaf_node = self.get_leaf(input_, node.l_node, axis) 
        elif input_[axis] > node.data[0][axis] and node.r_node != None:
            axis = 1 - axis
            leaf_node = self.get_leaf(input_, node.r_node, axis)
            
        return leaf_node
        
        
    def search(self, input_, leaf_node, nearest_node):
        
        father_node = leaf_node.father_node
        if father_node == None:
            return nearest_node
        
        if self.distance(father_node.data, input_) < self.distance(nearest_node.data, input_):
            nearest_node = father_node
            
        else:
            brother_node = self.get_brother_node(leaf_node)
            if brother_node != None and self.distance(brother_node.data, input_) < self.distance(nearest_node.data, input_):
                nearest_node = brother_node
                
        nearest_node = self.search(input_, father_node, nearest_node)
        
        return nearest_node
            

    def get_brother_node(self, node):
        father_node = node.father_node
        
        if node == father_node.l_node:
            return father_node.r_node
        else:
            return father_node.l_node
        
        
    def main(self, input_):
        data, label = self.get_data()
        self.subfig.scatter(input_[0], input_[1], marker='p', s=50, color='y')
        
        root_node = self.create_kd_tree(data, 0, None)
        
        nearest_node = self.get_leaf(input_, root_node, 0)
            
        nearest_node = self.search(input_, leaf_node=nearest_node, 
                                   nearest_node=nearest_node)
        
        self.subfig.scatter(nearest_node.data[0][0], nearest_node.data[0][1], 
                            marker='o', edgecolor='k', s=200, color='')
        
        plt.title("the nearest point is %s" %np.reshape(nearest_node.data, 
                                                        newshape=(2, )))
        
        plt.show()
        print("the nearest point is", nearest_node.data)
        print("the class is %s" %label[np.where(data==nearest_node.data)[0][1]])
        
            
            
   #%%         
class Node(object):
    
    def __init__(self, data, father_node):
        
        self.data = data
        self.father_node = father_node
        self.l_node = None
        self.r_node = None

        


#%%
if __name__ == '__main__':
    knn = K_NN()
    knn.main(input_=np.array([1, 1.5]))




