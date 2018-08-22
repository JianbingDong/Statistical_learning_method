# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:30:06 2018

@author: Jianbing_Dong
"""


from calculate_entropy import *
import numpy as np


def create_tree(dataset, attribute, eposilon=1e-16, 
               data_type=np.int32, class_type=np.int32,
               algorithm_type='ID3'):
    """
    This function is used to create the decision tree with ID3 algorithm.
    #arguments:
        dataset: np.array, organized like [[attributes, labels]];
        attribute: 1 rank np.array with integers, the indexes of the 
                    features in datasets. like [0, 1, 2];
        eposilon: the threash hold to create decision tree.
        algorithm_type: str, the algorithm used to create the decision_tree 
                        is specified by this parameter. 
                        default to 'ID3', means create with ID3 algorithm;
                        also could set to 'C4.5', means create with C4.5 algorithm.
    #returns:
        the decision_tree root node.
    """
    
    algorithm_dict = {'ID3': information_gain,
                      'C4.5': information_gain_ratio}
                      
    information_function = algorithm_dict[algorithm_type]
    
    if len(attribute) == 0:
        maxnum_class = cal_max_class_num(dataset)
        node = Class_Node(maxnum_class)
        return node
    
    the_classes = get_class(dataset[:,-1], dtype=class_type)
    if the_classes.size == 1:
        node = Class_Node(the_classes[0])
        return node
        
    Ag_gDA = 0
    Ag = attribute[0]
    for att_index in attribute:
        gDA = information_function(dataset, dataset[:, att_index], 
                        data_type=data_type, 
                        class_type=class_type)
        
        if gDA > Ag_gDA:
            Ag_gDA = gDA
            Ag = att_index
            
    if Ag_gDA < eposilon:
        maxnum_class = cal_max_class_num(dataset)
        node = Class_Node(maxnum_class)
        return node
        
    node = Feature_Node(Ag)
    Ag_atts = get_att_range(dataset[:, Ag])
    i = 0
    for att in Ag_atts:
        Ag_index_i = np.where(dataset[:, Ag] == att)
        
        dataset_i = dataset[Ag_index_i]
        
        attribute = np.array(attribute)
        attribute_i = np.delete(attribute, np.where(attribute==Ag))
        attribute_i = list(attribute_i)
        attribute = list(attribute)
        
        node_i = create_tree(dataset_i, attribute = attribute_i, 
                            eposilon=1e-16, data_type=data_type,
                            class_type=class_type)
        
        node._set_att('node_' + str(i), node_i, node_num=i+1)
        node._set_att('feature_' + str(i), att, node_num=i+1)
        i += 1
        
    return node
    
    
def find_class(node, data):
    """
    #arguments:
        data: np.array, the data need to decide class.
        node: the begin node to find class_.
    #returns:
        class_: integer, the final decision class label.
    """
    if hasattr(node, 'class_'):
        #This is Class_Node
        return node.class_
        
    #This is Feature_Node
    feature_axis = node.feature_axis

    node_num = node.node_num

    class_ = None
    for i in range(node_num):
        feature_i = getattr(node, 'feature_' + str(i))
        if data[feature_axis] == feature_i:
            node_i = getattr(node, 'node_' + str(i))
            class_ = find_class(node_i, data)
            break
        
    return class_
    
    
def decision_tree(data, datasets, attributes, eposilon=1e-16,
                  data_type=np.int32, class_type=np.int32):
    """
    This function is used to decide which class the data belongs to based on
    the decision tree created with dataset and attribute.
    #arguments:
        data: 1 rank np.array, the data needs to be classified.
        datasets: np.array, organized like [[attributes, labels]],
                    datas used to build decision tree.
        attributes: 1 rank np.array with integers, the indexes of the 
                    features in datasets. like [0, 1, 2];
        eposilon: the threash hold to create decision tree.
    """
    root_node = create_tree(datasets, attributes, eposilon=1e-16,
                           data_type=data_type, class_type=class_type)
    
    class_ = find_class(root_node, data)
    
    print(data, 'belongs to', class_)
        
        
   
class Class_Node(object):
    
    def __init__(self, class_):
        
        self.class_ = class_
#        self.type = 'class_node'
        
            
            
class Feature_Node(object):
    
    def __init__(self, feature_axis):
        
        self.feature_axis = feature_axis
#        self.type = 'feature_node'
            
    def _set_att(self, key, value, node_num):
        setattr(self, key, value)
        self.node_num = node_num
        


if __name__ == '__main__':
    
    datasets = np.array([[1, 1, 1, 1, 0],
                         [1, 1, 1, 2, 0],
                         [1, 2, 1, 2, 1],
                         [1, 2, 2, 1, 1],
                         [1, 1, 1, 1, 0],
                         [2, 1, 1, 1, 0],
                         [2, 1, 1, 2, 0],
                         [2, 2, 2, 2, 1],
                         [2, 1, 2, 3, 1],
                         [2, 1, 2, 3, 1],
                         [3, 1, 2, 3, 1],
                         [3, 1, 2, 2, 1],
                         [3, 2, 1, 2, 1],
                         [3, 2, 1, 3, 1],
                         [3, 1, 1, 1, 0]])
    
    data = np.array([1, 2, 1, 2])
    
    decision_tree(data, datasets, 
                  attributes=[0, 1, 2, 3],
                  eposilon=1e-16)

    

