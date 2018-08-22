# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 09:59:09 2018

@author: Jianbing_Dong
"""


import numpy as np



def get_class(classes, dtype = np.float64):
    """
    this function is used to calculate the classes.
    #arguments:
        classes: np.array, like [1, -1, 1, -1]
    """
    classes = np.reshape(classes, newshape=(classes.size,))
    theclass = classes[0]
    for class_i in classes[1:]:
        index = np.where(theclass == class_i)
        if index[0].size == 0:
            theclass = np.append(theclass, class_i)
            
    if isinstance(theclass, dtype):
        theclass = np.reshape(theclass, newshape=(1,))
        
    return theclass
    
    
def get_att_range(attribute, dtype = np.float64):
    """
    This function is used to get the number of the range of 
    the attribute.
    #arguments:
        attribute: np.array, like [0.1, 0.5, 0.1, 0.8]
    #Returns:
        the num: np.array, like [0.1, 0.5, 0.8]
    """
    attribute = np.reshape(attribute, newshape=(attribute.size,))
    num = attribute[0]
    for att_i in attribute:
        index = np.where(num == att_i)
        if index[0].size == 0:
            num = np.append(num, att_i)
            
    if isinstance(num, dtype):
        num = np.reshape(num, newshape=(1,))
            
    return num

    
    
def empirical_entropy(dataset, dtype = np.float64):
    """
    This function is used to calculate the empirical entropy of the
    dataset.
    #arguments:
        dataset: np.array, the dataset include attributes and labels.
                origanized with [[attributes, labels]]
    #returns:
        HD: the empirical entropy of the dataset.
    """
    if dataset.ndim == 1:
        dataset = np.reshape(dataset, newshape=(1,) + dataset.shape)
    the_class = get_class(dataset[:,-1], dtype=dtype)
    
    HD = 0
    D = dataset.shape[0]
    for class_i in the_class:
        Ck = np.where(dataset[:, -1] == class_i)[0].size + 1e-16
        HD -= (Ck / D) * np.log2(Ck / D)
        
    return HD
    
    
def conditional_entropy(dataset, attribute, 
                        data_type=np.int32, 
                        class_type=np.int32):
    """
    This function is used to calculate the conditional entropy with the
    attribute and the dataset.
    #arguments:
        dataset: np.array;
        attribute: np.array. one attribute from dataset.
    """
    D = dataset.shape[0]
    
    HDA = 0
    att_range = get_att_range(attribute, dtype=data_type)
    for att_i in att_range:
        index = np.where(attribute == att_i)
        data_i = dataset[index]
        Di = data_i.shape[0]
        
        HDA += (Di / D) * empirical_entropy(data_i, dtype=class_type)
        
    return HDA


    
def information_gain(dataset, attribute, 
                        data_type=np.int32, 
                        class_type=np.int32):
    """
    Calculate the information gain of the dataset about attribute.
    #arguments:
        dataset: np.array, organized like [[attributes, labels]]
        attributes: np.array, organized like 
                [atttibutes_i_1, atttibutes_i_1, ..., atttibutes_i_n]
        data_type: numpy type, the type of the dataset.
        class_type: numpy type, the type of the labels.
    """
    HD = empirical_entropy(dataset, dtype=class_type)
    HDA = conditional_entropy(dataset, attribute, data_type=data_type,
                              class_type=class_type)
    
    gDA = HD - HDA
    
    return gDA
    
    
def information_gain_ratio(dataset, attribute, 
                        data_type=np.int32, 
                        class_type=np.int32):
    """
    Calculate the information gain of the dataset about attribute.
    #arguments:
        dataset: np.array, organized like [[attributes, labels]]
        attributes: np.array, organized like 
                [atttibutes_i_1, atttibutes_i_1, ..., atttibutes_i_n]
        data_type: numpy type, the type of the dataset.
        class_type: numpy type, the type of the labels.
    """
    gDA = information_gain(dataset, attribute, data_type, class_type) 
    
    ###HAD
    D = dataset.shape[0]
    
    HAD = 0
    att_range = get_att_range(attribute, dtype=data_type)
    for att_i in att_range:
        index = np.where(attribute == att_i)
        data_i = dataset[index]
        Di = data_i.shape[0]
        
        HAD -= (Di / D) * np.log2(Di / D + 1e-16)
    ###

    gRDA = gDA / HAD

    return gRDA    
    
    
def cal_max_class_num(dataset):
    """
    This function is used to calculate the max number of classes in dataset.
    #arguments:
        dataset: np.array, organized like [[attributes, labels]]
    #return:
        maxnum_class: integer, the class label which has the max number of classes.
                        if their are more than one class have the max number, then
                        choose the former class as the output.
    """
    classes = dataset[:,-1]
    the_class = get_class(classes)
    
    maxnum_class = the_class[0]
    class_num = 1
    for class_i in the_class:
        indexes = np.where(classes == class_i)[0]
        if indexes.size > class_num:
            class_num = indexes.size
            maxnum_class = class_i
            
    return maxnum_class
    
    
    

#%%
if __name__ == '__main__':
    data = np.array([[1, 1, 1, 1, 0],
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
    
    gDA = information_gain(data, data[:, 3], 
                           data_type=np.int32,
                           class_type=np.int32)
    print(gDA)
    
    gRDA = information_gain_ratio(data, data[:, 3], 
                           data_type=np.int32,
                           class_type=np.int32)
    print(gRDA)
    
    max_ = cal_max_class_num(data)
    print(max_)
    
    
    
    

