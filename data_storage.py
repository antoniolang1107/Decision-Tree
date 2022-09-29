'''
Author: Antonio Lang
Date: 26 September 2022
'''

import numpy as np


def build_nparray(data):
    data_values = np.array(data[1:]) # 2D array with feature and label data

    # split 2D array to separate feature and label data
    feature_data = data_values[:, :len(data_values[0])-1] 
    label_data = data_values[:, len(data_values[0])-1:]
    label_data = np.transpose(label_data) 
    label_data = label_data[0]# convert label_data to 1D array
    feature_data = feature_data.astype(float)
    label_data = label_data.astype(int)
    
    return feature_data, label_data


def build_list(data):
    # converts the numpy arrays to lists
    feature_data, label_data = build_nparray(data)
    feature_data = feature_data.tolist()
    label_data = label_data.tolist()

    return feature_data, label_data


def build_dict(data):
    header_values = data[0]
    feature_names = header_values[:len(header_values)-1] # gets the names of the features
    feature_values = data[1:]
    feature_values = feature_values.astype(float)
    dict_line = zip(feature_names, feature_values)
    dict_data = []
    label_data = []

    for data_object in feature_values:
        features = data_object[:len(data_object)-1]
        label = data_object[len(data_object)-1]
        dict_line = zip(feature_names, features)
        label_data.append(int(label))
        dict_data.append(dict(dict_line))
    feature_dict = dict(enumerate(dict_data))
    label_dict = dict(enumerate(label_data))

    return feature_dict, label_dict