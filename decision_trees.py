import numpy as np
import math

import data_storage as ds

class Node:
    def __init__(self, feature=None, left_tree=None, right_tree=None, ig=None, prediction=None):
        self.feature = feature
        self.left_tree = left_tree
        self.right_tree = right_tree
        self.ig = ig
        self.prediction = prediction
    def is_decision(self):
        return False

class DecisionNode(Node):
    def __init__(self, labels):
        self.labels = labels
        self.prediction = self.calc_prediction()
    def calc_prediction(self):
        num_0 = 0
        num_1 = 0
        for label in self.labels:
            if label == 0:
                num_0 += 1
            elif label == 1:
                num_1 += 1
        if num_0 >= num_1:
            return 0
        else:
            return 1
    def get_prediction(self):
        return self.prediction
    def is_decision(self):
        return True


class Tree:
    def __init__(self, samples, labels, max_depth=-1):
        self.samples = samples
        self.labels = labels
        self.max_depth = 1 #max_depth
        if max_depth == -1: self.max_depth = float('inf')
        self.root_node = self.train_tree(samples, labels)
    def train_tree(self, samples, labels, curr_depth=0):
        if curr_depth < self.max_depth and samples.size > 0:
            best_split = self.get_best_split(samples, labels)

            if best_split['ig'] > 0:
                left_split = self.train_tree(best_split['left_samples'], best_split['left_labels'], curr_depth+1)
                right_split = self.train_tree(best_split['right_samples'], best_split['right_labels'], curr_depth+1)

                return Node(best_split['best_feature'], left_split, right_split, best_split['ig'])
        else:
            return DecisionNode(labels)
    def get_best_split(self, samples, labels):
        num_samples, num_features = np.shape(samples) 
        start_entropy = self.calc_label_entropy(labels)
        best_ig = 0
        best_feature = None
        best_split = {}
        best_split['ig'] = best_ig

        for feature in range(0, num_features):

            left_samples = []
            left_labels = []
            right_samples = []
            right_labels = []

            for sample in range(0, num_samples):
                if samples[sample, feature] == 0:
                    left_samples.append(samples[sample])
                    left_labels.append(labels[sample])
                elif samples[sample, feature] == 1:
                    right_samples.append(samples[sample])
                    right_labels.append(labels[sample])
            left_samples = np.array(left_samples)
            left_labels = np.array(left_labels)
            right_samples = np.array(right_samples)
            right_labels = np.array(right_labels)

            node_entropy = self.calc_node_entropy(num_samples, left_labels, right_labels)
            ig = self.calc_ig(start_entropy, node_entropy)

            if ig > best_ig:
                best_ig = ig
                best_split['best_feature'] = feature
                best_split['ig'] = best_ig
                best_split['left_samples'] = left_samples
                best_split['left_labels'] = left_labels
                best_split['right_samples'] = right_samples
                best_split['right_labels'] = right_labels
        return best_split

    def calc_label_entropy(self, labels):
        entropy = 0
        unique_labels = np.unique(labels)
        for label in unique_labels:
            probability = len(labels[labels==label]) / len(labels)
            entropy -= probability * math.log2(probability)
        return entropy
    def calc_node_entropy(self, num_samples, left_labels, right_labels):
        prob_left = left_labels.size / num_samples
        prob_right = right_labels.size / num_samples
        left_entropy = self.calc_label_entropy(left_labels)
        right_entropy = self.calc_label_entropy(right_labels)
        return prob_left * left_entropy + prob_right * right_entropy
    def calc_ig(self, parent_entropy, child_entropy):
        return parent_entropy - child_entropy
    def predict_label(self, data, tree):
        if tree.is_decision():
            return tree.get_prediction()
        feature_index = tree.feature
        data_feature = data[feature_index]
        if feature_index > data.size: feature_index -= data.size
        pruned_data = self.prune_features(data, feature_index)
        if data_feature == 0:
            return self.predict_label(pruned_data, tree.left_tree)
        else:
            return self.predict_label(pruned_data, tree.right_tree)
    def prune_features(self, data, feature):
        return np.delete(data, feature)

def DT_train_binary(X,Y,max_depth):
    return Tree(X, Y, max_depth)

def DT_make_prediction(x,DT):
    return DT.predict_label(x, DT.root_node)

def DT_test_binary(X,Y,DT):
    correct = []
    for i, test_sample in enumerate(X):
        predicted_val = DT_make_prediction(test_sample, DT)
        correct.append(predicted_val == Y[i])
    return np.sum(correct) / len(correct)


def RF_build_random_forest(X,Y,max_depth,num_of_trees):
    forest = []
    num_samples = len(X)
    index_list = [*range(0,num_samples)]
    for tree in range(num_of_trees):
        random_indecies = np.random.choice(index_list, int(num_samples/10), replace=False)
        samples = []
        labels = []
        for index in random_indecies:
            samples.append(X[index])
            labels.append(Y[index])
        samples = np.array(samples)
        labels = np.array(labels)
        forest.append(DT_train_binary(samples,labels, max_depth=1))
    return forest

def RF_test_random_forest(X,Y,RF):
    correct = 0
    for i, sample in enumerate(X):
        for tree in RF:
            prediction = DT_make_prediction(sample, tree)
        if prediction == Y[i]:
            correct += 1

    return correct / len(X)

def main():
    file_name = "cat_dog_data.csv"
    data = np.genfromtxt(file_name, dtype=str, delimiter=',')
    samples, labels = ds.build_nparray(data)
    # d_tree = DT_train_binary(samples, labels, max_depth=1)
    
    d_tree = DT_train_binary(samples,labels, max_depth=1)
    test_acc = DT_test_binary(samples,labels,d_tree)
    print(test_acc)
    # test_data = [1,1,1,0,0]
    # prediction = DT_make_prediction(test_data, d_tree)
    # print(prediction)

if __name__ == "__main__":
  main()