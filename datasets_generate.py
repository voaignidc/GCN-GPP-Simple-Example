#!/usr/bin/env python
# coding: utf-8

import csv
import random

import pandas as pd
import numpy as np

class MovementLib7Generate():
    """
    Generate new dataset with 'new_num_nodes' nodes, from MovementLib dataset.
    :path: Path to MovementLib dataset
    :new_num_nodes: Number of nodes in the new dataset.
    :num_nodes: Number of nodes in the origional dataset MovementLib.
    :num_feats: Dimension of node features in the origional dataset MovementLib.
    :num_classes: Number of node classes in the origional dataset MovementLib.
    """
    def __init__(self, path, new_num_nodes=7, num_nodes=360, num_feats=90, num_classes=15):
        self.num_nodes = num_nodes
        self.new_num_nodes = new_num_nodes
        self.data = pd.read_csv(path, header=None, iterator=True)
        self.datas = np.zeros((num_nodes, num_feats+1))
        self.num_per_classes = np.zeros((num_classes, ))
        self.read_all()

    def read_all(self):
        for i in range(self.num_nodes):
            data = self.data.get_chunk(1).values.astype('float').squeeze()
            self.datas[i] = data
            self.datas[i,-1] -= 1
            self.num_per_classes[int(self.datas[i,-1])] += 1
        # print(self.feats)
        # print(self.labels)
        # print(self.num_per_classes)
        
    def select(self):
        """
        Select 'new_num_nodes' nodes, with no more than 2 different classes.
        """
        class1, class2 = np.random.randint(0, high=15, size=2)
        print(class1,class2)
        class2_num = int(np.random.randint(1, high=7, size=1))
        class1_num = int(self.new_num_nodes - class2_num)
        print(class1_num,class2_num)
        
        class1_selected = random.sample(list(range(class1*24, int(class1*24 + self.num_per_classes[class1]))), class1_num)
        class2_selected = random.sample(list(range(class2*24, int(class2*24 + self.num_per_classes[class2]))), class2_num)
        print(class1_selected)
        print(class2_selected)
        return class1_selected, class2_selected
        
    def write(self, new_path):
        class1_selected, class2_selected = self.select()
        with open(new_path, 'w', newline='')as f:
            f_csv = csv.writer(f)
            for i in class1_selected:
                row = self.datas[i]
                row[-1]=0
                f_csv.writerow(row)   
            for i in class2_selected:
                row = self.datas[i]
                row[-1]=1
                f_csv.writerow(row)  

               
if __name__ == "__main__":                
    ml7g = MovementLib7Generate('./input/movement_libras.csv')
    for i in range(100):
        ml7g.write('./input/ML7_' + str(i) + '.csv')

        