import numpy as np
from sklearn.ensemble import RandomForestClassifier
from NaiveTree import NaiveTree
import DTArr

class Forest(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=2, min_samples_split=2,
                 tree_method='naive', **kwargs):
        '''constructor'''
        super().__init__(n_estimators=n_estimators, **kwargs)
        # self.custom_trees = custom_trees #not sure?
        self.tree_method = tree_method
        self.custom_trees = []
    
    def fit(self, x,y):

        # Dictionary of Methods 
        method_to_class = {
            'naive': NaiveTree,
            'array': DTArr,
            # Add more methods and corresponding classes as needed
        }

        # Check if self.tree_method is a valid method name
        if self.tree_method in method_to_class:
            tree_class = method_to_class[self.tree_method]
        else:
            raise ValueError("Invalid tree method specified.")

        for _ in range(self.n_estimators):
            custom_tree = tree_class #TODO: write params to customize
            self.custom_trees.append(custom_tree)


        self.estimators = self.custom_trees
        self.n_estimators = len(self.custom_trees)
        # self.fit(x,y)
        

    #TODO: predict method

