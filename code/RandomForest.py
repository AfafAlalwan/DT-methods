import numpy as np
from numpy import ndarray
# from sklearn._typing import ArrayLike, MatrixLike
from sklearn.ensemble import RandomForestClassifier
from code.NaiveTree import NaiveTree
from code.DTArr import DTArr
from sklearn.base import is_classifier

class Forest(RandomForestClassifier):
    def __init__(self, n_estimators=100, maxDepth=2, min_samples_split=2,
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
            print(f"max depth {self.maxDepth} and min samples = {self.min_samples_split} ")
            custom_tree = tree_class(max_depth=self.maxDepth, min_samples_split=self.min_samples_split ) #TODO: write params to customize
            # print(f"{x.shape} and {y.shape}")
            custom_tree.fit(X=x,Y=y)
            self.custom_trees.append(custom_tree)


        self.estimators = self.custom_trees
        self.n_estimators = len(self.custom_trees)
        # self.fit(x,y)
        

    def predict(self, X):
        # Check if this is a binary classification problem.
        # If it's binary classification, each tree should output class probabilities.
        is_binary_classification = is_classifier(self)

        # Initialize an array to store the predictions.
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples,), dtype=np.float32)

        # if is_binary_classification:
        #     # For binary classification, we'll store class probabilities.
        #     n_classes = 2
        #     predictions = np.zeros((n_samples, n_classes), dtype=np.float32)
        # else:
        #     # For multi-class classification or regression, we'll store the predicted values.
        #     predictions = np.zeros((n_samples,), dtype=np.float32)

        # Iterate through each tree in the ensemble.
        for tree in self.custom_trees:
            # Here, you can call your custom predict method for each tree.
            tree_prediction = tree.predict(X)

            # print("Shape of tree_prediction:", tree_prediction.shape)
            # print(tree_prediction)
            # print(is_binary_classification)    
            is_binary_classification = False

            predictions += tree_prediction
            final_predictions = predictions / self.n_estimators
            final_predictions = final_predictions.reshape(-1,1)
            final_predictions = np.round(final_predictions).astype(int)

        return final_predictions