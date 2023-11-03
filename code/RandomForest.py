import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .NaiveTree import NaiveTree
from .DTArr import DTArr
# from sklearn.base import is_classifier

class Forest(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=2, min_samples_split=2,
                 tree_method='naive', **kwargs):
        '''constructor'''
        super().__init__(n_estimators=n_estimators,max_depth=max_depth, min_samples_split=min_samples_split, **kwargs)
        self.tree_method = tree_method
        self.custom_trees = []
    
    def fit(self, x,y, trees=None):
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
            custom_tree = tree_class(max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split )
            if trees is None:
                custom_tree.fit(X=x,Y=y)
            else:
                custom_tree.fit(trees[_].root)
            self.custom_trees.append(custom_tree)


        self.estimators = self.custom_trees
        self.n_estimators = len(self.custom_trees)        

    def predict(self, X):
        # is_binary_classification = is_classifier(self)

        # Initialize an array to store the predictions.
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples,), dtype=np.float32)

        # Iterate through each tree in the ensemble.
        for tree in self.custom_trees:
            tree_prediction = tree.predict(X)
            tree_prediction = tree_prediction.reshape(-1)
            predictions += tree_prediction
            final_predictions = predictions / self.n_estimators
            # final_predictions = final_predictions.reshape(-1,1)
            final_predictions = np.round(final_predictions).astype(int)

        return final_predictions