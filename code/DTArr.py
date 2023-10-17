import numpy as np 

class DTArr:
    def __init__(self,max_depth=2, min_samples_split=2):

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        # Initialize arrays 
        # Adjust the size according to the tree's depth
        self.features = np.zeros(max_depth, dtype=np.uint8)
        self.threshold = np.zeros(max_depth, dtype=np.float32)
        self.child = np.zeros(2*max_depth, dtype=np.uint16)
        #TODO: check dtype

    def build_tree(self, node, data):
        if node >= self.max_depth:
            return
        
        best_feature = None
        best_threshold = None
        #TODO: get best split

        # Split data into left and right subsets
        left_data = data[data[best_feature] <= best_threshold]
        right_data = data[data[best_feature] > best_threshold]

        # Store feature and threshold in arrays
        self.features[node] = best_feature
        self.threshold[node] = best_threshold

        # Create child nodes
        left_child = 2 * node
        right_child = 2 * node + 1

        self.child[node] = left_child
        self.child[node + 1] = right_child

        # Recursively build the left and right subtrees
        self.build_tree(left_child, left_data)
        self.build_tree(right_child, right_data)
        

    def fit(self,X,Y):
        ''' function to train the tree '''

        print(f"{X.shape} and {Y.shape}")
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(0,dataset)
        

    def predict(self,X):
        ''' function to predict with DT-Arr kernel '''
        
        # Initialize node and feature
        node = 0
        feature = self.features[0]
        
        # Define the δ value
        delta = -float('inf')  
        
        # Main loop to traverse the decision tree
        while self.threshold[node] != delta:
            comparison = X[feature] <= self.threshold[node]
            node = comparison * self.child[2 * node] + (not comparison) * self.child[2 * node + 1]
            feature = self.features[node]

        # Get the class associated with the leaf node
        predicted_class = self.child[2 * node]

        return predicted_class
