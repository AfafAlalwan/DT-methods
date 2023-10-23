import numpy as np 

class DTArr:
    def __init__(self,max_depth=2, min_samples_split=5, random_split=True):

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        # Initialize arrays 
        # Adjust the size according to the tree's depth
        self.features = []
        self.threshold = []
        self.child = []
        self.delta = -1
        self.random_split = random_split
        self.node = 0 # increases with child index 

    def build_tree(self,data, curr_depth=0):
 
        X, Y = data[:,:-1], data[:,-1].reshape(-1, 1)
        num_samples, num_features = np.shape(X)
        
        if(num_samples >= self.min_samples_split and curr_depth <= self.max_depth): 

            best_feature, best_threshold, left_data, right_data = self.get_best_split(dataset=data,num_samples=num_samples, num_features=num_features)

            if left_data is not None or right_data is not None: 
                self.features.append(best_feature)
                self.threshold.append(best_threshold)

                if self.node != 0 :
                    left_child = self.node * 2 + 1
                    right_child = left_child + 1 

                else:
                    left_child = 1
                    right_child = 2 

                self.child.append(left_child)
                self.child.append(right_child)
                self.node = self.node + 1

                # Recursively build the left and right subtrees
                
                self.build_tree(left_data, curr_depth=curr_depth+1)
                self.build_tree(right_data,curr_depth=curr_depth+1)
            else:
                self.calculate_leaf_value(Y)

        else:
            self.calculate_leaf_value(Y)
        

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split'''

        best_feature = None
        best_threshold = None
        left_data = None
        right_data = None
        max_info_gain = self.delta

        if self.random_split:
            selected_feature_indices = np.random.choice(num_features, num_samples,replace=True)
        else:
            selected_feature_indices = range(num_features)

        # loop over all the features
        for feature_index in selected_feature_indices:
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    if curr_info_gain>max_info_gain:
                        best_feature = feature_index
                        best_threshold = threshold
                        left_data = dataset_left
                        right_data = dataset_right
                        max_info_gain = curr_info_gain
                # else:
                #     print("meow")
                #     best_feature = feature_index
                #     best_threshold = self.delta
                
            # print(f"node : {self.node}, feature: {best_feature} and threshold {best_threshold} ")

        # print(f"best feature: {best_feature} and best threshold: {best_threshold}" )
        # if best_threshold is None or best_feature is None:
        #     print(f"node : {self.node}, feature: {best_feature} and threshold {best_threshold} ")
            
        #     return self.delta, self.delta, None, None
        
        return best_feature, best_threshold,left_data, right_data


    def split(self, dataset, feature_index, threshold):
            ''' function to split the data '''
            
            dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
            dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
            return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        leaf_value = max(Y, key=Y.count)
        self.features.append(self.delta)
        self.threshold.append(self.delta)
        self.child.append(leaf_value)
        self.child.append(leaf_value)
    
    def fit(self,X,Y):
        ''' function to train the tree '''

        print(f"{X.shape} and {Y.shape}")
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        

    def predict(self,X):
        ''' function to predict with DT-Arr kernel '''
        # print("PRINTING THE TREE")
        # self.print_tree()
        # print("****************")
        preditions = [self.make_prediction(x, self.root) for x in X]
        return np.array(preditions)
    
    def print_tree(self, node = 0):
        if node >= len(self.features):
            return

        left_child = 2 * node 
        right_child = 2 * node + 1    

        print(f"Node: {node}, Feature: {self.features[node]}, Threshold: {self.threshold[node]}, Left Child: {self.child[left_child]}, Right Child: {self.child[right_child]} \n")
        node = node + 1
        self.print_tree(node)

    def make_prediction(self, x, tree):

        # Initialize node and feature
        node = 0
        feature = self.features[node]
        
        # Main loop to traverse the decision tree
        while self.threshold[node] != self.delta:
            if x[feature] <= self.threshold[node]:
                node = self.child[2 * node + 1] # go left 
            else:
                node = 2 if node == 0 else self.child[2 * node] # go right

            if node >= len(self.features):
                break
            else:
                feature = self.features[node]

        # Get the class associated with the leaf node
        predicted_class = self.child[node * 2] #should be the leaf node 

        return predicted_class
