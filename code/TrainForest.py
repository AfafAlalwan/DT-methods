from RandomForest import Forest

def fitModels(x,y):
    #TODO: split x and y if not splitted
    forest = Forest(tree_method='naive',n_estimators=25,n_jobs=8,max_depth=5)
    testModel(x,y,forest)

def testModel(x,y,model):
    #fitting

    #testing

    #accuracy
    pass

