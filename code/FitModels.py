from .RandomForest import Forest
import timeit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time
def FitModels(x_train,y_train,x_test=None,y_test=None,createTest=False):
    print(f"{type(x_train)} and {x_train.shape}")
    if x_test is None or y_test is None:
        x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,train_size=0.2)
        createTest = True

    if createTest:
        #TODO write csv
        pass

    # forest = Forest(tree_method='naive',n_estimators=10,n_jobs=8,max_depth=5)
    testModel(x_train,y_train,x_test,y_test,
              Forest(tree_method='naive',n_estimators=5,n_jobs=8,max_depth=5), name="RF")
    time.sleep(1)
    testModel(x_train,y_train,x_test,y_test,
              Forest(tree_method='naive',n_estimators=10,n_jobs=8,max_depth=3), name="RF2")
    time.sleep(1)
    testModel(x_train,y_train,x_test,y_test,
              Forest(tree_method='naive',n_estimators=15,n_jobs=8,max_depth=5), name="RF3")
    # testModel(x_train,y_train,x_test,y_test,
    #           Forest(tree_method='naive',n_estimators=20,n_jobs=8,max_depth=5), name="RF3")
    # testModel(x_train,y_train,x_test,y_test,
    #           Forest(tree_method='naive',n_estimators=25,n_jobs=8,max_depth=5), name="RF4")
    # testModel(x_train,y_train,x_test,y_test,
    #           Forest(tree_method='naive',n_estimators=30,n_jobs=8,max_depth=5), name="RF5")

def testModel(x_train,y_train,x_test,y_test,model,name):
    #fitting
    print(f"Fitting model {name}")
    model.fit(x_train,y_train) #train

    #testing
    print(f"Testing model {name}")
    start = timeit.default_timer()
    y_pred = model.predict(x_test)
    end = timeit.default_timer()

    print("Total time: " + str(end - start) + " ms")
    print("Throughput: " + str(len(x_test) / (float(end - start)*1000)) + " #elem/ms")

    #accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    confusion = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix: \n {confusion}")



    