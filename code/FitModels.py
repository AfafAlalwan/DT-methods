from .RandomForest import Forest
import timeit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np 

def FitModels(x_train,y_train,x_test=None,y_test=None,createTest=False):
    print(f"{type(x_train)} and {x_train.shape}")
    if x_test is None or y_test is None:
        x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,train_size=0.2)
        # createTest = True

    if createTest:#TODO: specify where to save these csv files
        train_output = "train.csv"
        test_output = "test.csv"
        train_data = np.column_stack((x_train, y_train))
        np.savetxt(train_output, train_data, delimiter=',')

        test_data = np.column_stack((x_test, y_test))
        np.savetxt(test_output, test_data, delimiter=',')

        print("Data saved to CSV files.")

    #TODO: decide how many forests and choose the best accuracy one to generate it in C
    testModel(x_train,y_train,x_test,y_test,
              Forest(tree_method='array',n_estimators=10,n_jobs=8,max_depth=5, min_samples_split=20), name="RF array")
    testModel(x_train,y_train,x_test,y_test,
              Forest(tree_method='naive',n_estimators=10,n_jobs=8,max_depth=5), name="RF naive")

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

    output = name + ".csv" #TODO: specify where to save it 
    np.savetxt(output, y_pred, delimiter=",", fmt="%d")
    #accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    confusion = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix: \n {confusion}")



    