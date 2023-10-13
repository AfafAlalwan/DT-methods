#!/usr/bin/env python3

import sys
import numpy as np

# sys.path.ap pend('/home/afaf/Desktop/Github Projects/DT-methods')  # Replace with the root directory of your project
# sys.path.append("../../")
from FitModels import FitModels

def readFile(path):
    f = open(path, 'r')
    X = []
    Y = []
    for row in f:
        entries = row.strip().split(",")
        y = int(entries[-1])  # Store the last column as Y
        x = [int(e) for e in entries[:-1]]  # Store the rest of the columns as X

        X.append(x)
        Y.append([y])

    return np.array(X).astype(dtype=np.int32), np.array(Y).astype(dtype=np.int32)

def main(argv):
	X,Y = readFile("/home/afaf/Desktop/Github Projects/DT-methods/data/adult/adult.data")

	FitModels(X,Y)

if __name__ == "__main__":
   main(sys.argv[1:])