'''
author = Jiaye 
'''
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.linalg
import os


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]
    
def LDA(input_x, input_y):
    input_x = np.mat(input_x)
    
    mu= np.mat(np.mean(input_x, axis=0))
    inputSize = input_x.shape
    y_set = np.unique(input_y)
    input_y = np.mat(input_y)
    sub_mu = []
    num_x = []
    Sw = Sb = 0
    for j in y_set:
        j = int(j)
        index = []
        index = [i for i in range(inputSize[0]) if input_y[i] == j]
        num_x= len(index)
        sub_mu = np.mat(np.mean(input_x[index,:], axis=0))
        for i in index:
            Sw += np.dot((input_x[i,:]-sub_mu).T,(input_x[i,:]-sub_mu))
        Sb += np.dot(num_x, np.dot((sub_mu-mu).T,((sub_mu-mu))))
    eig_vals, eig_vecs = np.linalg.eig(
        np.linalg.inv(Sw).dot(Sb))
    arg_index = np.argsort(-eig_vals)
    W = eig_vecs[:,arg_index[:inputSize[1]-2]]
    return W
    
    
if __name__=='__main__':
    path = 'C:\\Users\\mengm\\Desktop\\Jupyter\\MLaPP\\dataset\\iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    print(iris_type)
    x, y = np.split(data, (4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    W = LDA(x_train, y_train)
    y_pre = (np.dot(np.mat(x_test), W)).real
    plt.figure()
    colors = ['red', 'green', 'blue'] 
    for i in range(len(y_test)):
        for j in range(3):
            if y_test[i] == j:
                plt.scatter(y_pre[i, 0].tolist(), y_pre[i, 1].tolist(), marker='o', c=colors[j])  
    plt.show()
    