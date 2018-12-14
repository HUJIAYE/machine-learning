'''
author = Jiaye
'''

import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

#将原始数据标签做成字典
def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

#读取数据
def load_data(path):
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    x, y = np.split(data, (4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    return x_train, x_test, y_train, y_test

#定义激活函数
def sigmoid(x):
    return (math.exp(-x)+1)**(-1)
    
def g(x, y):
    return y(1-y)(x-y)

def BP(data, label, hidden_number, step, model = 'normal'):
    output_number = label.shape[1]
    input_number = data.shape[1]
    # 初始化参数  
    hidden_threshold = np.random.rand(data.shape[0], hidden_number)
    output_threshold = np.random.rand(data.shape[0], output_number)
    input_hidden_para = np.random.rand(data.shape[0], hidden_number, input_number)
    hidden_output_para = np.random.rand(data.shape[0], output_number, hidden_number)
    #训练网络：
    if model == 'normal':
        alpha = np.zeros((data.shape[0], hidden_number))
        error = np.zeros((data.shape[0], 1))
        beta = np.zeros((data.shape[0], output_number))
        for i in range(data.shape[0]):
            # input-hidden layer parameters
            for j in range(hidden_number):
                hidden_x = np.sum(data[i] * input_hidden_para[i, j].T) - hidden_threshold[i, j]
                alpha[i,j] = sigmoid(hidden_x)
            # hidden-output layer parameters
            for m in range(output_number):
                output_x = np.sum(alpha[i] * hidden_output_para[i, m].T) - output_threshold[i, m]
                beta[i, m] = sigmoid(output_x)
            error[i] = 0.5 * np.sum((beta[i] - label)**2)
            #更新参数
            for j in range():
                delta_w[i, j] = step * g(label[i], beta[i][j]) * alpha[i, j]
            
            accumulated_error = np.sum(error)
            print(error)
            
            
    
   
if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data('C:\\Users\\Jiaye\\Desktop\\machine-learning\\MLaPP\\dataset\\iris.data')
   #x_train.shape, x_test.shape, y_train.shape, y_test.shape： (90, 4) (60, 4) (90, 1) (60, 1)
    BP(x_train, y_train, 6, 0.01)
