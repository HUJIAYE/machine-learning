'''
author = Jiaye
'''
#coding=utf-8
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#将原始数据标签做成字典
def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 0.5, b'Iris-virginica': 1}
    return it[s]

#读取数据
def load_data(path):
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    x, y = np.split(data, (4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    return x_train, x_test, y_train, y_test

#定义激活函数
def sigmoid(x):
    x = float(x)
    try:
        ans = math.exp(-x)+1
    except OverflowError:
        ans = float('inf')
    ans_sig = 1/ans
    return ans_sig
    
def g_f(y_truth, y_pre):
    return y_pre*(1-y_pre)*(y_truth-y_pre)


    
def BP(data, label, hidden_number, step, model = 'normal'):
    
    label = [float(i) for i in label]
    output_number = len(label)
    input_number = data.shape[1]
    # 初始化参数  
    gama = np.random.rand(data.shape[0], hidden_number)
    theta = np.random.rand(data.shape[0], output_number)
    v = np.random.rand(data.shape[0], hidden_number, input_number)
    w = np.random.rand(data.shape[0], output_number, hidden_number)
    accumulated_error = []
    instan_error = 1
    max_epoch = 1000
    #训练网络：
    while instan_error >= 0.5 and max_epoch > 0:
        alpha = np.zeros((data.shape[0], hidden_number))
        error = np.zeros((data.shape[0], 1))
        beta = np.zeros((data.shape[0], output_number))
        delta_w = np.zeros((data.shape[0], output_number, hidden_number))
        delta_theta = np.zeros((data.shape[0], output_number))
        delta_gama = np.zeros((data.shape[0], hidden_number))
        delta_v = np.zeros((data.shape[0], hidden_number, input_number))
        for i in range(data.shape[0]):
            # input-hidden layer parameters
            
            for j in range(hidden_number):
                hidden_x = 0
                for k in range(input_number):
                    hidden_x += data[i,k] * v[i, j, k]  
                hidden_x = hidden_x - gama[i, j]
                alpha[i,j] = sigmoid(hidden_x)
            # hidden-output layer parameters
            for m in range(output_number):
                output_x = 0
                for k in range(hidden_number):
                    output_x += alpha[i,k] * w[i, m, k]
                output_x = output_x - theta[i, m]
                beta[i, m] = sigmoid(output_x)
                
            error[i] = 0.5 * np.sum((beta[i] - label)**2)
            #标准梯度
            e = np.zeros((hidden_number, 1))
            for h in range(hidden_number):
                w_g = 0
                for j in range(output_number):
                    w_g += g_f(label[i], beta[i, j]) * w[i, j, h]
                w_g = w_g * alpha[i,h]*(1-alpha[i,h])
                e[h] = w_g
            for j in range(output_number):
                for h in range(hidden_number):
                    delta_w[i, j, h] = step * g_f(label[i], beta[i][j]) * alpha[i, h]
                    delta_theta[i, j] = - step * g_f(label[i], beta[i][j])
                    delta_gama[i, h] = - step * e[h]
                    for k in range(input_number):
                        delta_v[i, h, k] = step * e[h] * data[i, k]  
            #更新参数
            gama = gama + delta_gama
            theta = theta + delta_theta
            v = v + delta_v
            w = w + delta_w
            #计算平均误差
            instan_error = np.mean(error)
            
        max_epoch -= 1
        
        if max_epoch % 10 == 0:
            accumulated_error.append(instan_error)
            print(instan_error)
    plt.plot(accumulated_error)
    plt.show()
            
    
   
if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data('C:\\Users\\Jiaye\\Desktop\\machine-learning\\MLaPP\\dataset\\iris.data')
   #x_train.shape, x_test.shape, y_train.shape, y_test.shape： (90, 4) (60, 4) (90, 1) (60, 1)
    BP(x_train, y_train, 6, 0.6)
