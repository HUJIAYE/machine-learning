'''
author = Jiaye Hu
'''
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

def load_data(path):
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    x, y = np.split(data, (4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    return x_train, x_test, y_train, y_test
    
#计算信息熵
def inf_ent(input):
    input = [float(i) for i in input]
    input_class = set(input)
    A=[]
    B=[]
    for i in range(len(input_class)):
        B.append([j for j in range(len(input)) if input[j] == int(i)])
    prob = []
    for i in B:
        prob.append(len(i)/len(input))
    inf_ent = 0
    for i in prob:
        if i>0:
            inf_ent -= (i*math.log(i, 2))
    return inf_ent
    
    
# 计算信息增益
def inf_gain(input_Dv, input_D):
    input_Dv = [float(i) for i in input_Dv]
    input = [float(i) for i in input_D]
    j = 0
    for i in np.argsort(input_Dv):
        input_D[j] = input[i]
        j = j+1
    input_Dv = sorted(input_Dv)
    item_len = len(input_D)
    gain_array = []
    for i in range(item_len-1):
        true_prob = inf_ent(input_D[:i+1])
        false_prob = inf_ent(input_D[i+1:])
        ans = inf_ent(input_D)-((i+1)/item_len*true_prob + (item_len-i-1)/item_len*false_prob)
        gain_array.append(ans)
    max_value = max(gain_array)
    max_num = gain_array.index(max_value)
    seg_value = ((input_Dv[max_num]+input_Dv[max_num+1])/2)
    return max_value, seg_value
        

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data('C:\\Users\\mengm\\Desktop\\Jupyter\\MLaPP\\dataset\\iris.data')
    print(inf_gain(x_train[:,1], y_train))