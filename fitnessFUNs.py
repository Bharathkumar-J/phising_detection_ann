# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:46:20 2019

@author: Ibrahim Aljarah, and Ruba Abu Khurma 
"""
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer
from sklearn import metrics
          
#____________________________________________________________________________________
def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def Ufun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y


def F1(x):
    s = np.sum(x ** 2)
    return s

def F2(x):
    o = sum(abs(x)) + prod(abs(x))
    return o


def F3(x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (np.sum(x[0:i])) ** 2
    return o


def F4(x):
    o = max(abs(x))
    return o


def F5(x):
    dim = len(x)
    o = np.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
    )
    return o


def F6(x):
    o = np.sum(abs((x + 0.5)) ** 2)
    return o


def F7(x):
    dim = len(x)

    w = [i for i in range(len(x))]
    for i in range(0, dim):
        w[i] = i + 1
    o = np.sum(w * (x ** 4)) + np.random.uniform(0, 1)
    return o


def F8(x):
    o = sum(-x * (np.sin(np.sqrt(abs(x)))))
    return o


def F9(x):
    dim = len(x)
    o = np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)) + 10 * dim
    return o


def F10(x):
    dim = len(x)
    o = (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim))
        - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim)
        + 20
        + np.exp(1)
    )
    return o


def F11(x):
    dim = len(x)
    w = [i for i in range(len(x))]
    w = [i + 1 for i in w]
    o = np.sum(x ** 2) / 4000 - prod(np.cos(x / np.sqrt(w))) + 1
    return o


def F12(x):
    dim = len(x)
    o = (math.pi / dim) * (
        10 * ((np.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
        + np.sum(
            (((x[: dim - 1] + 1) / 4) ** 2)
            * (1 + 10 * ((np.sin(math.pi * (1 + (x[1 :] + 1) / 4)))) ** 2)
        )
        + ((x[dim - 1] + 1) / 4) ** 2
    ) + np.sum(Ufun(x, 10, 100, 4))
    return o


def F13(x):
    if x.ndim==1:
        x = x.reshape(1,-1)

    o = 0.1 * (
        (np.sin(3 * np.pi * x[:,0])) ** 2
        + np.sum(
            (x[:,:-1] - 1) ** 2
            * (1 + (np.sin(3 * np.pi * x[:,1:])) ** 2), axis=1
        )
        + ((x[:,-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[:,-1])) ** 2)
    ) + np.sum(Ufun(x, 5, 100, 4))
    return o


def F14(x):
    aS = [
        [
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
        ],
        [
            -32, -32, -32, -32, -32,
            -16, -16, -16, -16, -16,
            0, 0, 0, 0, 0,
            16, 16, 16, 16, 16,
            32, 32, 32, 32, 32,
        ],
    ]
    aS = np.asarray(aS)
    bS = np.zeros(25)
    v = np.matrix(x)
    for i in range(0, 25):
        H = v - aS[:, i]
        bS[i] = np.sum((np.power(H, 6)))
    w = [i for i in range(25)]
    for i in range(0, 24):
        w[i] = i + 1
    o = ((1.0 / 500) + np.sum(1.0 / (w + bS))) ** (-1)
    return o


def F15(L):
    aK = [
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
        0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ]
    bK = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    aK = np.asarray(aK)
    bK = np.asarray(bK)
    bK = 1 / bK
    fit = np.sum(
        (aK - ((L[0] * (bK ** 2 + L[1] * bK)) / (bK ** 2 + L[2] * bK + L[3]))) ** 2
    )
    return fit


def F16(L):
    o = (
        4 * (L[0] ** 2)
        - 2.1 * (L[0] ** 4)
        + (L[0] ** 6) / 3
        + L[0] * L[1]
        - 4 * (L[1] ** 2)
        + 4 * (L[1] ** 4)
    )
    return o


def F17(L):
    o = (
        (L[1] - (L[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * L[0] - 6)
        ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(L[0])
        + 10
    )
    return o


def F18(L):
    o = (
        1
        + (L[0] + L[1] + 1) ** 2
        * (
            19
            - 14 * L[0]
            + 3 * (L[0] ** 2)
            - 14 * L[1]
            + 6 * L[0] * L[1]
            + 3 * L[1] ** 2
        )
    ) * (
        30
        + (2 * L[0] - 3 * L[1]) ** 2
        * (
            18
            - 32 * L[0]
            + 12 * (L[0] ** 2)
            + 48 * L[1]
            - 36 * L[0] * L[1]
            + 27 * (L[1] ** 2)
        )
    )
    return o


def F100(x, weights=(1.0, 1.0, 1.0)):
    
    F1_value = np.sum(x ** 2)
    F2_value = np.sum(np.abs(x)) + np.prod(np.abs(x))
    
    F4_value = np.max(np.abs(x))
    
    combined_value = (
        weights[0] * F1_value +
        weights[1] * F2_value +
        weights[2] * F4_value
    )
    
    return combined_value



#_____________________________________________________________________       
def getFunctionDetails(a):
    
    # [name, lb, ub, dim]
    param = {  0:["F1",-1,1]

            }
    return param.get(a, "nothing")



