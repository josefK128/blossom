from numpy import genfromtxt
import numpy as np

mydata = genfromtxt('test.csv', delimiter=',')
print(mydata)
print(type(mydata))
