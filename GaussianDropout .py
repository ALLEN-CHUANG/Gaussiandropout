
# coding: utf-8

# In[1]:

from keras import backend as K
import numpy as np

def GD(inputs, rate):
    if 0 < rate < 1:
        def noised():
            std = np.sqrt((rate)/(1.0-rate))
            return inputs *  K.random_normal(shape=K.shape(inputs),mean=1.0, stddev= std)
     
        return K.in_train_phase(noised, inputs, training = None)
    return inputs
    









