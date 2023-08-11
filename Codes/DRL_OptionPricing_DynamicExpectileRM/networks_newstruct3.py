# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 14:21:05 2021

@author: laced
"""
import numpy as np
from random import random as rand

from docutils.nodes import line
from keras.models import *
from keras.layers import *
import keras.backend as K
from numpy.random import geometric
import tensorflow as tf
from enumClasses import PerformanceMeasure
import h5py
#import boto3
from random import randint
import time
import math
from keras.optimizers import Adam
from keras import regularizers
import keras


class netWorkBuilder:

    def __init__(self,SR_horizon,minibatchSize,minibatchSize_pg,net_depth=4,initial_param1=0.5,initial_param2 = .05,
                 optimizerType = 'momentum',resnetType = 'const',includemaxpool = False,
                 skipConnection=True,withCorrData=True,smootherValue=0.01,reduce_max=True,seed=0,
                 firstLayer=1024,regularizer_multiplier=1e-6,lastLayer=1024,RNN=False,
                 weightNormalization=True,days_markowitz=30,strike=100,num_stocks=5,window=32,S_0=100,No_wealth=True):
        self.No_wealth = No_wealth
        self.S_0 = S_0
        self.window = window
        self.num_stocks = num_stocks
        self.strike=strike
        self.days_markowitz = days_markowitz
        self.weightNormalization = weightNormalization
        self.RNN = RNN
        self.lastLayer = lastLayer
        self.regularizer_multiplier = regularizer_multiplier
        self.firstLayer = firstLayer
        self.seed = seed
        self.reduce_max = reduce_max
        self.smootherValue = smootherValue
        self.withCorrData = withCorrData
        self.skipConnectionExist = skipConnection
        self.optimizerType = optimizerType
        self.resnetType = resnetType
        self.includemaxpool = includemaxpool
        self.SR_horizon = SR_horizon
        self.net_depth = net_depth
        self.initial_param1 = initial_param1
        self.initial_param2 = initial_param2
        self.minibatchSize = minibatchSize
        self.minibatchSize_pg = minibatchSize_pg
        self.name_counter = 0

    def creat_weight_variable(self,conv_filt_shape,trainable,scope,mean=0,name='',reuse=False, stddev = 0):
        with tf.name_scope(scope):
            with tf.compat.v1.variable_scope(scope +'var_scope', reuse=reuse):
                self.name_counter += 1

                if name == '':
                    name = 'v_' + str(self.name_counter)

                # weight initialization method of Xavier
                if mean == 0:
                    mean = self.initial_param1
                else:
                    mean = mean

                if stddev == 0:
                    if self.initial_param2 >= 1:
                        stddev = np.sqrt(self.initial_param2 / (conv_filt_shape[0]*conv_filt_shape[1]*(conv_filt_shape[2]+conv_filt_shape[3])))
                    else:
                        stddev = self.initial_param2

                # weight initialization method of Kaiming He et al.
                # stddev = np.sqrt(1/(conv_filt_shape[0]*conv_filt_shape[1]*conv_filt_shape[3]))


                if self.seed != 0:
                    seed = self.seed
                else:
                    seed = np.random.randint(0,1e8)



                # tf.compat.v1.glorot_normal_initializer(seed=seed)
                # tf.compat.v1.truncated_normal_initializer(mean=mean, stddev=stddev)
                var = tf.compat.v1.get_variable(shape=conv_filt_shape,
                                                 initializer=tf.compat.v1.truncated_normal_initializer(mean=mean, stddev=stddev,seed=seed),
                                                 regularizer=tf.keras.regularizers.l2(self.regularizer_multiplier),
                                                 trainable=trainable,
                                                 name=name)

                # var =  tf.Variable(tf.random.truncated_normal(conv_filt_shape, mean=mean,
                #                                               stddev=stddev, seed=seed), trainable=trainable)

                return var
            
    def creat_fc_weight_variable(self,conv_filt_shape,trainable,scope,mean=0,name='',reuse=False, stddev = 0):
        with tf.name_scope(scope):
            with tf.compat.v1.variable_scope(scope +'var_scope', reuse=reuse):
                self.name_counter += 1

                if name == '':
                    name = 'v_' + str(self.name_counter)

                # weight initialization method of Xavier
                if mean == 0:
                    mean = self.initial_param1
                else:
                    mean = mean

                if stddev == 0:
                    if self.initial_param2 >= 1:
                        stddev = np.sqrt(self.initial_param2 / (conv_filt_shape[0]*conv_filt_shape[1]))
                    else:
                        stddev = self.initial_param2

                if self.seed != 0:
                    seed = self.seed
                else:
                    seed = np.random.randint(0,1e8)


                var = tf.compat.v1.get_variable(shape=conv_filt_shape,
                                                 initializer=tf.compat.v1.truncated_normal_initializer(mean=mean, stddev=stddev,seed=seed),
                                                 regularizer=tf.keras.regularizers.l2(self.regularizer_multiplier),
                                                 trainable=trainable,
                                                 name=name)

                return var

    def create_new_conv_layer(self,input_data, filter_shape,num_input_channels,
                              num_output_channels,trainable,strides=1,padding='SAME',
                              scope='online_actor',name='',sigTanh=False,
                              linearTrans=False,weight_norm=False,reuse=False,stddev=0):
        with tf.name_scope(scope):
            conv_filt_shape = [filter_shape[0],filter_shape[1], num_input_channels, num_output_channels]

            W = self.creat_weight_variable(conv_filt_shape,trainable,scope,name=name,reuse=reuse,stddev=stddev)
            b = tf.Variable(tf.zeros([num_output_channels]), trainable=trainable,name='bias')

            out_layer = tf.nn.conv2d(input_data, W, [1, strides, 1, 1], padding=padding)
            x = tf.nn.bias_add(out_layer, b)
            if linearTrans == True:
                return x
            elif sigTanh == True:
                return tf.nn.tanh(x)
            else:
                return tf.nn.relu(x)
            
    
    def create_new_fc_layer(self,input_data,num_output_neurons,trainable,scope='online_actor',name='',Tanh=False,
                              linearTrans=False,reuse=False,stddev=0):
        with tf.name_scope(scope):
            num_input_neurons = input_data._shape_as_list()[1]
            shape = [num_input_neurons, num_output_neurons]

            W = self.creat_fc_weight_variable(shape,trainable,scope,name=name,reuse=reuse,stddev=stddev)
            bb = tf.Variable(tf.zeros([num_output_neurons]), trainable=trainable,name='bias')

            x = tf.matmul(input_data, W) + bb
            if linearTrans == True:
                return x
            elif Tanh == True:
                return tf.nn.tanh(x)
            else:
                return tf.nn.relu(x)

    def create_new_conv_layer_dilated(self,input_data, filter_shape,num_input_channels,
                              num_output_channels,trainable,dilation,strides=1,padding='SAME',
                                      sigTanh=False,scope='online_actor',name='h'):
        with tf.name_scope(scope):
            conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                               num_output_channels]
            #mean=0.1
            W = self.creat_weight_variable(conv_filt_shape,trainable,scope)
            b = tf.Variable(tf.zeros([num_output_channels]), trainable=trainable,name='bias')

            # W = tf.Variable(tf.random_uniform(conv_filt_shape, minval=self.initial_param1, maxval=self.initial_param2,
            #                                   dtype=tf.dtypes.float32), trainable=trainable, name=name)
            # b = tf.Variable(tf.zeros([num_output_channels]), trainable=trainable)

            out_layer = tf.nn.atrous_conv2d(input_data,W,dilation,padding=padding)
            x = tf.nn.bias_add(out_layer, b)
            if(sigTanh == True):
                x1 = tf.nn.sigmoid(x)
                x2 = tf.nn.tanh(x)
                x = tf.math.multiply(x1,x2)

                x = self.create_new_conv_layer(x, filter_shape=(1, 1),
                                               num_input_channels=num_output_channels,
                                               num_output_channels=num_output_channels,
                                               trainable=trainable, padding='VALID')
            else:
                x = tf.nn.relu(x)
                # x = tf.nn.leaky_relu(x, alpha=0.1)
                # return x
            return x

    def NN_net_pg_conv(self, inputTensor, trainable, scope):
        with tf.name_scope(scope):

            mIn = tf.compat.v1.placeholder(tf.float32,
                                           [None, inputTensor.shape[1], self.num_stocks, inputTensor.shape[3]-1])
            mIn_exp = mIn
            V_t = tf.multiply(mIn,0)[:,0:1,:,0:1]

            var1 = self.creat_weight_variable([5, 1, inputTensor.shape[3]-1, self.net_depth],
                                       trainable, scope, name='aloc_var1', reuse=False)
            var2 = self.creat_weight_variable([3, 1, self.net_depth, self.net_depth],
                                       trainable, scope, name='aloc_var2', reuse=False)
            var3 = self.creat_weight_variable([self.window - 4, 1,  self.net_depth, self.net_depth],
                                       trainable, scope, name='aloc_var3', reuse=False)
            var4 = self.creat_weight_variable([1, 1,  self.net_depth+1, 1],
                                       trainable, scope, name='aloc_var4', reuse=False)
            for i in range(inputTensor.shape[1] - self.window):
                input_t = mIn_exp[:,i:i+self.window,:,:]
                x = self.create_new_conv_layer(input_t, filter_shape=(5, 1),
                                                         num_input_channels=inputTensor.shape[3]-1,
                                                         num_output_channels=self.net_depth, trainable=trainable,
                                                         sigTanh=False, padding='VALID', name='aloc_var1',
                                                         scope=scope, weight_norm=False, reuse=True)

                x = self.create_new_conv_layer(x, filter_shape=(self.window - 4, 1),
                                               num_input_channels=self.net_depth,
                                               num_output_channels=self.net_depth, trainable=trainable,
                                               sigTanh=False, padding='VALID', name='aloc_var3',
                                               scope=scope, weight_norm=False, reuse=True, linearTrans=False)
                x = tf.concat([x,V_t],axis=3)
                
                x = self.create_new_conv_layer(x, filter_shape=(1, 1),
                                               num_input_channels=self.net_depth+1,
                                               num_output_channels=1, trainable=trainable,
                                               sigTanh=True, padding='VALID', name='aloc_var4',
                                               scope=scope, weight_norm=False, reuse=True, linearTrans=False)

                V_t_pre = V_t
                if (i == 0):
                    deltas = x
                else:
                    deltas = tf.concat([deltas, x], axis=1)
                V_t = V_t_pre + tf.expand_dims(tf.reduce_sum(tf.squeeze(x, 3) * (tf.exp(tf.expand_dims(mIn_exp[:, i + self.window, :, 0], 1)) - tf.exp(
                    tf.expand_dims(mIn_exp[:, i + self.window - 1, :, 0], 1))) * np.reshape(self.S_0,[1,1,self.num_stocks]),axis=2,keepdims=True),axis=3)
                
                # V_t = V_t_pre + tf.expand_dims(tf.reduce_sum(tf.squeeze(x, 3) * (tf.exp(tf.expand_dims(mIn_exp[:, i + self.window, :, 0], 1)) - tf.exp(
                #     tf.expand_dims(mIn_exp[:, i + self.window - 1, :, 0], 1))) * self.strike,axis=2,keepdims=True),axis=3)

            return mIn, V_t[:,0,0], tf.squeeze(deltas,3)
        
    
    def NN_net_pg_fc(self, inputTensor, trainable, scope):
        with tf.name_scope(scope):

            mIn = tf.compat.v1.placeholder(tf.float32,[None, inputTensor.shape[1], self.num_stocks + 1])      
            V_t = tf.multiply(mIn[:,0:1,0],0)
            rr = tf.multiply(mIn[:,0,0:-1],0)
            
            # self.creat_fc_weight_variable([inputTensor.shape[2],self.net_depth],trainable,scope,name='aloc_var1',reuse=False)
            self.creat_fc_weight_variable([inputTensor.shape[2] + self.num_stocks - 1,self.net_depth],trainable,scope,name='aloc_var1',reuse=False)
            self.creat_fc_weight_variable([self.net_depth,self.net_depth],trainable,scope,name='aloc_var2',reuse=False)
            self.creat_fc_weight_variable([self.net_depth,self.num_stocks],trainable,scope,name='aloc_var3',reuse=False)
            
            for i in range(inputTensor.shape[1] - 1):
                # input_t = tf.concat([mIn[:,i,:],V_t],axis=1)
                input_t = tf.concat([mIn[:,i,:],rr],axis=1)
                
                x = self.create_new_fc_layer(input_t, self.net_depth, trainable,scope=scope,name='aloc_var1',Tanh=False,reuse=True)
            
                x = self.create_new_fc_layer(x, self.net_depth, trainable,scope=scope,name='aloc_var2',Tanh=False,reuse=True)
                
                x = self.create_new_fc_layer(x, self.num_stocks, trainable,scope=scope,name='aloc_var3',Tanh=True,reuse=True)
                
                rr = x * 0
                
                if (i == 0):
                    deltas = tf.expand_dims(x,1)
                else:
                    deltas = tf.concat([deltas, tf.expand_dims(x,1)], axis=1)
                    
                V_t = V_t + tf.reduce_sum(x * (tf.exp(mIn[:, i + 1, 0:-1]) - tf.exp(mIn[:, i, 0:-1])) * np.reshape(self.S_0,[1,self.num_stocks]),axis=1,keepdims=True)

            return mIn, V_t, deltas    


    def NN_net_ddpg_actor(self, inputTensor, trainable, scope):
        with tf.name_scope(scope):
                
            mIn = tf.compat.v1.placeholder(tf.float32,[None, self.num_stocks + 1])
            
            x = mIn
            
            x = self.create_new_fc_layer(x, self.net_depth, trainable,scope=scope,name='',Tanh=False)
            
            x = self.create_new_fc_layer(x, self.net_depth, trainable,scope=scope,name='',Tanh=False)
            
            x = self.create_new_fc_layer(x, self.num_stocks, trainable,scope=scope,name='',Tanh=True)  
            
            return mIn, x

    def NN_net_ddpg_critic(self, inputTensor, trainable, scope):
        with tf.name_scope(scope):
            
            mIn = tf.compat.v1.placeholder(tf.float32,[None, self.num_stocks + 1])
            
            x = mIn
            
            x = self.create_new_fc_layer(x, self.net_depth, trainable,scope=scope,name='',Tanh=False)
            
            x = self.create_new_fc_layer(x, self.net_depth, trainable,scope=scope,name='',Tanh=False)
            
            x = self.create_new_fc_layer(x, self.net_depth, trainable,scope=scope,name='',Tanh=False)  
            
            return mIn, x

