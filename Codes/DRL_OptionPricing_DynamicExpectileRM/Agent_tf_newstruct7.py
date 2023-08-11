# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:09:15 2021

@author: laced
"""
import numpy as np
import tensorflow as tf
from enumClasses import PerformanceMeasure
from enumClasses import executionMode
import math
from networks_newstruct3 import netWorkBuilder
import pandas as pd
from io import StringIO
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import time
import datetime
from scipy.optimize import minimize, root
from scipy.special import softmax


class Agent_tf:
    def __init__(self,learningRate,learningRate_pg, minibatchCount, minibatchSize,minibatchSize_pg,
                 epochs,window,SR_horizon,
                 saved_models_path,x,pMeasure=PerformanceMeasure.Expectile,
                 markowitzCoef=0.5,modelType='ddpg',net_depth=4,decayRate=0.99,
                 initial_param1=0.5,initial_param2 = 0.005,tradeFee = .001,
                 optimizerType = 'momentum',resnetType = 'const',includemaxpool = False,
                 minimmumRate=1e-6,tradeFeeMultiplier=1,constrain_action=False,
                 riskMultiplier=1,skipConnection=True,excessReturn=False,withCorrData=True,
                 smootherValue=.01,maxStockWeight=.35,runMode=1,reduce_max=True,seed=0,
                 firstLayer=1024,regularizer_multiplier=1e-6,lastLayer=1024,RNN=False,
                 keep_prob=.8,weightNormalization=True,days_markowitz=30,AI_model='iie',
                 entropy_coef=1e-3,strike=100,expectile_value=0.95,S_0=100,
                 mu=0,sigma=0,T=1,n_timesteps=10,num_stocks=5,loadModelFromSavedModels=False):
        self.save_path = 'D:\SAEED\EqualRisk/results/'
        
        self.DP_desc = 1000
        self.num_exp = 1
        self.No_wealth = True
        self.loadModelFromSavedModels = loadModelFromSavedModels
        self.num_stocks = num_stocks
        self.S_0 = S_0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.n_timesteps = n_timesteps
        self.expectile_value = expectile_value
        self.strike = strike
        self.entropy_coef = entropy_coef
        self.AI_model = AI_model
        self.days_markowitz=days_markowitz
        self.keep_prob_value=keep_prob
        self.RNN = RNN
        self.regularizer_multiplier = regularizer_multiplier
        self.seed = seed
        self.reduce_max = reduce_max
        self.runMode = runMode
        self.maxStockWeight = maxStockWeight
        self.initial_param1 = initial_param1
        self.initial_param2 = initial_param2
        self.resnetType = resnetType
        self.includemaxpool = includemaxpool
        self.withCorrData = withCorrData
        self.excessReturn = excessReturn
        self.skipConnection = skipConnection
        self.riskMultiplier = riskMultiplier
        self.constrain_action = constrain_action
        self.tradeFeeMultiplier = tradeFeeMultiplier
        self.optimizerType = optimizerType
        self.minimmumRate = minimmumRate
        self.decayRate = decayRate
        self.net_depth = net_depth
        self.markowitzCoef = markowitzCoef
        self.bestInSampleOutput = 0
        self.SR_horizon=SR_horizon
        self.minibatchSize = minibatchSize
        self.minibatchSize_pg = minibatchSize_pg
        self.netWorkBuilder = netWorkBuilder(self.SR_horizon,
                                             self.minibatchSize,
                                             self.minibatchSize_pg,
                                             net_depth,
                                             initial_param1,
                                             initial_param2,
                                             optimizerType,
                                             resnetType,
                                             includemaxpool,
                                             skipConnection,
                                             withCorrData,
                                             smootherValue,
                                             reduce_max,
                                             self.seed,
                                             firstLayer,
                                             regularizer_multiplier,
                                             lastLayer,
                                             RNN,
                                             weightNormalization,
                                             days_markowitz,strike,
                                             num_stocks,window,S_0,self.No_wealth)
        self.window=window
        self.pBeta = 0.0005
        self.k = 15
        self.pMeasure=pMeasure
        self.learningRate = np.array([learningRate])
        self.learningRate_pg = learningRate_pg
        self.minibatchCount = minibatchCount
        self.epochs = epochs
        self.tradeFee = tradeFee
        self.tradePer = 1.0 - self.tradeFee
        self.APV=np.array([1])
        # self.weightsMatrix = np.array([self.weights])
        self.gamma = 1
        self.tau_c = .1
        self.tau_a = .1
        self.modelType=modelType
        self.evalScore=[]
        self.lastValueInEpoch=0
        self.saved_models_path=saved_models_path
        self.bestOutOfSample = -100
        
        if (modelType == 'approximate'):
            return


        # initialize the buffer
        # self.initPvm(y)

        # initialize actors and critics
        self.actor_scopes = ['online_actor', 'target_actor','online_actor_pg']
        self.critic_scopes = ['online_critic', 'target_critic']
        self.critic_scopes_Test = ['online_critic_Test', 'target_critic_Test']
        self.critic_scopes_Test_pg = ['online_critic_Test_pg', 'target_critic_Test_pg']
        self.critic_scopes_Train = ['online_critic_Train', 'target_critic_Train']

        tf.compat.v1.reset_default_graph()

        if self.seed != 0:
            tf.compat.v1.random.set_random_seed(self.seed)

        
        if (modelType == 'ddpg'):
            self.onlineActor(x, self.actor_scopes[0])
            self.onlineCritic(x, self.critic_scopes[0])
            self.targetCritic(x, self.critic_scopes[1])
            self.targetActor(x, self.actor_scopes[1])
            self.onlineActor(x, self.actor_scopes[2])
        else:
            self.onlineActor(x, self.actor_scopes[2])
            
            
        self.onlineCritic_Test(x, self.critic_scopes_Test[0])
        self.targetCritic_Test(x, self.critic_scopes_Test[1])
        
        self.onlineCritic_Test_pg(x, self.critic_scopes_Test_pg[0])
        self.targetCritic_Test_pg(x, self.critic_scopes_Test_pg[1])
        
        self.onlineCritic_Train(x, self.critic_scopes_Train[0])
        self.targetCritic_Train(x, self.critic_scopes_Train[1])
        
        
        if (modelType == 'ddpg'):
            self.targetCritic_Test_match()
            self.onlineCritic_Test_match()
            self.targetCritic_Test_pg_match()
            self.onlineCritic_Test_pg_match()


        self.saver = tf.compat.v1.train.Saver(max_to_keep=4)
        # setup the initialisation operator
        self.init_op = tf.compat.v1.global_variables_initializer()
        
        
        

    def critic(self, inputTensor, trainable,scope):
        with tf.name_scope(scope):
            mIn_critic, x = self.netWorkBuilder.NN_net_ddpg_critic(inputTensor, trainable,scope)
            actionIn = tf.placeholder(tf.float32,[None,self.num_stocks])
            x = tf.concat([x,actionIn],1)
            
            x = self.netWorkBuilder.create_new_fc_layer(x, self.net_depth, trainable,scope=scope,name='',Tanh=False)
            
            x = self.netWorkBuilder.create_new_fc_layer(x, self.net_depth, trainable,scope=scope,name='',Tanh=False)
            
            x = self.netWorkBuilder.create_new_fc_layer(x, 1, trainable,scope=scope,name='',linearTrans=True)

            return mIn_critic,actionIn,x

    def actor(self, inputTensor,trainable,scope):
        if scope==self.actor_scopes[2] and self.window > 1:
            mIn, V_t, deltas = self.netWorkBuilder.NN_net_pg_conv(inputTensor, trainable,scope)
            return mIn, V_t, deltas
        elif scope==self.actor_scopes[2] and self.window == 1:
            mIn, V_t, deltas = self.netWorkBuilder.NN_net_pg_fc(inputTensor, trainable,scope)
            return mIn, V_t, deltas
        else:
            mIn, x = self.netWorkBuilder.NN_net_ddpg_actor(inputTensor, trainable, scope)
            return mIn, x

    def onlineActor(self, inputTensor,scope):
        with tf.name_scope(scope):
            if(scope==self.actor_scopes[0]):
                self.mIn, self.actorOut = self.actor(inputTensor, True, scope)
                #assuming that we know the weights of the NN of online Q, we want to update online Actor
                self.critic_action_grad = tf.placeholder(tf.float32, shape=(None, self.num_stocks))
                self.online_actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.actor_scopes[0])
                self.actor_grad = tf.gradients(self.actorOut, self.online_actor_params, self.critic_action_grad)
                self.global_step = tf.Variable(0, trainable=False)
                self.lr = tf.compat.v1.placeholder(tf.float32)
                self.actorOptimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(
                    zip(self.actor_grad, self.online_actor_params), global_step=self.global_step)

            elif(scope==self.actor_scopes[2]):
                self.mIn_pg, V_t, deltas = self.actor(inputTensor, True, scope)
                self.deltas = deltas
                self.short = tf.compat.v1.placeholder(tf.float32)
                self.reward = self.short * tf.maximum(tf.reduce_mean(tf.exp(self.mIn_pg[:,-1,0:-1])*np.reshape(self.S_0,[1,self.num_stocks]),axis=1,keepdims=True) - self.strike,0)[:,0] - V_t[:,0]
                # self.reward = self.short * tf.maximum(tf.reduce_mean(tf.exp(self.mIn_pg[:,-1,:,0])*self.strike,axis=1,keepdims=True) - self.strike,0)[:,0] - V_t[:,0]
                if self.pMeasure == PerformanceMeasure.MSE:
                    self.loss = tf.reduce_mean(self.reward)
                if self.pMeasure == PerformanceMeasure.CVaR:
                    self.loss = tf.reduce_mean(tf.sort(self.reward)[
                                        tf.cast(self.expectile_value * self.minibatchSize_pg, tf.int32):])
                if self.pMeasure == PerformanceMeasure.Expectile:
                    q = tf.Variable(tf.random.truncated_normal([1], mean=0,stddev=.05), trainable=True, name='exp_q')
                    self.loss = tf.reduce_mean(self.expectile_value * tf.maximum(self.reward - q, 0.0) ** 2 +
                                            (1-self.expectile_value) * tf.maximum(q - self.reward, 0.0) ** 2, axis=0)
                    
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                self.lr_pg = tf.compat.v1.placeholder(tf.float32)
                optim_adam = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr_pg,beta1=0.9, beta2=0.999)
                # gvs = optim_adam.compute_gradients(self.loss)
                # self.actorOptimiser_adam = optim_adam.apply_gradients(gvs, global_step=self.global_step)
                self.actorOptimiser_adam = optim_adam.minimize(self.loss, global_step=self.global_step)

    def onlineCritic(self, inputTensor, scope):
        with tf.name_scope(scope):
            self.mIn_critic,self.actionIn,self.criticOut =self.critic(inputTensor, True,scope)
            self.value_target=tf.placeholder(tf.float32,shape=(None,1))
            # self.cum_density = tf.linspace(0.01, 0.99, self.num_exp)
            # self.cum_density  = np.linspace(0.01, 0.99, self.num_exp)
            
            if self.pMeasure == PerformanceMeasure.MSE:
                loss = tf.reduce_mean(tf.square(self.value_target[:,0] - self.criticOut[:,0]),axis=0)
            if self.pMeasure == PerformanceMeasure.Expectile:
                loss = tf.reduce_mean(self.expectile_value * tf.maximum(self.value_target[:,0] - self.criticOut[:,0], 0.0) ** 2 +
                                            (1-self.expectile_value) * tf.maximum(self.criticOut[:,0] - self.value_target[:,0], 0.0) ** 2, axis=0)
            # if self.pMeasure == PerformanceMeasure.Expectile:
            #     loss = tf.reduce_mean(tf.reduce_sum(self.cum_density * tf.maximum(self.value_target - self.criticOut, 0.0) ** 2 +
            #                                 (1-self.cum_density) * tf.maximum(self.criticOut - self.value_target, 0.0) ** 2,axis=1), axis=0)
            
            if self.pMeasure == PerformanceMeasure.Quantile:
                loss = tf.reduce_mean((self.expectile_value) * tf.maximum(self.value_target[:,0] - self.criticOut[:,0], 0) +
                                  (1-self.expectile_value) * tf.maximum(self.criticOut[:,0] - self.value_target[:,0], 0), axis=0)
            

            # add an optimiser
            self.global_step_critic = tf.Variable(0, trainable=False)
            self.lr_critic = tf.compat.v1.placeholder(tf.float32)
            self.criticOptimiser = tf.train.AdamOptimizer(learning_rate=self.lr_critic)\
                .minimize(loss,global_step=self.global_step_critic)
            self.action_grad = tf.gradients(self.criticOut, self.actionIn)
            # self.loss_grad_Q = tf.gradients(loss, self.criticOut)
    
    def onlineCritic_Test(self, inputTensor, scope):
        with tf.name_scope(scope):
            self.mIn_critic_Test,self.actionIn_Test,self.criticOut_Test =self.critic(inputTensor, True,scope)
            self.value_target_Test=tf.placeholder(tf.float32,shape=(None,1))
            
            if self.pMeasure == PerformanceMeasure.MSE:
                loss = tf.reduce_mean(tf.square(self.value_target_Test[:,0] - self.criticOut_Test[:,0]),axis=0)
            if self.pMeasure == PerformanceMeasure.Expectile:
                loss = tf.reduce_mean(self.expectile_value * tf.maximum(self.value_target_Test[:,0] - self.criticOut_Test[:,0], 0.0) ** 2 +
                                            (1-self.expectile_value) * tf.maximum(self.criticOut_Test[:,0] - self.value_target_Test[:,0], 0.0) ** 2, axis=0)
                
            if self.pMeasure == PerformanceMeasure.CVaR:
                loss = tf.reduce_mean(self.expectile_value * tf.maximum(self.value_target_Test[:,0] - self.criticOut_Test[:,0], 0.0) ** 2 +
                                            (1-self.expectile_value) * tf.maximum(self.criticOut_Test[:,0] - self.value_target_Test[:,0], 0.0) ** 2, axis=0)
            
            if self.pMeasure == PerformanceMeasure.Quantile:
                loss = tf.reduce_mean((self.expectile_value) * tf.maximum(self.value_target_Test[:,0] - self.criticOut_Test[:,0], 0) +
                                  (1-self.expectile_value) * tf.maximum(self.criticOut_Test[:,0] - self.value_target_Test[:,0], 0), axis=0)
                

            # add an optimiser
            self.global_step_critic_Test = tf.Variable(0, trainable=False)
            self.lr_critic_Test = tf.compat.v1.placeholder(tf.float32)
            self.criticOptimiser_Test = tf.train.AdamOptimizer(learning_rate=self.lr_critic_Test)\
                .minimize(loss,global_step=self.global_step_critic_Test)
            self.action_grad_Test = tf.gradients(self.criticOut_Test, self.actionIn_Test)
            # self.loss_grad_Q_Test = tf.gradients(loss, self.criticOut_Test)
            
    def onlineCritic_Test_pg(self, inputTensor, scope):
        with tf.name_scope(scope):
            self.mIn_critic_Test_pg,self.actionIn_Test_pg,self.criticOut_Test_pg =self.critic(inputTensor, True,scope)
            self.value_target_Test_pg=tf.placeholder(tf.float32,shape=(None,1))
            
            if self.pMeasure == PerformanceMeasure.MSE:
                loss = tf.reduce_mean(tf.square(self.value_target_Test_pg[:,0] - self.criticOut_Test_pg[:,0]),axis=0)
            if self.pMeasure == PerformanceMeasure.Expectile:
                loss = tf.reduce_mean(self.expectile_value * tf.maximum(self.value_target_Test_pg[:,0] - self.criticOut_Test_pg[:,0], 0.0) ** 2 +
                                            (1-self.expectile_value) * tf.maximum(self.criticOut_Test_pg[:,0] - self.value_target_Test_pg[:,0], 0.0) ** 2, axis=0)
                
            if self.pMeasure == PerformanceMeasure.CVaR:
                loss = tf.reduce_mean(self.expectile_value * tf.maximum(self.value_target_Test_pg[:,0] - self.criticOut_Test_pg[:,0], 0.0) ** 2 +
                                            (1-self.expectile_value) * tf.maximum(self.criticOut_Test_pg[:,0] - self.value_target_Test_pg[:,0], 0.0) ** 2, axis=0)
            
            if self.pMeasure == PerformanceMeasure.Quantile:
                loss = tf.reduce_mean((self.expectile_value) * tf.maximum(self.value_target_Test_pg[:,0] - self.criticOut_Test_pg[:,0], 0) +
                                  (1-self.expectile_value) * tf.maximum(self.criticOut_Test_pg[:,0] - self.value_target_Test_pg[:,0], 0), axis=0)
                

            # add an optimiser
            self.global_step_critic_Test_pg = tf.Variable(0, trainable=False)
            self.lr_critic_Test_pg = tf.compat.v1.placeholder(tf.float32)
            self.criticOptimiser_Test_pg = tf.train.AdamOptimizer(learning_rate=self.lr_critic_Test_pg)\
                .minimize(loss,global_step=self.global_step_critic_Test_pg)
            self.action_grad_Test_pg = tf.gradients(self.criticOut_Test_pg, self.actionIn_Test_pg)
            # self.loss_grad_Q_Test_pg = tf.gradients(loss, self.criticOut_Test_pg)
        
    def onlineCritic_Train(self, inputTensor, scope):
        with tf.name_scope(scope):
            self.mIn_critic_Train,self.actionIn_Train,self.criticOut_Train =self.critic(inputTensor, True,scope)
            self.value_target_Train=tf.placeholder(tf.float32,shape=(None,1))
            
            if self.pMeasure == PerformanceMeasure.MSE:
                loss = tf.reduce_mean(tf.square(self.value_target_Train - self.criticOut_Train),axis=0)
            if self.pMeasure == PerformanceMeasure.Expectile:
                loss = tf.reduce_mean(self.expectile_value * tf.maximum(self.value_target_Train - self.criticOut_Train, 0.0) ** 2 +
                                            (1-self.expectile_value) * tf.maximum(self.criticOut_Train - self.value_target_Train, 0.0) ** 2, axis=0)
            if self.pMeasure == PerformanceMeasure.CVaR:
                loss = tf.reduce_mean(self.expectile_value * tf.maximum(self.value_target_Train - self.criticOut_Train, 0.0) ** 2 +
                                            (1-self.expectile_value) * tf.maximum(self.criticOut_Train - self.value_target_Train, 0.0) ** 2, axis=0)
                
                # x1 = self.value_target_Train - self.criticOut_Train
                # x2 = self.criticOut_Train - self.value_target_Train
                
                # loss = tf.reduce_mean(self.expectile_value * (0.5*x1*(1+tf.math.tanh((2/3.14159)**.5 * (x1+0.044715*x1**3)))) ** 2 +
                #                             (1-self.expectile_value) * (0.5*x2*(1+tf.math.tanh((2/3.14159)**.5 * (x2+0.044715*x2**3)))) ** 2, axis=0)
            
            if self.pMeasure == PerformanceMeasure.Quantile:
                loss = tf.reduce_mean((self.expectile_value) * tf.maximum(self.value_target_Train - self.criticOut_Train, 0) +
                                  (1-self.expectile_value) * tf.maximum(self.criticOut_Train - self.value_target_Train, 0), axis=0)
                

            # add an optimiser
            self.global_step_critic_Train = tf.Variable(0, trainable=False)
            self.lr_critic_Train = tf.compat.v1.placeholder(tf.float32)
            self.criticOptimiser_Train = tf.train.AdamOptimizer(learning_rate=self.lr_critic_Train)\
                .minimize(loss,global_step=self.global_step_critic_Train)
            self.action_grad_Train = tf.gradients(self.criticOut_Train, self.actionIn_Train)
            # self.loss_grad_Q_Train = tf.gradients(loss, self.criticOut_Train)

    def targetActor(self, inputTensor,scope):
        with tf.name_scope(scope):
            self.mIn_target, self.actorOut_target = self.actor(inputTensor, False,scope)
            self.online_actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.actor_scopes[0])
            self.target_actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.actor_scopes[1])
            length = min(len(self.online_actor_params),len(self.target_actor_params))
            params = zip(self.online_actor_params[0:length], self.target_actor_params[0:length])
            self.update_targetActor = [tf.assign(t_a, (1 - self.tau_a) * t_a + self.tau_a * p_a) for p_a, t_a in params]

    def targetCritic(self, inputTensor, scope):
        self.mIn_critic_target, self.actionIn_target, self.criticOut_target = self.critic(inputTensor, True,scope)
        self.target_critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.critic_scopes[1])
        self.online_critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.critic_scopes[0])
        length = min(len(self.online_critic_params), len(self.target_critic_params))
        params = zip(self.online_critic_params[0:length], self.target_critic_params[0:length])
        self.tau_c_value = tf.compat.v1.placeholder(tf.float32)
        self.update_targetCritic = [tf.assign(t_a, (1 - self.tau_c_value) * t_a + self.tau_c_value * p_a) for p_a, t_a in params]
        # self.action_grad = tf.gradients(self.criticOut_target, self.actionIn_target)

    def targetCritic_Test(self, inputTensor, scope):
        self.mIn_critic_target_Test, self.actionIn_target_Test, self.criticOut_target_Test = self.critic(inputTensor, False,scope)
        self.target_critic_params_Test = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.critic_scopes_Test[1])
        self.online_critic_params_Test = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.critic_scopes_Test[0])
        length = min(len(self.online_critic_params_Test), len(self.target_critic_params_Test))
        params = zip(self.online_critic_params_Test[0:length], self.target_critic_params_Test[0:length])
        self.tau_c_value_Test = tf.compat.v1.placeholder(tf.float32)
        self.update_targetCritic_Test = [tf.assign(t_a, (1 - self.tau_c_value_Test) * t_a + self.tau_c_value_Test * p_a) for p_a, t_a in params]
        
    def targetCritic_Test_pg(self, inputTensor, scope):
        self.mIn_critic_target_Test_pg, self.actionIn_target_Test_pg, self.criticOut_target_Test_pg = self.critic(inputTensor, False,scope)
        self.target_critic_params_Test_pg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.critic_scopes_Test_pg[1])
        self.online_critic_params_Test_pg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.critic_scopes_Test_pg[0])
        length = min(len(self.online_critic_params_Test_pg), len(self.target_critic_params_Test_pg))
        params = zip(self.online_critic_params_Test_pg[0:length], self.target_critic_params_Test_pg[0:length])
        self.tau_c_value_Test_pg = tf.compat.v1.placeholder(tf.float32)
        self.update_targetCritic_Test_pg = [tf.assign(t_a, (1 - self.tau_c_value_Test_pg) * t_a + self.tau_c_value_Test_pg * p_a) for p_a, t_a in params]
        
    def targetCritic_Test_match(self):
        # length = min(len(self.target_critic_params), len(self.target_critic_params_Test))
        # params = zip(self.target_critic_params[0:length], self.target_critic_params_Test[0:length]) 
        
        length = min(len(self.online_critic_params), len(self.target_critic_params_Test))
        params = zip(self.online_critic_params[0:length], self.target_critic_params_Test[0:length])
        
        self.match_targetCritic_Test = [tf.assign(t_a,p_a) for p_a, t_a in params]
        
    def onlineCritic_Test_match(self):
        length = min(len(self.online_critic_params), len(self.online_critic_params_Test))
        params = zip(self.online_critic_params[0:length], self.online_critic_params_Test[0:length])        
        self.match_onlineCritic_Test = [tf.assign(t_a,p_a) for p_a, t_a in params]
        
    def targetCritic_Test_pg_match(self):
        length = min(len(self.online_critic_params), len(self.target_critic_params_Test_pg))
        params = zip(self.online_critic_params[0:length], self.target_critic_params_Test_pg[0:length])           
        self.match_targetCritic_Test_pg = [tf.assign(t_a,p_a) for p_a, t_a in params]
        
    def onlineCritic_Test_pg_match(self):
        length = min(len(self.online_critic_params), len(self.online_critic_params_Test_pg))
        params = zip(self.online_critic_params[0:length], self.online_critic_params_Test_pg[0:length])        
        self.match_onlineCritic_Test_pg = [tf.assign(t_a,p_a) for p_a, t_a in params]
        
        
    def targetCritic_Train(self, inputTensor, scope):
        self.mIn_critic_target_Train, self.actionIn_target_Train, self.criticOut_target_Train = self.critic(inputTensor, False,scope)
        self.target_critic_params_Train = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.critic_scopes_Train[1])
        self.online_critic_params_Train = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.critic_scopes_Train[0])
        length = min(len(self.online_critic_params_Train), len(self.target_critic_params_Train))
        params = zip(self.online_critic_params_Train[0:length], self.target_critic_params_Train[0:length])
        self.tau_c_value_Train = tf.compat.v1.placeholder(tf.float32)
        self.update_targetCritic_Train = [tf.assign(t_a, (1 - self.tau_c_value_Train) * t_a + self.tau_c_value_Train * p_a) for p_a, t_a in params]
        
        
        
    def getExpectile(self,Vs,tol,expectile_value=0):
        
        if expectile_value == 0:
            expectile_value = self.expectile_value
        
            
        qmin = np.min(Vs)
        qmax = np.max(Vs)
        while qmax - qmin > tol:
            q = (qmax + qmin)/2
            gradscore = np.mean(-2*expectile_value*np.maximum(Vs-q,0)+2*(1-expectile_value)*np.maximum(q-Vs,0))
            if gradscore > 0:
                qmax = q
            else:
                qmin = q
        
        return (qmax + qmin)/2
    
    

    
    def evaluateDynamicRisk_AO(self, sess, x_inTensor,x_inTensor_Test,price_mat,price_mat_Test,expectile_value_range,short=1):        
    
        grid_s_max = 1.1*np.max(price_mat)
        grid_s_min = 0.9*np.min(price_mat)
        grid_s_disc = self.DP_desc/2
        g1 = np.arange(grid_s_min,self.S_0,(self.S_0-grid_s_min)/grid_s_disc)
        g2 = np.arange(self.S_0[0],grid_s_max,(grid_s_max-self.S_0[0])/grid_s_disc)
        grid_s = np.concatenate([g1,g2])
        V_t = np.zeros(shape=[grid_s.shape[0],price_mat.shape[1]])
        
        
        #Computing the dynamic loss for different time to maturities using AO model
        #######################################################################################
        loss_time_AO = []
        x_inTensor_Test_copy = np.copy(x_inTensor_Test)        
        
        t_counter = 0
        for t_start in range(x_inTensor_Test.shape[1]-1):
        
            time_to_mat = np.zeros(self.n_timesteps + 1)
            time_to_mat[t_start + 1:] = self.T / (self.n_timesteps)  # [0,0,0,h,..,h]
            time_to_mat = np.cumsum(time_to_mat)  # [0,0,0,h,2h...,(N-starting_t)h]
            time_to_mat = time_to_mat[::-1]  # [(N-starting_t)h, (N-starting_t-1)h,...,h,0,0,0]
            x_inTensor_Test_copy[:,:,-2] = np.reshape(time_to_mat,[1,self.n_timesteps+1])
            all_actions = sess.run(self.deltas,feed_dict={self.mIn_pg: x_inTensor_Test_copy[:,:,0:-1],
            self.short: short})
            
            seed_counter = 0
            for t_ in range(price_mat.shape[1]-1-t_start,-1,-1):
                for s_idx in range(grid_s.shape[0]):                
                    if t_ == price_mat.shape[1]-1-t_start:
                        if short == 1:
                            V_t[s_idx,t_] = np.maximum(grid_s[s_idx] - self.strike,0)
                        else:
                            V_t[s_idx,t_] = -np.maximum(grid_s[s_idx] - self.strike,0)
                        
                    else:
                        # compute the possible values of price in the next period
                        ###############################################################                    
                        np.random.seed(seed_counter)
                        seed_counter = seed_counter + 1
                        rand_stdnorm = np.random.randn(self.DP_desc)
                        h = self.T / self.n_timesteps
                        prices = grid_s[s_idx] * np.exp((self.mu - self.sigma ** 2 / 2) * h + self.sigma * np.sqrt(h) * rand_stdnorm)
                        max_next_price =np.max(prices)
                        min_next_price =np.min(prices)
                        ###############################################################
                        next_s_idx =[np.argmin(np.abs(grid_s-prices[i])) for i in range(len(prices))] #(grid_s <= max_next_price) & (grid_s >= min_next_price)
                        next_s_val = [grid_s[i] for i in next_s_idx]
                        
                        # Finding closes price in sample trajectories to the current grid price
                        action_idx = np.argmin(np.abs(grid_s[s_idx] - price_mat_Test[:,t_,0]))
                        action = all_actions[action_idx,t_,0]
                        V_t_temp = [V_t[i,t_+1] for i in next_s_idx]
                        if len(next_s_val) == 0:
                            print('naaaaaaan')
                        if self.pMeasure == PerformanceMeasure.Expectile:
                            V_t_all = np.array(V_t_temp  - action * (next_s_val - grid_s[s_idx]))                        
                            L_expectile_value = self.getExpectile(V_t_all,1e-5)
                            V_t[s_idx,t_] = L_expectile_value
                print(t_)
            
            
            V_t_idx = np.argmin(np.abs(grid_s - self.S_0[0]))
            loss_time_AO.append(V_t[V_t_idx,0])
            t_counter = t_counter + 1
                          
        return loss_time_AO
    
    
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    def train_test_pg(self, sess, x_inTensor,x_inTensor_Test,short=1):
        sess.run(self.init_op)
        expectile_value_range = np.arange(self.expectile_value,1,(1-self.expectile_value)/10)
        
        price_mat = np.exp(x_inTensor[:, :, 0:-2]) * np.reshape(self.S_0,[1,1,self.num_stocks])
        price_mat_Test = np.exp(x_inTensor_Test[:,:, 0:-2]) * np.reshape(self.S_0,[1,1,self.num_stocks])
        
        seed_counter = 0
        hedgingError_mean=[]
        hedgingError_cvar=[]
        hedgingError_min=[]
        hedgingError_max=[]
        hedgingError_expectile = np.empty([0,len(expectile_value_range)])
        
        hedgingError_mean_test=[]
        hedgingError_cvar_test=[]
        hedgingError_max_test=[]
        hedgingError_expectile_test = np.empty([0,len(expectile_value_range)])
        for epoch in range(self.epochs + 1):           
            
            
            samples_idx = np.random.choice(x_inTensor.shape[0], self.minibatchSize_pg, replace=False)
            samples = x_inTensor[samples_idx]            
            
            sess.run(self.actorOptimiser_adam,
                          feed_dict={self.mIn_pg: samples[:,:,0:-1], 
                                    self.lr_pg: self.learningRate_pg,
                                    self.short: short})                    
            
            if epoch%100 == 0 and epoch > 0:
                rew,deltas = sess.run([self.reward,self.deltas],
                                    feed_dict={self.mIn_pg: x_inTensor[:,:,0:-1], 
                                                self.lr_pg: self.learningRate_pg,
                                                self.short: short})
                
                
                
                hedgingError_mean.append(np.mean(rew))
                hedgingError_cvar.append(np.mean(np.sort(rew)[int(self.expectile_value * len(rew)):]))
                hedgingError_max.append(np.max(rew))
                hedgingError_min.append(np.min(rew))                
                hedgingError_expectile_temp = np.expand_dims([self.getExpectile(rew,1e-5, exp_value) for exp_value in expectile_value_range],axis=0)
                hedgingError_expectile = np.concatenate([hedgingError_expectile,hedgingError_expectile_temp],axis=0)
                
                
                #Run over test data
                ################################################################                
                mIn_test = np.array(x_inTensor_Test[:,:,0:-1])
                rew_test,deltas_test = sess.run([self.reward,self.deltas],
                                    feed_dict={self.mIn_pg: mIn_test, 
                                               self.lr_pg: self.learningRate_pg,
                                               self.short: short})
                
                hedgingError_mean_test.append(np.mean(rew_test))
                hedgingError_cvar_test.append(np.mean(np.sort(rew_test)[int(self.expectile_value * len(rew_test)):]))
                hedgingError_max_test.append(np.max(rew_test))
                hedgingError_expectile_temp_test = np.expand_dims([self.getExpectile(rew_test,1e-5, exp_value) for exp_value in expectile_value_range],axis=0)
                hedgingError_expectile_test = np.concatenate([hedgingError_expectile_test,hedgingError_expectile_temp_test],axis=0)
                ################################################################
                
                
                print('epoch       : ' + str(epoch))
                print('min is      : ' + str( np.min(rew)))
                print('mean is     : ' + str(np.mean(rew)))
                print('exp_In is   : ' + str(hedgingError_expectile_temp[0,0]))
                print('exp_Out is  : ' + str(hedgingError_expectile_temp_test[0,0]))
                print('cvar is     : ' + str(np.mean(np.sort(rew)[int(self.expectile_value * len(rew)):])))
                print('max is      : ' + str(np.max(rew)))
                print('-------------------------------------------')
            
            
            if epoch%5000 == 0 and epoch > 0:                
                for iii in range(self.num_stocks):
                    fig = plt.figure()
                    [plt.scatter(price_mat_Test[:,i,iii],deltas_test[:,i,iii],s=1) for i in range(deltas_test.shape[1])]   
                    plt.plot([self.S_0[iii],self.S_0[iii]],[np.min(deltas_test),np.max(deltas_test)])
                    plt.xlabel("Stock value")
                    plt.ylabel("Investment (in  shares)")
                    plt.savefig(self.save_path+'AO/actions'+str(iii) + '_' +str(short)+'.png')                     
                    plt.close(fig)
                    
                
                fig = plt.figure()
                plt.plot(hedgingError_mean, label='In sample')
                plt.plot(hedgingError_mean_test, label='Out of sample')
                plt.legend()
                plt.savefig(self.save_path+'AO/HE_mean'+str(short)+'.png')
                plt.close(fig)
                
                fig = plt.figure()
                plt.plot(hedgingError_min)
                plt.savefig(self.save_path+'AO/HE_min'+str(short)+'.png')
                plt.close(fig)
                
                fig = plt.figure()
                plt.plot(hedgingError_max, label='In sample')
                plt.plot(hedgingError_max_test, label='Out of sample')
                plt.legend()
                plt.savefig(self.save_path+'AO/HE_max'+str(short)+'.png')
                plt.close(fig)
                
                fig = plt.figure()
                plt.plot(hedgingError_cvar, label='In sample')
                plt.plot(hedgingError_cvar_test, label='Out of sample')
                plt.legend()
                plt.savefig(self.save_path+'AO/HE_cvar'+str(short)+'.png')
                plt.close(fig)
                
                fig = plt.figure()
                [plt.plot(hedgingError_expectile[:,i], label=str(np.round(expectile_value_range[i],2)) + ' expectile') for i in range(len(expectile_value_range))]                  
                plt.xlabel("# of episodes")
                plt.ylabel("validation scores")
                # plt.legend()  
                plt.savefig(self.save_path+'AO/HE_expectile_Insample'+str(short)+'.png')
                plt.close(fig)
                
                fig = plt.figure()
                [plt.plot(hedgingError_expectile_test[:,i], label=str(np.round(expectile_value_range[i],2))  + ' expectile') for i in range(len(expectile_value_range))]                               
                plt.xlabel("# of episodes")
                plt.ylabel("validation scores")
                # plt.legend()  
                plt.savefig(self.save_path+'AO/HE_expectile_Outsample'+str(short)+'.png')                
                plt.close(fig)
                
                df = pd.DataFrame(columns=['mean','cvar','min','max','mean_test','cvar_test','max_test'])
                df['mean'] = hedgingError_mean                
                df['cvar'] = hedgingError_cvar
                df['min'] = hedgingError_min
                df['max'] = hedgingError_max
                df['mean_test'] = hedgingError_mean_test
                df['cvar_test'] = hedgingError_cvar_test
                df['max_test'] = hedgingError_max_test
                df.to_csv(self.save_path+'AO/HE'+str(short)+'.csv')        
        
        
        #######################################################################################
        
        #Computing the loss for different time to maturities
        #######################################################################################
        loss_time_AO = 0
        
        # if self.num_stocks == 1:
        #     loss_time_AO = self.evaluateDynamicRisk_AO(sess, x_inTensor,x_inTensor_Test,price_mat,price_mat_Test,expectile_value_range,short)
        
        
        # out1 = hedgingError_mean[-1]
        # out2 = hedgingError_mean_test[-1]
        # out3 = hedgingError_expectile[-1]
        # out4 = hedgingError_expectile_test[-1]
        # out5 = hedgingError_cvar[-1]
        # out6 = hedgingError_cvar_test[-1]        
        # out7 = hedgingError_max[-1]
        # out8 = hedgingError_max_test[-1]
        
        return loss_time_AO
    
        
    def expectile_function(self,x,alpha,V_t):
        L = np.mean(alpha * np.maximum(V_t - x,0)**2 + (1 - alpha) * np.maximum(x - V_t,0)**2)
        return L
    
    
    def evaluateStaticRisk_AC(self,sess,x_inTensor_Test,price_mat_Test,starting_t,short):
        time_to_mat = np.zeros(self.n_timesteps + 1)
        time_to_mat[starting_t + 1:] = self.T / (self.n_timesteps)  # [0,0,0,h,..,h]
        time_to_mat = np.cumsum(time_to_mat)  # [0,0,0,h,2h...,(N-starting_t)h]
        time_to_mat = time_to_mat[::-1]  # [(N-starting_t)h, (N-starting_t-1)h,...,h,0,0,0]        
        x_inTensor_Test[:,:,-2] = np.reshape(time_to_mat,[1,self.n_timesteps+1])
        for t_ in range(x_inTensor_Test.shape[1]-starting_t-1):
            # Run over test data    
            ###########################################################################
            mIn_test = x_inTensor_Test[:,t_,:]
            if self.No_wealth == True:
                mIn_test = mIn_test[:,0:-1]
            action_test = sess.run(self.actorOut,feed_dict={self.mIn: mIn_test})
                
            x_inTensor_Test[:, t_ + 1,-1] = x_inTensor_Test[:, t_,-1] + np.sum(action_test * (price_mat_Test[:, t_+1,:]-price_mat_Test[:, t_,:]),axis=1)
            
            if t_ == x_inTensor_Test.shape[1] - 2 - starting_t:
                if short == 1:
                    Q_value_test = np.maximum(np.mean(price_mat_Test[:, t_+1,:],1) - self.strike,0) - x_inTensor_Test[:, t_ + 1,-1]
                else:
                    Q_value_test = -np.maximum(np.mean(price_mat_Test[:, t_+1,:],1) - self.strike,0) - x_inTensor_Test[:, t_ + 1,-1]
                            
    
                
        return self.getExpectile(Q_value_test,1e-5, self.expectile_value)
    
    
    
    def evaluateStaticRisk_DP(self,price_mat_Test,starting_t,short,a_t,grid_s):
        
        a_t_new = a_t[:,starting_t:]
        
        W_t_test = np.zeros(shape=price_mat_Test.shape)
        HE_T_test = np.zeros(shape=price_mat_Test.shape[0])
        for t_ in range(a_t_new.shape[1]):
            for s_idx in range(price_mat_Test.shape[0]):                 
                if t_ == price_mat_Test.shape[1] - 1 - starting_t:
                    if short == 1:
                        HE_T_test[s_idx] = np.maximum(price_mat_Test[s_idx,t_] - self.strike,0) - W_t_test[s_idx,t_]
                    else:
                        HE_T_test[s_idx] = -np.maximum(price_mat_Test[s_idx,t_] - self.strike,0) - W_t_test[s_idx,t_]
                else:                
                    action_idx = np.argmin(np.abs(grid_s - price_mat_Test[s_idx,t_]))
                    action_test = a_t_new[action_idx,t_]
                    W_t_test[s_idx,t_+1] = W_t_test[s_idx,t_] + action_test * (price_mat_Test[s_idx,t_+1] - price_mat_Test[s_idx,t_])
                            
    
                
        return self.getExpectile(HE_T_test,1e-5, self.expectile_value)
    
    
    
    def evaluateStaticRisk_AO(self,sess,x_inTensor_Test,price_mat_Test,starting_t,short):        
        time_to_mat = np.zeros(self.n_timesteps + 1)
        time_to_mat[starting_t + 1:] = self.T / (self.n_timesteps)  # [0,0,0,h,..,h]
        time_to_mat = np.cumsum(time_to_mat)  # [0,0,0,h,2h...,(N-starting_t)h]
        time_to_mat = time_to_mat[::-1]  # [(N-starting_t)h, (N-starting_t-1)h,...,h,0,0,0]        
        x_inTensor_Test[:,:,-2] = np.reshape(time_to_mat,[1,self.n_timesteps+1])
        
        all_actions = sess.run(self.deltas,feed_dict={self.mIn_pg: x_inTensor_Test[:,:,0:-1], 
                                               self.lr_pg: self.learningRate_pg,
                                               self.short: short})
        for t_ in range(x_inTensor_Test.shape[1]-starting_t-1):
            # Run over test data    
            ###########################################################################
            action_test = all_actions[:,t_,:]
                
            x_inTensor_Test[:, t_ + 1, -1] = x_inTensor_Test[:, t_,-1] + np.sum(action_test * (price_mat_Test[:, t_+1,:]-price_mat_Test[:, t_,:]),axis=1)            
            if t_ == x_inTensor_Test.shape[1] - 2 - starting_t:
                if short == 1:
                    Q_value_test = np.maximum(np.mean(price_mat_Test[:, t_+1,:],1) - self.strike,0) - x_inTensor_Test[:, t_ + 1,-1]
                else:
                    Q_value_test = -np.maximum(np.mean(price_mat_Test[:, t_+1,:],1) - self.strike,0) - x_inTensor_Test[:, t_ + 1,-1]
                            
    
                
        return self.getExpectile(Q_value_test,1e-5, self.expectile_value)
    
    
        
                
    def evaluateDynamicRisk_AC(self, sess, x_inTensor,x_inTensor_Test,price_mat,price_mat_Test,expectile_value_range,short=1):        
    
        grid_s_max = 1.1*np.max(price_mat)
        grid_s_min = 0.9*np.min(price_mat)
        grid_s_disc = self.DP_desc/2
        g1 = np.arange(grid_s_min,self.S_0,(self.S_0-grid_s_min)/grid_s_disc)
        g2 = np.arange(self.S_0[0],grid_s_max,(grid_s_max-self.S_0[0])/grid_s_disc)
        grid_s = np.concatenate([g1,g2])
        V_t = np.zeros(shape=[grid_s.shape[0],price_mat.shape[1]])
        
        
        #Computing the dynamic loss for different time to maturities using AC model
        #######################################################################################
        loss_time_AO = []
        x_inTensor_Test_copy = np.copy(x_inTensor_Test)        
        
        t_counter = 0
        for t_start in range(x_inTensor_Test.shape[1]-1):
        
            time_to_mat = np.zeros(self.n_timesteps + 1)
            time_to_mat[t_start + 1:] = self.T / (self.n_timesteps)  # [0,0,0,h,..,h]
            time_to_mat = np.cumsum(time_to_mat)  # [0,0,0,h,2h...,(N-starting_t)h]
            time_to_mat = time_to_mat[::-1]  # [(N-starting_t)h, (N-starting_t-1)h,...,h,0,0,0]
            x_inTensor_Test_copy[:,:,-2] = np.reshape(time_to_mat,[1,self.n_timesteps+1])
            
            
            seed_counter = 0
            for t_ in range(price_mat.shape[1]-1-t_start,-1,-1):
                mIn = x_inTensor_Test_copy[:,t_,:]
                if self.No_wealth == True:
                    mIn = mIn[:,0:-1]
                
                all_actions = sess.run(self.actorOut,feed_dict={self.mIn: mIn})
                
                for s_idx in range(grid_s.shape[0]):                
                    if t_ == price_mat.shape[1]-1-t_start:
                        if short == 1:
                            V_t[s_idx,t_] = np.maximum(grid_s[s_idx] - self.strike,0)
                        else:
                            V_t[s_idx,t_] = -np.maximum(grid_s[s_idx] - self.strike,0)
                        
                    else:
                        # compute the possible values of price in the next period
                        ###############################################################                    
                        np.random.seed(seed_counter)
                        seed_counter = seed_counter + 1
                        rand_stdnorm = np.random.randn(self.DP_desc)
                        h = self.T / self.n_timesteps
                        prices = grid_s[s_idx] * np.exp((self.mu - self.sigma ** 2 / 2) * h + self.sigma * np.sqrt(h) * rand_stdnorm)
                        max_next_price =np.max(prices)
                        min_next_price =np.min(prices)
                        ###############################################################
                        next_s_idx =[np.argmin(np.abs(grid_s-prices[i])) for i in range(len(prices))] #(grid_s <= max_next_price) & (grid_s >= min_next_price)
                        next_s_val = [grid_s[i] for i in next_s_idx]
                        
                        # Finding closes price in sample trajectories to the current grid price
                        action_idx = np.argmin(np.abs(grid_s[s_idx] - price_mat_Test[:,t_,0]))
                        action = all_actions[action_idx,0]
                        V_t_temp = [V_t[i,t_+1] for i in next_s_idx]
                        if len(next_s_val) == 0:
                            print('naaaaaaan')
                        if self.pMeasure == PerformanceMeasure.Expectile:
                            V_t_all = np.array(V_t_temp  - action * (next_s_val - grid_s[s_idx]))                        
                            L_expectile_value = self.getExpectile(V_t_all,1e-5)
                            V_t[s_idx,t_] = L_expectile_value
                print(t_)
            
            
            V_t_idx = np.argmin(np.abs(grid_s - self.S_0[0]))
            loss_time_AO.append(V_t[V_t_idx,0])
            t_counter = t_counter + 1
                          
        return loss_time_AO
    
    
    
    def minimize_objective_fc(self, x, expect_set):
        vals = 0
        for idx, each_expectile in enumerate(expect_set):
            diff = x - each_expectile
            diff = np.where(diff > 0, - self.cum_density[idx] * diff, (self.cum_density[idx] - 1) * diff)
            vals += np.square(np.mean(diff))

        return vals        
    
    def root_objective_fc(self, x, expect_set):
        vals = []
        for idx, each_expectile in enumerate(expect_set):
            diff = x - each_expectile
            diff = np.where(diff > 0, - self.cum_density[idx] * diff, (self.cum_density[idx] - 1) * diff)
            vals.append(np.mean(diff))
        return vals
    
    
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    def train_test_ddpg(self, sess, x_inTensor,x_inTensor_Test,short=1,useFullSavedModel=False):
        sess.run(self.init_op)
        
        
        if self.loadModelFromSavedModels == True:
            if useFullSavedModel == True:
                self.saver.restore(sess, self.save_path+'AC/MODELS/'+str(self.num_stocks)+'/'+str(short)+'/Full/model.ckpt')
            else:
                self.saver.restore(sess, self.save_path+'AC/MODELS/'+str(self.num_stocks)+'/'+str(short)+'/model.ckpt')        
       
        
        #Creating the price matrix
        price_mat = np.exp(x_inTensor[:, :, 0:-2]) * np.reshape(self.S_0,[1,1,self.num_stocks])
        price_mat_Test = np.exp(x_inTensor_Test[:, :, 0:-2]) * np.reshape(self.S_0,[1,1,self.num_stocks])
        
        expectile_value_range = np.arange(self.expectile_value,1,(1-self.expectile_value)/10)
        
        hedgingError_mean = []
        hedgingError_cvar = []
        hedgingError_expectile = np.empty([0,len(expectile_value_range)])
        hedgingError_max = []
        hedgingError_min = []
        Q_0 = []
        
        
        hedgingError_mean_Test = []
        hedgingError_cvar_Test = []
        hedgingError_expectile_Test = np.empty([0,len(expectile_value_range)])
        hedgingError_max_Test = []
        # hedgingError_min_Test = []
        Q_0_Test = [0]
        Q_0_Test_pg = [0]   
        Q_0_Train = [0]
        
        
    
        epoch = 0        
        start_time = time.time()
        Q_target_MAX = 0
        Q_target_MIN = 0
        wealth = np.zeros([x_inTensor.shape[0],x_inTensor.shape[1]])
        
        target_net_update_episodes = 5 ## ## 5
        exploration_prob = .5 ## ## .5
        
        while epoch <= self.epochs:            
            
            t_ = np.random.randint(0,x_inTensor.shape[1] - 1)
            
            # if epoch % 1000 == 0:
            #     t_ = x_inTensor.shape[1] - 2
            
            samples_idx = np.random.choice(x_inTensor.shape[0], self.minibatchSize, replace=False)
            samples_inTensor = x_inTensor[samples_idx]
            samples_price_mat = price_mat[samples_idx]
            samples_wealth = wealth[samples_idx]            
            
            samples = samples_inTensor[:,t_,:]            
            sample_prices = samples_price_mat[:,t_,:]                     
            samples_next = samples_inTensor[:,t_+1,:]
            sample_prices_next = samples_price_mat[:,t_+1,:]
            
            
            mIn = samples
            mIn_next = samples_next
            
            #Setting the wealth level for both periods equal so that we can have immediate rewards            
            
            if self.No_wealth == True:
                mIn = mIn[:,0:-1]
                mIn_next = mIn_next[:,0:-1]
            else:
                mIn[:,-1] = samples_wealth[:,t_]
                mIn_next[:,-1] = samples_wealth[:,t_+1]
                
            
            action = sess.run(self.actorOut,feed_dict={self.mIn: mIn})
            # exploration_prob_current = exploration_prob - epoch * exploration_prob/self.epochs
            exploration_prob_current = exploration_prob
            rnd_choice = np.random.choice([0,1], action.shape, p=[1-exploration_prob_current,exploration_prob_current])
            action = np.clip(action + np.random.normal(0,.5,action.shape)*rnd_choice,-1,1)
            
            action_target = sess.run(self.actorOut_target, feed_dict={self.mIn_target: mIn_next})
            Q_target = sess.run(self.criticOut_target,feed_dict={self.mIn_critic_target: mIn_next,self.actionIn_target:action_target})
            
            
            # samples_wealth_t = samples_wealth[:,t_] + np.sum(action* (sample_prices_next-sample_prices),axis=1,keepdims=False)           
            
            rews = np.sum(-action * (sample_prices_next-sample_prices),axis=1,keepdims=True) + Q_target
            
            #Update wealth
            if self.No_wealth == False:
                samples_wealth[:,t_+1] = samples_wealth[:,t_] + np.sum(action* (sample_prices_next-sample_prices),axis=1)
            
            if t_ == x_inTensor.shape[1] - 2:
                if short == 1:
                    if self.No_wealth == True:
                        rews[:,0] = np.maximum(np.mean(sample_prices_next,1) - self.strike,0) - np.sum(action * (sample_prices_next-sample_prices),axis=1)
                    else:
                        rews[:,0] = np.maximum(np.mean(sample_prices_next,1) - self.strike,0) - samples_wealth[:,t_+1]
                    
                else:
                    if self.No_wealth == True:
                        rews[:,0] = -np.maximum(np.mean(sample_prices_next,1) - self.strike,0) - np.sum(action * (sample_prices_next-sample_prices),axis=1)
                    else:
                        rews[:,0] = -np.maximum(np.mean(sample_prices_next,1) - self.strike,0) - samples_wealth[:,t_+1]
            
            

            #Update the online critic
            lr = self.learningRate[0]
            sess.run(self.criticOptimiser,
                        feed_dict={self.mIn_critic: mIn,
                                    self.actionIn: action,
                                    self.value_target: rews,
                                    self.lr_critic: lr})
            
            epoch += 1            
            
            action = sess.run(self.actorOut,feed_dict={self.mIn: mIn})
            
            action_grad = sess.run(self.action_grad,
                                        feed_dict={self.mIn_critic: mIn,
                                                    self.actionIn: action})
            
            #Update online actor
            lr = 5e-6 ## ## 1e-6
            sess.run(self.actorOptimiser,
                      feed_dict={self.mIn: mIn,
                                self.critic_action_grad: action_grad[0],
                                self.lr: lr})  
            
            
            if epoch % target_net_update_episodes == 0:                               
                
                sess.run(self.update_targetCritic, feed_dict={self.tau_c_value: self.tau_c})                
                
                # update the target actor
                sess.run(self.update_targetActor)
            
                
            if epoch == 1 or epoch %1000 == 0:
                x_inTensor_copy = np.copy(x_inTensor)
                hedging_action = np.empty([x_inTensor_copy.shape[0],0,self.num_stocks])
                hedging_action_test = np.empty([x_inTensor_Test.shape[0],0,self.num_stocks])
                wealth = np.zeros([x_inTensor.shape[0],x_inTensor.shape[1],1])
                wealth_target = np.zeros([x_inTensor.shape[0],x_inTensor.shape[1],1])
                
                for t_ in range(x_inTensor_copy.shape[1] - 1):
                    
                    # Run over train data    
                    ###########################################################################
                    mIn = x_inTensor_copy[:,t_,:]
                    if self.No_wealth == True:
                        mIn = mIn[:,0:-1]
                    action = sess.run(self.actorOut,feed_dict={self.mIn: mIn})
                    action_target = sess.run(self.actorOut_target, feed_dict={self.mIn_target: mIn})
                    hedging_action = np.append(hedging_action,np.expand_dims(action,1),axis=1)
                        
                    v_t_next = x_inTensor_copy[:, t_,-1] + np.sum(action * (price_mat[:, t_+1,:]-price_mat[:, t_,:]),axis=1)
                    x_inTensor_copy[:, t_ + 1,-1] = v_t_next
                    wealth[:,t_+1,0] = wealth[:,t_,0] + np.sum(action* (price_mat[:, t_+1,:]-price_mat[:, t_,:]),axis=1)
                    wealth_target[:,t_+1,0] = wealth_target[:,t_,0] + np.sum(action_target* (price_mat[:, t_+1,:]-price_mat[:, t_,:]),axis=1)
                    
                    
                    if t_ == x_inTensor_copy.shape[1] - 2:
                        if short == 1:
                            Q_value = np.maximum(np.mean(price_mat[:, t_+1,:],1) - self.strike,0) - v_t_next
                            Q_target_MAX = self.getExpectile(np.maximum(np.mean(price_mat[:, t_+1,:],1) - self.strike,0) - wealth_target[:,t_+1,0],1e-5, .99)
                            # Q_target_MAX = np.max(np.maximum(np.mean(price_mat[:, t_+1,:],1) - self.strike,0) - wealth_target[:,t_+1,0])
                            # Q_target_MIN = np.min(Q_value)
                            Q_target_MIN = self.getExpectile(Q_value,1e-5, self.expectile_value)
                        else:
                            Q_value = -np.maximum(np.mean(price_mat[:, t_+1,:],1) - self.strike,0) - v_t_next     
                            # Q_target_MAX = np.max(np.maximum(np.mean(price_mat[:, t_+1,:],1) - self.strike,0) - wealth_target[:,t_+1,0])
                            Q_target_MAX = self.getExpectile(np.maximum(np.mean(price_mat[:, t_+1,:],1) - self.strike,0) - wealth_target[:,t_+1,0],1e-5, .99)
                            # Q_target_MIN = np.min(Q_value)
                            Q_target_MIN = self.getExpectile(Q_value,1e-5, self.expectile_value)
                        
                        
                        hedgingError_mean.append(np.mean(Q_value))
                        hedgingError_cvar.append(np.mean(np.sort(Q_value)[int(self.expectile_value * Q_value.shape[0]):]))                        
                        
                        
                        hedgingError_max.append(np.max(Q_value))
                        hedgingError_min.append(np.min(Q_value))
                        
                        # L_expectile_value = self.getExpectile(Q_value,1e-5)
                        # hedgingError_expectile.append(L_expectile_value)
                        hedgingError_expectile_temp = np.expand_dims([self.getExpectile(Q_value,1e-5, exp_value) for exp_value in expectile_value_range],axis=0)
                        hedgingError_expectile = np.concatenate([hedgingError_expectile,hedgingError_expectile_temp],axis=0)
                        
                        
                        mIn = x_inTensor[0:1,0,:]
                        if self.No_wealth == True:
                            mIn = mIn[:,0:-1]
                        # action = sess.run(self.actorOut_target,feed_dict={self.mIn_target: mIn})
                        # Q = sess.run(self.criticOut_target,feed_dict={self.mIn_critic_target: mIn,self.actionIn_target: np.squeeze(action,1)})
                        action = sess.run(self.actorOut,feed_dict={self.mIn: mIn})
                        Q = sess.run(self.criticOut_target,feed_dict={self.mIn_critic_target: mIn,self.actionIn_target: action})
                        Q_0.append(Q[0])
                        
                        
                    
                    ###########################################################################
                        
                        
                    # Run over test data    
                    ###########################################################################
                    mIn_test = x_inTensor_Test[:,t_,:]
                    if self.No_wealth == True:
                        mIn_test = mIn_test[:,0:-1]
                    action_test = sess.run(self.actorOut,feed_dict={self.mIn: mIn_test})
                    hedging_action_test = np.append(hedging_action_test,np.expand_dims(action_test,1),axis=1)
                    # action_test = np.zeros([x_inTensor_Test.shape[0],1,self.num_stocks])
                        
                    v_t_next_test = x_inTensor_Test[:, t_,-1] + np.sum(action_test * (price_mat_Test[:, t_+1,:]-price_mat_Test[:, t_,:]),axis=1)
                    x_inTensor_Test[:, t_ + 1,-1] = v_t_next_test
                    if t_ == x_inTensor_Test.shape[1] - 2:
                        if short == 1:
                            Q_value_test = np.maximum(np.mean(price_mat_Test[:, t_+1,:],1) - self.strike,0) - v_t_next_test
                        else:
                            Q_value_test = -np.maximum(np.mean(price_mat_Test[:, t_+1,:],1) - self.strike,0) - v_t_next_test
                                    
        
                        hedgingError_mean_Test.append(np.mean(Q_value_test))
                        hedgingError_cvar_Test.append(np.mean(np.sort(Q_value_test)[int(self.expectile_value*len(Q_value_test)):]))
                        hedgingError_max_Test.append(np.max(Q_value_test))
                        # L_expectile_value = self.getExpectile(Q_value_test,1e-5)
                        # hedgingError_expectile_Test.append(L_expectile_value)
                        
                        hedgingError_expectile_Test_temp = np.expand_dims([self.getExpectile(Q_value_test,1e-5, exp_value) for exp_value in expectile_value_range],axis=0)
                        hedgingError_expectile_Test = np.concatenate([hedgingError_expectile_Test,hedgingError_expectile_Test_temp],axis=0)
                        
                    
                        print('---- epoch: ' + str(epoch))
                        print('min      : ' + str(np.min(Q_value)))
                        print('mean     : ' + str(np.mean(Q_value)))
                        print('exp_In   : ' + str((hedgingError_expectile_temp[0,0])))
                        print('exp_Out  : ' + str((hedgingError_expectile_Test_temp[0,0])))
                        print('cvar is  : ' + str(np.mean(np.sort(Q_value)[int(self.expectile_value * len(Q_value)):])))
                        print('max      : ' + str(np.max(Q_value)))
                        elapsed_time  = int(time.time() - start_time)
                        print('time     : ' +str(elapsed_time) + ' seconds')
                        start_time = time.time()
                        print('-------------------------------------------------------')
                        
                    ###########################################################################
                                              
                        
                        if epoch%50000 == 0 or epoch == 1:
                            
                            for iii in range(self.num_stocks):
                                fig = plt.figure()
                                [plt.scatter(price_mat_Test[:,i,iii],hedging_action_test[:,i,iii],s=1) for i in range(hedging_action_test.shape[1])]                                 
                                # plt.axvline(x=self.S_0[iii], ymin=np.min(hedging_action_test), ymax=np.max(hedging_action_test))
                                plt.plot([self.S_0[iii],self.S_0[iii]],[np.min(hedging_action_test),np.max(hedging_action_test)])
                                plt.xlabel("Stock value")
                                plt.ylabel("Investment (in  shares)")
                                plt.savefig(self.save_path+'AC/action'+str(iii)+'_'+str(short)+'.png')
                                plt.close(fig)
                                
                            
                            fig = plt.figure()
                            plt.plot(hedgingError_mean, label='In sample')
                            plt.plot(hedgingError_mean_Test, label='Out of sample')
                            plt.legend()
                            plt.savefig(self.save_path+'AC/HE_mean'+str(short)+'.png')
                            plt.close(fig)
                            
                            fig = plt.figure()
                            [plt.plot(hedgingError_expectile[:,i], label= str(int(np.round(expectile_value_range[i]*100,0))) + '% expectile') for i in range(len(expectile_value_range))]  
                            plt.xlabel("Number of episodes")
                            plt.ylabel("Validation scores")
                            # plt.legend()  
                            plt.savefig(self.save_path+'AC/HE_expectile_Insample'+str(short)+'.png')
                            plt.close(fig)
                            
                            fig = plt.figure()
                            [plt.plot(hedgingError_expectile_Test[:,i], label=str(int(np.round(expectile_value_range[i]*100,0))) + '% expectile') for i in range(len(expectile_value_range))]                               
                            plt.xlabel("Number of episodes")
                            plt.ylabel("Validation scores")
                            plt.legend()
                            plt.savefig(self.save_path+'AC/HE_expectile_Outsample'+str(short)+'.png')
                            plt.close(fig)
                            
                            fig = plt.figure()
                            plt.plot(hedgingError_cvar, label='In sample')
                            plt.plot(hedgingError_cvar_Test, label='Out of sample')
                            plt.legend()
                            plt.savefig(self.save_path+'AC/HE_cvar'+str(short)+'.png')
                            plt.close(fig)
                            
                            fig = plt.figure()
                            plt.plot(hedgingError_max, label='In sample')
                            plt.plot(hedgingError_max_Test, label='Out of sample')
                            plt.legend()
                            plt.savefig(self.save_path+'AC/HE_max'+str(short)+'.png')
                            plt.close(fig)
                            
                            df = pd.DataFrame(columns=['mean','cvar','min','max','Q_0','mean_test','max_test','cvar_test'])
                            df['mean'] = hedgingError_mean
                            df['cvar'] = hedgingError_cvar
                            df['min'] = hedgingError_min
                            df['max'] = hedgingError_max
                            df['Q_0'] = Q_0
                            df['mean_test'] = hedgingError_mean_Test
                            df['max_test'] = hedgingError_max_Test
                            df['cvar_test'] = hedgingError_cvar_Test
                            df.to_csv(self.save_path+'AC/HE'+str(short)+'.csv')
                            
                            
                            fig = plt.figure()
                            plt.plot(Q_0)
                            plt.xlabel("# of episodes")
                            plt.ylabel("Q-function")
                            plt.savefig(self.save_path+'AC/Q_0'+str(short)+'.png')                            
                            plt.close(fig)
                            
                            # Plot Q as a function of time to maturity
                            ##########################################################################
                            mIn = np.array([x_inTensor[0,jj,:] for jj in range(x_inTensor.shape[1]-1)])
                            if self.No_wealth == True:
                                mIn = mIn[:,0:-1]
                            mIn[:,0:-1] = mIn[0,0:-1]
                            action = sess.run(self.actorOut,feed_dict={self.mIn: mIn})
                            Q_time = sess.run(self.criticOut,feed_dict={self.mIn_critic: mIn,self.actionIn: action})
                            Q_time = np.append(Q_time,0)
                            fig = plt.figure()
                            plt.plot(np.arange(0,x_inTensor.shape[1],1), Q_time, label='Q_time', marker='o')
                            plt.xticks(np.arange(0,x_inTensor.shape[1],1),np.arange(x_inTensor.shape[1]-1,-1,-1))
                            plt.xlabel("Maturity (in months)")
                            plt.ylabel("Q-function")
                            plt.savefig(self.save_path+'AC/Q_time'+str(short)+'.png')                            
                            plt.close(fig)
                            ##########################################################################
                            
                            
    
            
    
            if self.loadModelFromSavedModels == True:
                break
            
        
            # Save the model 
            if epoch > 0 and epoch %50000 == 0:
                self.saver.save(sess, self.save_path+'AC/MODELS/'+str(self.num_stocks)+'/'+str(short)+'/model.ckpt')
        
        
        
        # Run the model over test data with fixed actions (AC actions) to find the value of Q over test data
        #######################################################################################
        
        epoch = 1
        x_inTensor_Test_copy = np.copy(x_inTensor_Test) #If for pricing purposes, we use the train data, otherwise the test data
        all_actions = np.zeros([x_inTensor_Test.shape[0],x_inTensor_Test.shape[1],self.num_stocks])
        
        for t_ in range(0,x_inTensor_Test_copy.shape[1] - 1):
            # np.random.seed(epoch*t_)
            mIn = x_inTensor_Test_copy[:,t_,:]
            if self.No_wealth == True:
                mIn = mIn[:,0:-1]
            all_actions[:,t_,:] = sess.run(self.actorOut,feed_dict={self.mIn: mIn})        
            
        
        
        # #Copy the trained critic model into the test critic
        if useFullSavedModel == False:
            sess.run(self.match_onlineCritic_Test) 
            sess.run(self.match_targetCritic_Test) 
        
        
        while epoch <= int(self.epochs/2):   
            
            if useFullSavedModel == True:
                break
            
            t_ = np.random.randint(0,x_inTensor_Test_copy.shape[1] - 1)
            
            samples_idx = np.random.choice(x_inTensor_Test_copy.shape[0], self.minibatchSize, replace=False)
            samples_inTensor_Test = x_inTensor_Test_copy[samples_idx]
            samples_price_mat_Test = price_mat_Test[samples_idx]  
            samples_actions_Test = all_actions[samples_idx]             
            
            samples = samples_inTensor_Test[:,t_,:]
            samples_next = samples_inTensor_Test[:,t_+1,:]
            sample_prices = samples_price_mat_Test[:,t_,:]
            sample_prices_next = samples_price_mat_Test[:,t_+1,:]  
            
            
            # np.random.seed(epoch*t_)
            mIn = samples
            mIn_next = samples_next
            
            
            mIn = mIn[:,0:-1]
            mIn_next = mIn_next[:,0:-1]
            
            action = np.copy(samples_actions_Test[:,t_,:])            
            rnd_choice = np.random.choice([0,1], action.shape, p=[1-exploration_prob,exploration_prob]) 
            action = np.clip(action + np.random.normal(0,.05,action.shape)*rnd_choice,-1,1)
            
           
            action_target = np.copy(samples_actions_Test[:,t_+1,:])# sess.run(self.actorOut, feed_dict={self.mIn: mIn_next})
            Q_target = sess.run(self.criticOut_target_Test,
                                      feed_dict={self.mIn_critic_target_Test: mIn_next,
                                                self.actionIn_target_Test:action_target})            
        
            rews = np.sum(-action * (sample_prices_next-sample_prices),1,keepdims=True) + Q_target
            
            
            if t_ == x_inTensor_Test_copy.shape[1] - 2:
                if short == 1:
                    rews[:,0] = np.maximum(np.mean(sample_prices_next,1) - self.strike,0) - np.sum(action * (sample_prices_next-sample_prices),axis=1)
                else:
                    rews[:,0] = - np.maximum(np.mean(sample_prices_next,1) - self.strike,0) - np.sum(action * (sample_prices_next-sample_prices),axis=1)

            #Update the online critic
            lr = self.learningRate[0]/10
            sess.run(self.criticOptimiser_Test,
                        feed_dict={self.mIn_critic_Test: mIn,
                                    self.actionIn_Test: action,
                                    self.value_target_Test: rews,
                                    self.lr_critic_Test: lr})
       
            
            if epoch % target_net_update_episodes == 0:   
                sess.run(self.update_targetCritic_Test, feed_dict={self.tau_c_value_Test: self.tau_c})
                        
                        
            epoch += 1
            
            
            
            if epoch % 10000 == 0:
                
                mIn = x_inTensor_Test[0:1,0,:]
                if self.No_wealth == True:
                        mIn = mIn[:,0:-1]
                action = all_actions[0:1,0,:]# sess.run(self.actorOut,feed_dict={self.mIn: mIn})
                # Q = sess.run(self.criticOut_target_Test,feed_dict={self.mIn_critic_target_Test: mIn,self.actionIn_target_Test: action})
                Q = sess.run(self.criticOut_Test,feed_dict={self.mIn_critic_Test: mIn,self.actionIn_Test: action})
                Q_0_Test.append(Q[0])
                
                fig = plt.figure()
                plt.plot(Q_0_Test)
                plt.savefig(self.save_path+'AC/Q_0_Test'+str(short)+'.png')
                plt.close(fig)
        
        # #######################################################################################  
        
        # # Save the model 
        if useFullSavedModel == False:
            self.saver.save(sess, self.save_path+'AC/MODELS/'+str(self.num_stocks)+'/'+str(short)+'/Full/model.ckpt')
        
        
        #Run the Actor-Only model to find the optimal strategy
        #######################################################################################
        for epoch in range(60000 + 1):
            
            if useFullSavedModel == True:
                break
            
            
            samples_idx = np.random.choice(x_inTensor.shape[0], self.minibatchSize_pg, replace=False)
            samples = x_inTensor[samples_idx] 
            
            sess.run(self.actorOptimiser_adam,
                          feed_dict={self.mIn_pg: samples[:,:,0:-1], 
                                    self.lr_pg: self.learningRate_pg,
                                    self.short: short})
                    
            
            if epoch%10000 == 0:
                rew,deltas = sess.run([self.reward,self.deltas],
                                    feed_dict={self.mIn_pg: x_inTensor[:,:,0:-1], 
                                                self.lr_pg: self.learningRate_pg,
                                                self.short: short})
                  
                
                hedgingError_expectile_temp = np.expand_dims([self.getExpectile(rew,1e-5, exp_value) for exp_value in expectile_value_range],axis=0)
                
                
                
                print('epoch       : ' + str(epoch))
                print('exp_In is   : ' + str(hedgingError_expectile_temp[0,0]))
                print('-------------------------------------------')
        
        # #######################################################################################
        
        # # Save the model 
        if useFullSavedModel == False:
            self.saver.save(sess, self.save_path+'AC/MODELS/'+str(self.num_stocks)+'/'+str(short)+'/Full/model.ckpt')       
        
        
        
        # Run the model over test data with fixed actions (AO actions) to find the value of Q over test data (Notice the line below)
        # If this is used for pricing purposes, we train the Q function over the train data instead of test data
        #######################################################################################
        
        epoch = 1
        x_inTensor_Test_copy = np.copy(x_inTensor_Test) #If for pricing purposes, we use the train data, otherwise the test data
        
        
        all_actions = np.zeros([x_inTensor_Test_copy.shape[0],x_inTensor_Test_copy.shape[1],self.num_stocks])        
        
        all_actions[:,0:-1,:] = sess.run(self.deltas,feed_dict={self.mIn_pg: x_inTensor_Test_copy[:,:,0:-1], 
                                                self.lr_pg: self.learningRate_pg,
                                                self.short: short})      
            
        
       
        
        #Copy the trained critic model into the test critic
        if useFullSavedModel == False:
            sess.run(self.match_onlineCritic_Test_pg) 
            sess.run(self.match_targetCritic_Test_pg) 
        
        
        while epoch <= int(self.epochs/2):   
            
            if useFullSavedModel == True:
                break
            
            t_ = np.random.randint(0,x_inTensor_Test_copy.shape[1] - 1)
            
            samples_idx = np.random.choice(x_inTensor_Test_copy.shape[0], self.minibatchSize, replace=False)
            samples_inTensor_Test = x_inTensor_Test_copy[samples_idx]
            samples_price_mat_Test = price_mat_Test[samples_idx]  
            samples_actions_Test = all_actions[samples_idx]             
            
            samples = samples_inTensor_Test[:,t_,:]
            samples_next = samples_inTensor_Test[:,t_+1,:]
            sample_prices = samples_price_mat_Test[:,t_,:]
            sample_prices_next = samples_price_mat_Test[:,t_+1,:]  
            
            
            # np.random.seed(epoch*t_)
            mIn = samples
            mIn_next = samples_next
            
            
            mIn = mIn[:,0:-1]
            mIn_next = mIn_next[:,0:-1]
            
            action = np.copy(samples_actions_Test[:,t_,:])            
            rnd_choice = np.random.choice([0,1], action.shape, p=[1-exploration_prob,exploration_prob]) 
            action = np.clip(action + np.random.normal(0,.05,action.shape)*rnd_choice,-1,1)
            
           
            action_target = np.copy(samples_actions_Test[:,t_+1,:])# sess.run(self.actorOut, feed_dict={self.mIn: mIn_next})
            Q_target = sess.run(self.criticOut_target_Test_pg,
                                      feed_dict={self.mIn_critic_target_Test_pg: mIn_next,
                                                self.actionIn_target_Test_pg:action_target})            
        
            rews = np.sum(-action * (sample_prices_next-sample_prices),1,keepdims=True) + Q_target
            
            
            if t_ == x_inTensor_Test_copy.shape[1] - 2:
                if short == 1:
                    rews[:,0] = np.maximum(np.mean(sample_prices_next,1) - self.strike,0) - np.sum(action * (sample_prices_next-sample_prices),axis=1)
                else:
                    rews[:,0] = - np.maximum(np.mean(sample_prices_next,1) - self.strike,0) - np.sum(action * (sample_prices_next-sample_prices),axis=1)

            #Update the online critic
            lr = self.learningRate[0]/10
            sess.run(self.criticOptimiser_Test_pg,
                        feed_dict={self.mIn_critic_Test_pg: mIn,
                                    self.actionIn_Test_pg: action,
                                    self.value_target_Test_pg: rews,
                                    self.lr_critic_Test_pg: lr})
       
            
            if epoch % target_net_update_episodes == 0:   
                sess.run(self.update_targetCritic_Test_pg, feed_dict={self.tau_c_value_Test_pg: self.tau_c})
                        
                        
            epoch += 1
            
            
            
            if epoch % 10000 == 0:
                
                mIn = x_inTensor_Test_copy[0,0:1,:]
                if self.No_wealth == True:
                    mIn = mIn[:,0:-1]
                
                action = all_actions[0:1,0,:]# sess.run(self.actorOut,feed_dict={self.mIn: mIn})
                # Q = sess.run(self.criticOut_target_Test_pg,feed_dict={self.mIn_critic_target_Test_pg: mIn,self.actionIn_target_Test_pg: action})
                Q = sess.run(self.criticOut_Test_pg,feed_dict={self.mIn_critic_Test_pg: mIn,self.actionIn_Test_pg: action})
                Q_0_Test_pg.append(Q[0])
                
                fig = plt.figure()
                plt.plot(Q_0_Test_pg)
                plt.savefig(self.save_path+'AC/Q_0_Test_pg'+str(short)+'.png')
                plt.close(fig)
        
        #######################################################################################  
        
        
        # # Save the model 
        if useFullSavedModel == False:
            self.saver.save(sess, self.save_path+'AC/MODELS/'+str(self.num_stocks)+'/'+str(short)+'/Full/model.ckpt')
        
        
        #Computing the dynamic loss for different time to maturities using AO model
        #######################################################################################
        loss_time_AO = []
        x_inTensor_Test_copy = np.copy(x_inTensor_Test)
        
        t_counter = 0
        for t_ in range(x_inTensor_Test.shape[1]-1):
        
            time_to_mat = np.zeros(self.n_timesteps + 1)
            time_to_mat[t_ + 1:] = self.T / (self.n_timesteps)  # [0,0,0,h,..,h]
            time_to_mat = np.cumsum(time_to_mat)  # [0,0,0,h,2h...,(N-starting_t)h]
            time_to_mat = time_to_mat[::-1]  # [(N-starting_t)h, (N-starting_t-1)h,...,h,0,0,0]
            x_inTensor_Test_copy[:,:,-2] = np.reshape(time_to_mat,[1,self.n_timesteps+1])
            
            all_actions = sess.run(self.deltas,feed_dict={self.mIn_pg: x_inTensor_Test_copy[:,:,0:-1],
            self.lr_pg: self.learningRate_pg,
            self.short: short})
            
            mIn = x_inTensor_Test_copy[0:1,0,:]
            if self.No_wealth == True:
                mIn = mIn[:,0:-1]
            
            
            action = all_actions[0:1,0,:]
            Q = sess.run(self.criticOut_Test_pg,feed_dict={self.mIn_critic_Test_pg: mIn,self.actionIn_Test_pg: action})
            # Q = sess.run(self.criticOut,feed_dict={self.mIn_critic: mIn,self.actionIn: action})
            loss_time_AO.append(Q[0,0])
            t_counter = t_counter + 1
            
            
        
        
        #Computing the static loss for different time to maturities of the AO policy
        #######################################################################################
        loss_time_AO_static = []
        t_counter = 0
        for starting_t in range(x_inTensor_Test.shape[1]-1):
            x_inTensor_Test_copy = np.copy(x_inTensor_Test)
            loss_time_AO_static.append(self.evaluateStaticRisk_AO(sess,x_inTensor_Test_copy,price_mat_Test,starting_t,short))
        
        
        
        #Computing the dynamic loss for different time to maturities of the AC policy using the Q function
        #######################################################################################                             
        loss_time_AC = []        
        
        for t_ in x_inTensor_Test[0,0:-1,-2]:
            mIn = np.copy(x_inTensor_Test[0,0:1,:])
            if self.No_wealth == True:
                mIn = mIn[:,0:-1]
            mIn[:,-1] = t_
            action = sess.run(self.actorOut,feed_dict={self.mIn: mIn})
            Q = sess.run(self.criticOut_Test,feed_dict={self.mIn_critic_Test: mIn,self.actionIn_Test: action})
            # Q = sess.run(self.criticOut_target,feed_dict={self.mIn_critic_target: mIn,self.actionIn_target: action})            
            loss_time_AC.append(Q[0,0])
        
        
        #Computing the dynamic loss for different time to maturities of the AC policy using the DP model
        #######################################################################################     
        if self.num_stocks == 1:
            loss_time_AC_using_DP = 0 #self.evaluateDynamicRisk_AC(sess, x_inTensor,x_inTensor_Test,price_mat,price_mat_Test,expectile_value_range,short)
        else:
            loss_time_AC_using_DP = 0
            
            
        #Computing the static loss for different time to maturities of the AC policy
        #######################################################################################
        loss_time_AC_static = []
        t_counter = 0
        for starting_t in range(x_inTensor_Test.shape[1]-1):
            x_inTensor_Test_copy = np.copy(x_inTensor_Test)
            loss_time_AC_static.append(self.evaluateStaticRisk_AC(sess,x_inTensor_Test_copy,price_mat_Test,starting_t,short))
        
        
        if self.pMeasure == PerformanceMeasure.MSE:
            out1 = hedgingError_mean[-1]
            out2 = hedgingError_mean_Test[-1]
        elif self.pMeasure == PerformanceMeasure.CVaR:
            out1 = hedgingError_cvar[-1]
            out2 = hedgingError_cvar_Test[-1]
        elif self.pMeasure == PerformanceMeasure.Expectile:
            out1 = hedgingError_expectile[-1,0]
            out2 = hedgingError_expectile_Test[-1,0]
        
        
        # Starting point to plot the graphs
        s_point = 0
        
        # Plot the results for static risk
        #######################################################################################
        fig = plt.figure()
        if short == 1:
            loss_time_AC_static = list(np.array(loss_time_AC_static))# - .05)
        else:
            loss_time_AC_static = list(np.array(loss_time_AC_static))# - .1)
            
        loss_time_AC_static = np.append(loss_time_AC_static,0).tolist()
        loss_time_AO_static = np.append(loss_time_AO_static,0).tolist()
        ll = len(loss_time_AO_static)
        plt.plot(np.arange(s_point,x_inTensor.shape[1],1), loss_time_AC_static[0:ll-s_point], label='DRM policy', marker='o')
        plt.plot(np.arange(s_point,x_inTensor.shape[1],1),loss_time_AO_static[0:ll-s_point], label='SRM policy', marker='o')
        plt.xticks(np.arange(s_point,x_inTensor.shape[1],1),np.arange(x_inTensor.shape[1]-1,s_point-1,-1))
        plt.xlabel("Maturity (in months)")
        plt.ylabel("Static " + str(int(self.expectile_value*100)) + "%-expectile hedging risk")
        plt.legend()
        plt.savefig(self.save_path+'AC/staticHedgingRisk'+str(short)+'.png')
        plt.close(fig)
        loss_time_AC_static = loss_time_AC_static[0:ll-s_point]
        loss_time_AO_static = loss_time_AO_static[0:ll-s_point]
        
        # Plot the results for dynamic risk
        #######################################################################################
        fig = plt.figure()
        if short == -1:
            loss_time_AC = list(np.array(loss_time_AC))# - 0.4)
            loss_time_AO = list(np.array(loss_time_AO))# - 0.4)
        else:
            loss_time_AC = list(np.array(loss_time_AC))
            loss_time_AO = list(np.array(loss_time_AO))# + np.array([0,0.1,0.2,0.3,0.5,0.7,0.8,0.7,0.6,0.6,0.2]))
            
        loss_time_AC = np.append(loss_time_AC,0).tolist()
        loss_time_AO = np.append(loss_time_AO,0).tolist() 
        if self.num_stocks == 1 and loss_time_AC_using_DP != 0:
            loss_time_AC_using_DP = np.append(loss_time_AC_using_DP,0).tolist()
        ll = len(loss_time_AC)
        plt.plot(np.arange(s_point,x_inTensor.shape[1],1), loss_time_AC[0:ll-s_point], label='RL based estimation', marker='o')
        if self.num_stocks == 1 and loss_time_AC_using_DP != 0:
            plt.plot(np.arange(s_point,x_inTensor.shape[1],1), loss_time_AC_using_DP[0:ll-s_point], label='DP based estimation', marker='o')
        # plt.plot(np.arange(s_point,x_inTensor.shape[1],1), loss_time_AO[0:ll-s_point], label='SRM policy', marker='o')
        plt.xticks(np.arange(s_point,x_inTensor.shape[1],1),np.arange(x_inTensor.shape[1]-1,s_point-1,-1))
        plt.xlabel("Maturity (in months)")
        plt.ylabel("Dynamic " + str(int(self.expectile_value*100)) + "%-expectile hedging risk")
        plt.legend()
        plt.savefig(self.save_path+'AC/dynamicHedgingRisk'+str(short)+'.png')
        plt.close(fig)
        loss_time_AC = loss_time_AC[0:ll-s_point]
        loss_time_AO = loss_time_AO[0:ll-s_point]
        
        return out1, out2, Q_0, Q_0_Train, Q_0_Test,np.round(loss_time_AC,2), np.round(loss_time_AO,2),np.round(loss_time_AC_static,2),np.round(loss_time_AO_static,2), np.round(loss_time_AC_using_DP,2)
    
    
    
    def evaluateDynamicRisk_DP(self, x_inTensor,x_inTensor_Test,price_mat,price_mat_Test,all_actions,expectile_value_range,short):        
    
        grid_s_max = 1.1*np.max(price_mat_Test)
        grid_s_min = 0.9*np.min(price_mat_Test)
        grid_s_disc = self.DP_desc/2
        g1 = np.arange(grid_s_min,self.S_0,(self.S_0-grid_s_min)/grid_s_disc)
        g2 = np.arange(self.S_0[0],grid_s_max,(grid_s_max-self.S_0[0])/grid_s_disc)
        grid_s = np.concatenate([g1,g2])
        V_t = np.zeros(shape=[grid_s.shape[0],price_mat_Test.shape[1]])
        
        
        #Computing the dynamic loss for different time to maturities using DP model
        #######################################################################################
        loss_time_DP = []
        x_inTensor_Test_copy = np.copy(x_inTensor_Test)        
        
        t_counter = 0
        for t_start in range(x_inTensor_Test.shape[1]-1):            
            
            a_t = all_actions[:,t_start:]
            
            seed_counter = 0
            for t_ in range(price_mat_Test.shape[1]-1-t_start,-1,-1):                
                
                for s_idx in range(grid_s.shape[0]):                
                    if t_ == price_mat_Test.shape[1]-1-t_start:
                        if short == 1:
                            V_t[s_idx,t_] = np.maximum(grid_s[s_idx] - self.strike,0)
                        else:
                            V_t[s_idx,t_] = -np.maximum(grid_s[s_idx] - self.strike,0)
                        
                    else:
                        # compute the possible values of price in the next period
                        ###############################################################                    
                        np.random.seed(seed_counter)
                        seed_counter = seed_counter + 1
                        rand_stdnorm = np.random.randn(self.DP_desc)
                        h = self.T / self.n_timesteps
                        prices = grid_s[s_idx] * np.exp((self.mu - self.sigma ** 2 / 2) * h + self.sigma * np.sqrt(h) * rand_stdnorm)
                        max_next_price =np.max(prices)
                        min_next_price =np.min(prices)
                        ###############################################################
                        next_s_idx =[np.argmin(np.abs(grid_s-prices[i])) for i in range(len(prices))] #(grid_s <= max_next_price) & (grid_s >= min_next_price)
                        next_s_val = [grid_s[i] for i in next_s_idx]
                        
                        # Finding closest price in test sample trajectories to the current grid price
                        action_idx = np.argmin(np.abs(grid_s - price_mat_Test[s_idx,t_]))
                        action = a_t[action_idx,t_]
                        
                        V_t_temp = [V_t[i,t_+1] for i in next_s_idx]
                        if len(next_s_val) == 0:
                            print('naaaaaaan')
                        if self.pMeasure == PerformanceMeasure.Expectile:
                            V_t_all = np.array(V_t_temp  - action * (next_s_val - grid_s[s_idx]))                        
                            L_expectile_value = self.getExpectile(V_t_all,1e-5)
                            V_t[s_idx,t_] = L_expectile_value
                print(t_)
            
            
            V_t_idx = np.argmin(np.abs(grid_s - self.S_0[0]))
            loss_time_DP.append(V_t[V_t_idx,0])
            t_counter = t_counter + 1
                          
        return loss_time_DP
    
    
    
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    def train_test_approximate(self, sess, x_inTensor,x_inTensor_Test,short=1):        
        #Creating the price matrix
        price_mat = np.exp(x_inTensor[:, :, 0:-2]) * np.reshape(self.S_0,[1,1,self.num_stocks])
        price_mat_Test = np.exp(x_inTensor_Test[:, :, 0:-2]) * np.reshape(self.S_0,[1,1,self.num_stocks])
        
        hedgingError_mean = []
        hedgingError_cvar = []
        hedgingError_max = []
        hedgingError_min = []
        
        hedgingError_mean_Test = []
        hedgingError_cvar_Test = []
        hedgingError_max_Test = []
        hedgingError_min_Test = []
        
        expectile_value_range = np.arange(self.expectile_value,1,(1-self.expectile_value)/10)
        
    
        grid_s_max = 1.1*np.max(price_mat)
        grid_s_min = 0.9*np.min(price_mat)
        grid_s_disc = self.DP_desc/2
        g1 = np.arange(grid_s_min,self.S_0,(self.S_0-grid_s_min)/grid_s_disc)
        g2 = np.arange(self.S_0[0],grid_s_max,(grid_s_max-self.S_0[0])/grid_s_disc)
        grid_s = np.concatenate([g1,g2])
        V_t = np.zeros(shape=[grid_s.shape[0],price_mat.shape[1]])
        a_t = np.zeros(shape=[grid_s.shape[0],price_mat.shape[1]])
        
        seed_counter = 0
        for t_ in range(price_mat.shape[1]-1,-1,-1):
            for s_idx in range(grid_s.shape[0]):                
                if t_ == price_mat.shape[1]-1:
                    if short == 1:
                        V_t[s_idx,t_] = np.maximum(grid_s[s_idx] - self.strike,0)
                    else:
                        V_t[s_idx,t_] = -np.maximum(grid_s[s_idx] - self.strike,0)
                    
                else:
                    # compute the possible values of price in the next period
                    ###############################################################
                    # distance = np.abs(grid_s[s_idx] - price_mat[:,t_])
                    # nearest_idx = distance <= np.sort(distance)[int(.01*len(distance))]
                    # max_next_price = np.max(price_mat[nearest_idx,t_+1])
                    # min_next_price = np.min(price_mat[nearest_idx,t_+1])
                    
                    np.random.seed(seed_counter)
                    seed_counter = seed_counter + 1
                    rand_stdnorm = np.random.randn(self.DP_desc)
                    h = self.T / self.n_timesteps
                    prices = grid_s[s_idx] * np.exp((self.mu - self.sigma ** 2 / 2) * h + self.sigma * np.sqrt(h) * rand_stdnorm)
                    max_next_price =np.max(prices)
                    min_next_price =np.min(prices)
                    ###############################################################
                    next_s_idx =[np.argmin(np.abs(grid_s-prices[i])) for i in range(len(prices))] #(grid_s <= max_next_price) & (grid_s >= min_next_price)
                    next_s_val = [grid_s[i] for i in next_s_idx]
                    action_disc = np.arange(-1,1+.1,.1)
                    V_t_temp = [V_t[i,t_+1] for i in next_s_idx]
                    if len(next_s_val) == 0:
                        print('naaaaaaan')
                    if self.pMeasure == PerformanceMeasure.CVaR:
                        V_t_all = np.array([V_t_temp  - a * (next_s_val - grid_s[s_idx]) for a in action_disc])
                        V_t_cvar = [np.mean(np.sort(V_t_all[i,:])[int(self.expectile_value*len(next_s_val)):]) for i in range(len(action_disc))]                    
                        V_t[s_idx,t_] = np.min(V_t_cvar)
                        a_t[s_idx,t_] = action_disc[np.argmin(V_t_cvar)]
                    elif self.pMeasure == PerformanceMeasure.MSE:
                        V_t_all = np.array([V_t_temp  - a * (next_s_val - grid_s[s_idx]) for a in action_disc])
                        V_t_mean = np.mean(V_t_all,1)
                        V_t[s_idx,t_] = np.min(V_t_mean)
                        a_t[s_idx,t_] = action_disc[np.argmin(V_t_mean)]
                    elif self.pMeasure == PerformanceMeasure.Expectile:
                        V_t_all = np.array([V_t_temp  - a * (next_s_val - grid_s[s_idx]) for a in action_disc])
                        # L_expectile = [[np.mean(self.expectile_value*np.maximum(V_t_all[i,:]-b,0.0)**2 + (1-self.expectile_value)*np.maximum(b - V_t_all[i,:],0.0)**2) for b in V_t_all[i,:]] for i in range(len(action_disc))]
                        # L_expectile = np.array(L_expectile)
                        # L_expectile_value = [V_t_all[i,np.argmin(L_expectile[i,:])] for i in range(len(action_disc))]
                        # V_t[s_idx,t_] = np.min(L_expectile_value)
                        # a_t[s_idx,t_] = action_disc[np.argmin(L_expectile_value)]
                        
                        L_expectile_value = [self.getExpectile(V_t_all[i,:],1e-5) for i in range(len(action_disc))]  
                        V_t[s_idx,t_] = np.min(L_expectile_value)
                        a_t[s_idx,t_] = action_disc[np.argmin(L_expectile_value)]
                        
                        # L_expectile = [minimize(self.expectile_function, V_t_all[i,0],args=(self.expectile_value,V_t_all[i,:])) for i in range(len(action_disc))]                        
                        # L_expectile_value = [L_expectile[i].fun for i in range(len(action_disc))]                        
                        # V_t[s_idx,t_] = L_expectile[np.argmin(L_expectile_value)].x[0]
                        # a_t[s_idx,t_] = action_disc[np.argmin(L_expectile_value)]
            print(t_)
        
      
        
        # Compute the in sample performance 
        W_t = np.zeros(shape=price_mat.shape)
        HE_T = np.zeros(shape=price_mat.shape[0])
        action = np.zeros(shape=price_mat.shape)
        for t_ in range(price_mat.shape[1]):
            for s_idx in range(price_mat.shape[0]):                 
                if t_ == price_mat.shape[1]-1:
                    if short == 1:
                        HE_T[s_idx] = np.maximum(price_mat[s_idx,t_,0] - self.strike,0) - W_t[s_idx,t_]
                    else:
                        HE_T[s_idx] = -np.maximum(price_mat[s_idx,t_,0] - self.strike,0) - W_t[s_idx,t_]
                else:                
                    action_idx = np.argmin(np.abs(grid_s - price_mat[s_idx,t_,0]))
                    action[s_idx,t_] = a_t[action_idx,t_]
                    W_t[s_idx,t_+1] = W_t[s_idx,t_] + action[s_idx,t_] * (price_mat[s_idx,t_+1,0] - price_mat[s_idx,t_,0])
                    
                
        # Compute the out of sample performance
        W_t_test = np.zeros(shape=price_mat_Test.shape)
        HE_T_test = np.zeros(shape=price_mat_Test.shape[0])
        action_test = np.zeros(shape=price_mat_Test.shape)
        for t_ in range(price_mat_Test.shape[1]):
            for s_idx in range(price_mat_Test.shape[0]):                 
                if t_ == price_mat_Test.shape[1]-1:
                    if short == 1:
                        HE_T_test[s_idx] = np.maximum(price_mat_Test[s_idx,t_] - self.strike,0) - W_t_test[s_idx,t_]
                    else:
                        HE_T_test[s_idx] = -np.maximum(price_mat_Test[s_idx,t_] - self.strike,0) - W_t_test[s_idx,t_]
                else:                
                    action_idx = np.argmin(np.abs(grid_s - price_mat_Test[s_idx,t_]))
                    action_test[s_idx,t_] = a_t[action_idx,t_]
                    W_t_test[s_idx,t_+1] = W_t_test[s_idx,t_] + action_test[s_idx,t_] * (price_mat_Test[s_idx,t_+1] - price_mat_Test[s_idx,t_])        
        
        
        #Computing the dynamic loss for different time to maturities using DP model
        #######################################################################################
        # loss_time_DP = self.evaluateDynamicRisk_DP(x_inTensor,x_inTensor_Test,price_mat,price_mat_Test,a_t,expectile_value_range,short)
        loss_time_DP = []        
        
        for t_ in range(price_mat.shape[1]):
            
            V_t_idx = np.argmin(np.abs(grid_s - self.S_0[0]))
            Q_DP = V_t[V_t_idx,t_]            
            loss_time_DP.append(Q_DP)
            
            
        #Computing the static loss for different time to maturities using DP model
        #######################################################################################
        loss_time_DP_static = []
        for starting_t in range(price_mat.shape[1] - 1):
            loss_time_DP_static.append(self.evaluateStaticRisk_DP(price_mat_Test,starting_t,short,a_t,grid_s))
        
        
        # fig = plt.figure()
        # plt.plot(np.arange(.5,1,.01),cvar)
        # # plt.legend()
        # plt.savefig('D:/Saeed/Equal_Risk_RL/EqualRisk/results/APP/cvar'+str(short)+'.png')
        # plt.close(fig)
        # print('------')
        
        fig = plt.figure()
        [plt.scatter(price_mat[:,i,0],action[:,i], label=str(i), s=1) for i in range(1,price_mat.shape[1]-1)]
        plt.plot([self.S_0[0],self.S_0[0]],[np.min(action),np.max(action)])
        plt.xlabel("Stock value")
        plt.ylabel("Investment (in  shares)")
        # plt.legend()
        plt.savefig(self.save_path+'APP/action_t_'+str(short)+'.png')
        plt.close(fig)
        print('------')
        
        # for i in range(1,price_mat.shape[1]-1):
        #     fig = plt.figure()
        #     plt.scatter(price_mat[:,i],action[:,i], label=str(i), s=1)
        #     # plt.legend()
        #     plt.savefig('D:/Saeed/Equal_Risk_RL/EqualRisk/results/APP/action_t_'+str(i)+str(short)+'.png')
        #     plt.close(fig)
        #     print('------')
        
        Q = V_t[np.argmin(np.abs(grid_s - price_mat[0,0,0])),0]
        if self.pMeasure == PerformanceMeasure.MSE:
            out1 = np.mean(HE_T)
            out2 = np.mean(HE_T_test)
            
        elif self.pMeasure == PerformanceMeasure.CVaR:
            out1 = np.mean(np.sort(HE_T)[int(self.expectile_value*len(HE_T)):])
            out2 = np.mean(np.sort(HE_T_test)[int(self.expectile_value*len(HE_T_test)):])
            
        elif self.pMeasure == PerformanceMeasure.Expectile:
            L_expectile_value = np.array([self.getExpectile(HE_T,1e-5)])
            out1 = L_expectile_value
            
            L_expectile_value = np.array([self.getExpectile(HE_T_test,1e-5)])
            out2 = L_expectile_value
                        
            # L_expectile = np.array([np.mean(self.expectile_value*np.maximum(HE_T-b,0.0)**2 + (1-self.expectile_value)*np.maximum(b - HE_T,0.0)**2) for b in HE_T])                        
            # out1 = HE_T[np.argmin(L_expectile)]
            
            # L_expectile = np.array([np.mean(self.expectile_value*np.maximum(HE_T_test-b,0.0)**2 + (1-self.expectile_value)*np.maximum(b - HE_T_test,0.0)**2) for b in HE_T_test])                        
            # out2 = HE_T_test[np.argmin(L_expectile)]
        
        out3 = np.max(HE_T)
        out4 = np.max(HE_T_test)
        
        out5 = np.mean(np.sort(HE_T)[int(self.expectile_value*len(HE_T)):])
        out6 = np.mean(np.sort(HE_T_test)[int(self.expectile_value*len(HE_T_test)):])
                          
        return out1, out2, Q, a_t, out3, out4, out5, out6, loss_time_DP, loss_time_DP_static
        
       

    
        
                
                     
            