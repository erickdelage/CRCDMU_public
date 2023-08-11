# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 14:07:03 2021

@author: laced
"""
import numpy as np
import pandas as pd
#import boto3
import os
from pandas import datetime
from sklearn import preprocessing
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import scipy.linalg

class loadData:

    def simulate_data(self):
        # Parameters
        self.num_stocks = 1
        mu =np.array([-0.001457745,-0.001698238,-6.74272E-05,0.000550385,-0.000396287])
        sigma = np.array([0.029755134,0.024296697,0.029509163,0.034521379,0.024571856])
        S_0 = [78.81,1877.94,221.77,137.25,1450.16]  # Initial stock price
        corr_matrix = [[1,0.713252821,0.774422565,0.538329077,0.768048243],
                       [0.713252821,	1,	0.69026091,	0.268519055,	0.683706513],
                       [0.774422565,	0.69026091,	1,	0.480734025,	0.805403846],
                       [0.538329077,	0.268519055,	0.480734025,	1,	0.606025566],
                       [0.768048243,	0.683706513,	0.805403846,	0.606025566,	1]]
        L_cholesky = scipy.linalg.cholesky(corr_matrix, lower=True)
        self.strike = 78.81    # One stock: 78.81   Multiple stocks: 753.186
        # self.strike = 753.186
        T = 60 / 260  # Time-to-maturity of the vanilla put option
        self.n_sims = 2000  # Total number of paths to simulate
        self.n_timesteps = 12  # Monthly hedging
        train_percentage = .5

        # Simulation of BSM dataset
        seed = 10
        h = T / self.n_timesteps  # step-size
        self.Price_mat_unnorm = np.zeros((self.n_sims, self.n_timesteps + 1,len(mu)))  # matrix of simulated stock prices
        self.Price_mat_unnorm[:, 0, :] = S_0
        np.random.seed(seed)
        rand_stdnorm = np.random.randn(self.n_sims,self.n_timesteps,len(mu))  
        rets = np.array(np.exp((mu - sigma ** 2 / 2) * h + sigma * np.sqrt(h) * rand_stdnorm)) - 1
        if self.num_stocks > 1:
            rets = np.array([np.dot(rets[i,:,:],L_cholesky) for i in range(rets.shape[0])])
        self.Price_mat_unnorm[:, 1:,:] = S_0 * np.cumprod(rets+1,1)

        # Apply a transformation to stock prices
        prepro_stock = "Log-moneyness"  # {Log, Log-moneyness, Nothing}
        if (prepro_stock == "Log"):
            Price_mat = np.log(self.Price_mat_unnorm)
        elif (prepro_stock == "Log-moneyness"):
            Price_mat =  np.log(self.Price_mat_unnorm / np.reshape(S_0,[1,1,len(S_0)]))
            # Price_mat =  np.log(self.Price_mat_unnorm / self.strike)

        # Construct the train and test sets
        # - The feature vector for now is [S_n, T-t_n]; the portfolio value V_{n} will be added further into the code at each time-step
        self.train_input = np.zeros((int(train_percentage * self.n_sims), self.n_timesteps + 1, self.num_stocks+2))
        
        test_input = np.zeros((int(np.round(1-train_percentage,3) * self.n_sims), self.n_timesteps + 1, self.num_stocks+2))
        # test_buffer     = np.zeros((self.n_timesteps+1, 100000,7))
        time_to_mat = np.zeros(self.n_timesteps + 1)
        time_to_mat[1:] = T / (self.n_timesteps)  # [0,h,h,h,..,h]
        # time_to_mat[1:] = 1
        time_to_mat = np.cumsum(time_to_mat)  # [0,h,2h,...,Nh]
        time_to_mat = time_to_mat[::-1]  # [Nh, (N-1)h,...,h,0]

        self.train_input[:, :,0:-2] = Price_mat[0:int(train_percentage * self.n_sims), :, 0:self.num_stocks]
        self.train_input[:, :,-2] = np.reshape(time_to_mat,[1,self.n_timesteps+1])
        test_input[:, :,0:-2] = Price_mat[int(train_percentage * self.n_sims):, :, 0:self.num_stocks]
        test_input[:, :,-2] = np.reshape(time_to_mat,[1,self.n_timesteps+1])

        return self.train_input,test_input,S_0[0:self.num_stocks],mu[0:self.num_stocks],sigma[0:self.num_stocks],T,self.n_timesteps,self.strike, self.num_stocks

