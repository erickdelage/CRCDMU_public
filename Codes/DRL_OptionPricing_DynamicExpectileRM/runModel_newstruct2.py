from Agent_tf_newstruct7 import Agent_tf
# from Agent import Agent_tf
from loadData3 import loadData
import argparse
import numpy as np
import pandas as pd
from enumClasses import PerformanceMeasure
# from hyperopt import hp
import tensorflow as tf
import matplotlib.pyplot as plt
import math


parser = argparse.ArgumentParser()
tf.compat.v1.disable_eager_execution()

parser.add_argument("--runMode",default=5) #1: run CV, 2: run test by csv file, 3: run test to choose seed, 4: run test from script

# parser.add_argument("--strike",default=100)
parser.add_argument("--modelType",default='approximate') #'pg' approximate ddpg
parser.add_argument("--RNN",default=0) #0
parser.add_argument("--window",default=1) #128
parser.add_argument("--net_depth",default=32) #6
parser.add_argument("--expectile_value",default=.9)
parser.add_argument("--learningRate_pg",default=5e-5) #1e-4
parser.add_argument("--learningRate",default=1e-3) 
#One stock   critic: 1e-3 & a ctor: 5e-6
#Five stocks critic: 1e-3 & actor: 1e-6

parser.add_argument("--epoch",default=100000) #40
parser.add_argument("--minibatchSize",default=200) #256
parser.add_argument("--minibatchSize_pg",default=500) #256



parser.add_argument("--maxpool",default=0)
parser.add_argument("--resnet",default='none') #none and const and random and padding
parser.add_argument("--decayRate",default=.9999) #.99099+
parser.add_argument("--minimmumRate",default=1e-5) #1e-5
parser.add_argument("--regularizer_multiplier",default=0) #1e-6
parser.add_argument("--reduce_max",default=0)
parser.add_argument("--optimizer",default='adam') #momentum and adam and rmsprop and sgd
parser.add_argument("--withCorrData",default=0)
parser.add_argument("--markowitzCoef",default=1) #1
parser.add_argument("--markowitzCoef_bench",default=1) #1
parser.add_argument("--excessReturn",default=0)
parser.add_argument("--firstLayer",default=0)
parser.add_argument("--lastLayer",default=0)
parser.add_argument("--tradeFee",default=0.0005) #.0005
parser.add_argument("--initial_param1",default=0) #0
parser.add_argument("--initial_param2",default=1) #2

parser.add_argument("--weightNormalization",default=0) #0
parser.add_argument("--AI_model",default='wavenet5') #lstm , iie , wavenet, wavenet2 ,cost_sensitive, cost_sensitive2
parser.add_argument("--shuffle",default=1) #1
parser.add_argument("--keep_prob",default=.8)
parser.add_argument("--skipConnection",default=1)

parser.add_argument("--days_markowitz",default=128) #1
parser.add_argument("--minibatchCount",default=120) #This must always be lower than days_markowitz



parser.add_argument("--pMeasure",default=0) #1
parser.add_argument("--noise",default=0) #.1
parser.add_argument("--exploit_noise",default=0) #1
parser.add_argument("--entropy_coef",default=0) #1


parser.add_argument("--tradeFeeMultiplier",default=0) #1e-3
parser.add_argument("--riskMultiplier",default=0) #1
parser.add_argument("--constrain_action",default=0)
parser.add_argument("--smootherValue",default=0)
parser.add_argument("--maxStockWeight",default=0.15)
parser.add_argument("--loadAllFeatures",default=0)
parser.add_argument("--loadModelFromSavedModels",default=0)
parser.add_argument("--saveModel",default=1)
parser.add_argument("--market_data",default='canada') #canada or usa or sp or coins
parser.add_argument("--dataSet",default='newData/') #newData/ or newData_all/ or selected/
parser.add_argument("--use_pre_pvm",default=0)
parser.add_argument("--seed",default=1120)



parser.add_argument("--stocks_to_invest",default=10) #.2
parser.add_argument("--loadAllData",default=0) #0
parser.add_argument("--trainingDays",default=2766)
parser.add_argument("--early_stopping",default=0) #0
parser.add_argument("--onlyTest",default=0)
parser.add_argument("--SR_horizon",default=32)
parser.add_argument("--trainedModelToUse",default=0)


args = parser.parse_args()
#  0.0026150904111835473

for key,value in args.__dict__.items():
    print(key + ' = ' + str(value))

print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')


# set the parameters
window=int(args.window)
minibatchSize = int(args.minibatchSize)
minibatchSize_pg = int(args.minibatchSize_pg)
SR_horizon = int(args.SR_horizon)
learningRate_pg = float(args.learningRate_pg)
learningRate = float(args.learningRate)
# strike = float(args.strike)
minibatchCount = int(args.minibatchCount)
epochs = int(args.epoch)
trainingDays=int(args.trainingDays)
pMeasure=PerformanceMeasure(int(args.pMeasure))
modelType=args.modelType
runMode=int(args.runMode)
loadModelFromSavedModels=bool(int(args.loadModelFromSavedModels))
onlyTest=bool(int(args.onlyTest))
trainedModelToUse=str(args.trainedModelToUse)
markowitzCoef = float(args.markowitzCoef)
markowitzCoef_bench = float(args.markowitzCoef_bench)
saved_models_path = 'savedModels'
net_depth = int(args.net_depth)
early_stopping = bool(int(args.early_stopping))
loadAllData = bool(int(args.loadAllData))
decayRate = float(args.decayRate)
initial_param1 = float(args.initial_param1)
initial_param2 = float(args.initial_param2)
stocks_to_invest = int(args.stocks_to_invest)
days_markowitz = int(args.days_markowitz)
tradeFee = float(args.tradeFee)
optimizer=str(args.optimizer)
resnet=str(args.resnet)
maxpool=bool(int(args.maxpool))
minimmumRate=float(args.minimmumRate)
tradeFeeMultiplier=float(args.tradeFeeMultiplier)
constrain_action=bool(int(args.constrain_action))
riskMultiplier=float(args.riskMultiplier)
skipConnection=bool(int(args.skipConnection))
excessReturn=bool(int(args.excessReturn))
withCorrData=bool(int(args.withCorrData))
smootherValue=float(args.smootherValue)
maxStockWeight=float(args.maxStockWeight)
loadAllFeatures=bool(int(args.loadAllFeatures))
saveModel=bool(int(args.saveModel))
reduce_max=int(args.reduce_max)
market_data=str(args.market_data)
use_pre_pvm=bool(int(args.use_pre_pvm))
RNN=bool(int(args.RNN))
seed=int(args.seed)
firstLayer=int(args.firstLayer)
lastLayer=int(args.lastLayer)
regularizer_multiplier=float(args.regularizer_multiplier)
keep_prob=float(args.keep_prob)
dataSet=str(args.dataSet)
AI_model=str(args.AI_model)
weightNormalization=bool(int(args.weightNormalization))
shuffle=bool(int(args.shuffle))
noise=float(args.noise)
exploit_noise=float(args.exploit_noise)
entropy_coef=float(args.entropy_coef)
expectile_value = float(args.expectile_value)

dataLoader = loadData()
x,x_Test,S_0,mu,sigma,T,n_timesteps,strike,num_stocks=dataLoader.simulate_data()

print('==================== Shape of X: {0}'.format(str(x.shape)))


def def_agent(S_0,mu,sigma,T,n_timesteps,num_stocks,pMeasure=PerformanceMeasure.Expectile,loadModelFromSavedModels=False):
    agent = Agent_tf(learningRate,learningRate_pg, minibatchCount, minibatchSize,minibatchSize_pg,
                         epochs, window, SR_horizon,
                         saved_models_path,
                         x, pMeasure, markowitzCoef, modelType=modelType,
                         net_depth=net_depth,
                         decayRate=decayRate, initial_param1=initial_param1,
                         initial_param2=initial_param2, tradeFee=tradeFee,
                         optimizerType=optimizer, resnetType=resnet, includemaxpool=maxpool,
                         minimmumRate=minimmumRate, tradeFeeMultiplier=tradeFeeMultiplier,
                         constrain_action=constrain_action, riskMultiplier=riskMultiplier,
                         skipConnection=skipConnection, excessReturn=excessReturn,
                         withCorrData=withCorrData, smootherValue=smootherValue,
                         maxStockWeight=maxStockWeight, runMode=runMode, reduce_max=reduce_max,
                         seed=seed, firstLayer=firstLayer,
                         regularizer_multiplier=regularizer_multiplier, lastLayer=lastLayer,
                         RNN=RNN,keep_prob=keep_prob,weightNormalization=weightNormalization,
                         days_markowitz=days_markowitz,AI_model=AI_model,
                         entropy_coef=entropy_coef,strike=strike,expectile_value=expectile_value,
                         S_0=S_0,mu=mu,sigma=sigma,T=T,n_timesteps=n_timesteps,num_stocks=num_stocks,loadModelFromSavedModels=loadModelFromSavedModels)
    return agent


if modelType == 'pg':
    
    if expectile_value == .5:
        agent = def_agent(S_0,mu,sigma,T,n_timesteps,num_stocks,pMeasure=PerformanceMeasure.MSE)
    else:
        agent = def_agent(S_0,mu,sigma,T,n_timesteps,num_stocks,pMeasure=PerformanceMeasure.Expectile)
        
    with tf.compat.v1.Session() as sess:        
        long_loss_time = agent.train_test_pg(sess, x, x_Test, -1) 
    
    if expectile_value == .5:
        agent = def_agent(S_0,mu,sigma,T,n_timesteps,num_stocks,pMeasure=PerformanceMeasure.MSE)
    else:
        agent = def_agent(S_0,mu,sigma,T,n_timesteps,num_stocks,pMeasure=PerformanceMeasure.Expectile)
    with tf.compat.v1.Session() as sess:        
        short_loss_time = agent.train_test_pg(sess, x, x_Test, 1)
    
    
    
    
    
    
    
    
    
        
        
   
    
    
    # print(short_exp)
    # print(long_exp)
    
    # print(short_exp_test)
    # print(long_exp_test)    
    
    # print(short_cvar)
    # print(long_cvar)
    
    # print(short_cvar_test)
    # print(long_cvar_test)   
    
    # print(short_max)
    # print(long_max)
    
    # print(short_max_test)
    # print(long_max_test) 
    
    print(short_loss_time)
    print(long_loss_time)
    
    
    # ERP = (short_HE + long_HE)/2
    
elif modelType == 'ddpg':
    
    # agent = def_agent(pMeasure=PerformanceMeasure.Expectile)   
    # with tf.compat.v1.Session() as sess:
    #     perf_in, perf_out,HE1, V_0 = agent.train_test_ddpg(sess, x, x_Test,-1)
        
    # agent = def_agent(pMeasure=PerformanceMeasure.MSE)
    # with tf.compat.v1.Session() as sess:
    #     perf_in, perf_out,HE2, V_0 = agent.train_test_ddpg(sess, x, x_Test,-1)
    
    # agent = def_agent(pMeasure=PerformanceMeasure.MSE)
    # with tf.compat.v1.Session() as sess:
    #     perf_in, perf_out,HE3, V_0 = agent.train_test_ddpg(sess, x, x_Test,-1)
    
    # fig = plt.figure()
    # plt.plot(HE1,label='Expectile1')
    # plt.plot(HE2,label='MSE1')
    # plt.plot(HE3,label='MSE2')
    # plt.legend()
    # plt.savefig('D:/Saeed/Equal_Risk_RL/EqualRisk/results/AC/Q_0.png')
    
    
    # plt.close(fig)
    
    epochs = 500000 ## ## 2000000
    if expectile_value == .5:
        agent = def_agent(S_0,mu,sigma,T,n_timesteps,num_stocks,pMeasure=PerformanceMeasure.MSE)
    else:
        agent = def_agent(S_0,mu,sigma,T,n_timesteps,num_stocks,pMeasure=PerformanceMeasure.Expectile,loadModelFromSavedModels=False)
    with tf.compat.v1.Session() as sess:
        perf_in_short, perf_out_short, short_HE,short_HE_Train, short_HE_Test, short_loss_time_AC,\
        short_loss_time_AO,short_loss_time_AC_static,short_loss_time_AO_static, short_loss_time_AC_using_DP = agent.train_test_ddpg(sess, x, x_Test,1,useFullSavedModel=False)
        
        
    
    
    print(short_HE[-1])
    print(perf_in_short)
    print(perf_out_short)
    print(short_HE_Train[-1])
    print(short_HE_Test[-1])
    print(short_loss_time_AC)
    print(short_loss_time_AO)
    print(short_loss_time_AC_static)
    print(short_loss_time_AO_static)
    print(short_loss_time_AC_using_DP)
    
    
    
    
    epochs = 1000000 ## ## 2000000
    if expectile_value == .5:
        agent = def_agent(S_0,mu,sigma,T,n_timesteps,num_stocks,pMeasure=PerformanceMeasure.MSE)
    else:
        agent = def_agent(S_0,mu,sigma,T,n_timesteps,num_stocks,pMeasure=PerformanceMeasure.Expectile,loadModelFromSavedModels=False)
    with tf.compat.v1.Session() as sess:
        perf_in_long, perf_out_long,long_HE,long_HE_Train,long_HE_Test, long_loss_time_AC,long_loss_time_AO,\
        long_loss_time_AC_static,long_loss_time_AO_static, long_loss_time_AC_using_DP = agent.train_test_ddpg(sess, x, x_Test,-1,useFullSavedModel=False)
        
        
    
    print(long_HE[-1])       
    print(perf_in_long)    
    print(perf_out_long) 
    print(long_HE_Train[-1])        
    print(long_HE_Test[-1])    
    print(long_loss_time_AC)    
    print(long_loss_time_AO)
    print(long_loss_time_AC_static)    
    print(long_loss_time_AO_static) 
    print(long_loss_time_AC_using_DP)
    
    
    
    
   
    
    
    
    
    
    
    
    
    print('-----------------------------------------------------')
    print('-----------------------------------------------------')
    print('-----------------------------------------------------')
    
    
    
    
    
    
    # print(short_HE[-1])
    # print(long_HE[-1])    
    
    # print(perf_in_short)
    # print(perf_in_long)
    
    # print(perf_out_short)
    # print(perf_out_long) 

    # print(short_HE_Train[-1])
    # print(long_HE_Train[-1])     
    
    # print(short_HE_Test[-1])
    # print(long_HE_Test[-1])
    
    print(short_loss_time_AC)
    print(long_loss_time_AC)
    
    print(short_loss_time_AO)
    print(long_loss_time_AO)
    
    print(short_loss_time_AO_static)
    print(short_loss_time_AC_static)
    
    print(long_loss_time_AO_static)
    print(long_loss_time_AC_static)    
    
    print(short_loss_time_AC_using_DP)
    print(long_loss_time_AC_using_DP)
    
    
     
        
elif modelType == 'approximate':
    agent = def_agent(S_0,mu,sigma,T,n_timesteps,num_stocks,pMeasure=PerformanceMeasure.Expectile)
    with tf.compat.v1.Session() as sess:
        perf_in_long, perf_out_long, long_HE,a_t,perf_in_long_max,perf_out_long_max,perf_in_long_cvar,perf_out_long_cvar,long_loss_time_DP,long_loss_time_DP_static= agent.train_test_approximate(sess,x, x_Test,-1)
        
    agent = def_agent(S_0,mu,sigma,T,n_timesteps,num_stocks,pMeasure=PerformanceMeasure.Expectile)
    with tf.compat.v1.Session() as sess:
        perf_in_short, perf_out_short, short_HE,a_t,perf_in_short_max,perf_out_short_max,perf_in_short_cvar,perf_out_short_cvar,short_loss_time_DP,short_loss_time_DP_static = agent.train_test_approximate(sess,x, x_Test,1)
    
    # print(short_HE)
    # print(long_HE)
    
    # print(perf_in_short)
    # print(perf_in_long)
    
    # print(perf_out_short) 
    # print(perf_out_long)
    
    # print(perf_in_short_cvar)
    # print(perf_in_long_cvar)
    
    # print(perf_out_short_cvar)
    # print(perf_out_long_cvar) 
    
    # print(perf_in_short_max)
    # print(perf_in_long_max)
    
    # print(perf_out_short_max)
    # print(perf_out_long_max) 
    
    print(short_loss_time_DP)
    print(long_loss_time_DP)
    
    print(short_loss_time_DP_static)
    print(long_loss_time_DP_static)
    
   
    # cvar_short = []
    # cvar_long = []
    # cvar_counter = []
    # for cvar_val in np.arange(.99,.1,-.01):            
    #     cvar_short.append(agent.train_test_approximate(x, x_Test,1,loss_type,cvar_val))
    #     cvar_long.append(agent.train_test_approximate(x, x_Test,-1,loss_type,cvar_val))
    #     cvar_counter.append(cvar_val)
    
    #     fig = plt.figure()
    #     plt.plot(cvar_counter,cvar_short)
    #     # plt.legend()
    #     plt.savefig('D:/Saeed/Equal_Risk_RL/EqualRisk/results/APP/cvar_short.png')
    #     plt.close(fig)
    #     print('------')
        
    #     fig = plt.figure()
    #     plt.plot(cvar_counter,cvar_long)
    #     # plt.legend()
    #     plt.savefig('D:/Saeed/Equal_Risk_RL/EqualRisk/results/APP/cvar_long.png')
    #     plt.close(fig)
    #     print('------')
       
    
    
    # ERP = (short_HE + long_HE)/2
        
