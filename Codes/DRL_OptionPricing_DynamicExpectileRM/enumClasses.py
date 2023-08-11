from enum import Enum


class PerformanceMeasure(Enum):
    Expectile=0
    MSE=2
    Quantile=1
    CVaR=3

class executionMode(Enum):
    train = 0
    test = 1
    validation = 2
    training_done = 3