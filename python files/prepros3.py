import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv('drawndata2.csv')
