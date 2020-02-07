import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

np.set_printoptions(precision=2, edgeitems=20, linewidth=1000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 3)


ROOT_PATH = "/workspace/ydj/kaggle/kernel/titanic/data/"
TRAIN_CSV_PATH, TEST_CSV_PATH = ROOT_PATH + "original/train.csv", ROOT_PATH + "original/test.csv"
TRAIN_PROC_CSV_PATH, TEST_PROC_CSV_PATH = ROOT_PATH + "processed/train.csv", ROOT_PATH + "processed/test.csv"

