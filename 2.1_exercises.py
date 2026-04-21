import csv
import dateutil.parser
import demjson3
import json
import math
import matplotlib.pyplot as plt
import numpy
import sklearn
from collections import defaultdict
from sklearn import linear_model

dataDir = "C:/Users/aryan/OneDrive/Desktop/Coding/Machine Learning/Recommender_Systems/data/"
path = dataDir + "fantasy_100.json"

f = open(path)
data = []

for l in f:
    d = json.loads(l)
    data.append(d)
    
f.close()

# Fix: Print from the data list, not the loop variable
print(data[0])  # This will work
print(f"Loaded {len(data)} items")