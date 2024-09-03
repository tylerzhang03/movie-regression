import random
random.seed(19340532)

import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

ratings = []

with open('data.txt', 'r') as file:
    for line in file:
        line = line.strip()
        
        if ':' in line:
            movieId = line.split(':')[0].strip()
            
        elif ',' in line:
            split = line.split(',')
            split[2] = math.ceil(int(split[2].split('-')[1])/3)
            rating = [movieId] + split
            ratings.append(rating)
            
ratings = pd.DataFrame(ratings, columns = ['movieId', 'userId', 'rating', 'date'])

test_set = ratings.groupby('movieId').sample(n=1)
training_set = ratings.drop(test_set.index)

model = RandomForestRegressor(n_estimators=10, n_jobs=-1)
model.fit(training_set.iloc[:,np.r_[0,3]], training_set.iloc[:,2])

y_true = test_set.iloc[:,2]
y_pred = model.predict(test_set.iloc[:,np.r_[0,3]])

print(mean_squared_error(y_true, y_pred, squared=False))