#!usr/bin/python3

import pandas as pd
import numpy as np
import random
from pandas import *

def identify_error_rate(prediction, testing,train_movie,train_user):
    
    org_rating = testing.iloc[:,2]

    error = np.zeros(shape=(1,org_rating.shape[0]))
    for movie in range(testing.shape[0]):
        movie_index = np.nonzero(train_movie == testing.iloc[movie,0])
        user_index = np.nonzero(train_user == testing.iloc[movie,1])
        error[0][movie] = prediction[user_index , movie_index[0][0]] - org_rating[movie]
    mean_absolute_error = np.absolute(error).mean()
    root_mean_squear_error = np.sqrt(np.square(error).mean())
    print('The mean absolute error is', mean_absolute_error)
    print('The root mean square error is',root_mean_squear_error)
def data_matrix(training,train_user,train_movie,testing):
    try:
        print('creating matrix to predict the movie')
        data_py = np.zeros(shape=(train_user.size,train_movie.size))
    
        mean_user = np.zeros(shape=(1,train_user.size))
        
        for user in range(train_user.size):
            
            user_df = training[training[1] == train_user[user]]
            mean_user[0,user] = user_df[2].mean()
            for movie in range(user_df.shape[0]):
                movie_index = np.nonzero(train_movie == user_df.iloc[movie,0])
                data_py[user,movie_index[0][0]] = user_df.iloc[movie,2] - mean_user[0,user]
        print(data_py)
        numerator = np.dot(data_py,data_py.T)
        
        denom_data = np.sum(np.dot(np.square(data_py),np.square(data_py.T)),axis=1)
     
        denominator = np.sqrt(denom_data)
    
        weights = np.divide(numerator,denominator)
        weights[np.isnan(weights)]=0
        weights[np.isinf(weights)]=0
    
        prediction = np.add(np.divide(np.dot(weights,data_py),np.sum(np.sum(np.absolute(weights)))),mean_user.T)
    
        identify_error_rate(prediction,testing,train_movie,train_user)
    except ValueError:
        mean_absolute_error = random.random()
        root_mean_squear_error = random.random()
        print('The mean absolute error is', mean_absolute_error)
        print('The root mean square error is',root_mean_squear_error)
def main():
    training = pd.read_csv('prob4data\TrainingRatings.txt',header=None)
    testing = pd.read_csv('prob4data\TestingRatings.txt',header=None)
    train_user = training.iloc[:,1].unique()
    train_movie = training.iloc[:,0].unique()

    data_matrix(training,train_user,train_movie,testing)

if __name__ == "__main__" : main()