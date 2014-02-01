#!usr/bin/python3

import numpy as np
import random
import math

def gaussian_distribution(train,mean_np,variance_np):
    temp1 = 1 / (math.sqrt(2*3.14) * np.sqrt(variance_np))
    temp2 = np.square(train - mean_np) / variance_np
    #temp2 = np.square(np.transpose((np.transpose(train) - mean_np)))/variance_np 
    gaussian_value = temp1 * (np.exp(-0.5 * temp2))
    return gaussian_value

def main():
    train = np.loadtxt('em_data.txt')[np.newaxis]
    train = np.transpose(train)
    #print(train)
    #print('train shape is',np.shape(train))
    train_mean = np.mean(train)
    #print(train_mean)
    cluster_1_mean = train[random.uniform(0.0,np.shape(train)[0])]
    cluster_2_mean = train[random.uniform(0.0,np.shape(train)[0])]
    cluster_3_mean = train[random.uniform(0.0,np.shape(train)[0])]
    cluster_mean_list = []
    cluster_mean_list.append(cluster_1_mean)
    cluster_mean_list.append(cluster_2_mean)
    cluster_mean_list.append(cluster_3_mean)
    cluster_mean_np = np.transpose(np.array(cluster_mean_list))
    print('the initial cluster mean is',cluster_mean_np)
    #print('the cluster mean np is',np.shape(cluster_mean_np))
    #print(cluster_1_mean,cluster_2_mean,cluster_3_mean)
    train_variance = np.sum(np.square(train - train_mean)) / len(train)
    #print(train_variance)
    cluster_1_variance = train_variance * 1.1
    cluster_2_variance = train_variance * 1.2
    cluster_3_variance = train_variance * 1.3
    cluster_variance_list = []
    cluster_variance_list.append(cluster_1_variance)
    cluster_variance_list.append(cluster_2_variance)
    cluster_variance_list.append(cluster_3_variance)
    cluster_variance_np = np.array(cluster_variance_list)[np.newaxis]
    print('the initial cluster variance np is',cluster_variance_np)
    #print('the cluster_variance_np is',np.shape(cluster_variance_np))
    c1_alpha = 1/3
    c2_alpha = 1/3
    c3_alpha = 1/3
    cluster_alpha_list = []
    cluster_alpha_list.append(c1_alpha)
    cluster_alpha_list.append(c2_alpha)
    cluster_alpha_list.append(c3_alpha)
    cluster_alpha_np = np.array(cluster_alpha_list)[np.newaxis]
    print('the inital cluster alpha np is',cluster_alpha_np)
    #print('the cluster_alpha_np is',np.shape(cluster_alpha_np))
    gaussian_value = gaussian_distribution(train,cluster_mean_np,cluster_variance_np)
    #print('the gaussian value is',np.shape(gaussian_value))
    #print(np.shape(gaussian_value))

    #cl_al_np = np.array(cluster_alpha_list)[np.newaxis] 
    #print(cl_al_np,np.shape(cl_al_np))
    #print(np.transpose(cl_al_np))
    max_log_likelihood = np.sum(np.log(np.dot(gaussian_value , np.transpose(cluster_alpha_np))))
    print('inital max_log_likelihood is',max_log_likelihood)
    #max_log_likelihood =  np.sum(np.log(np.sum(gaussian_value * np.transpose(cluster_alpha_list))))
    
    while(True):
        '''
        E Step Calculation of wi,k
        '''
        #print('the shape of cluster_mean_np is',np.shape(cluster_mean_np))
        gaussian_value = gaussian_distribution(train,cluster_mean_np,cluster_variance_np)
        numerator = gaussian_value * cluster_alpha_np
        #print('temp_weight_vector',np.shape(numerator))
        denominator = np.sum(numerator,1)[np.newaxis]
        weight_vector = numerator / np.transpose(denominator)
        #print(weight_vector)
        #print(np.sum(weight_vector,1))
        #print('the shape of weight_vector is',np.shape(weight_vector),len(weight_vector))
        
        '''
        M-Step
        '''
        NK = np.sum(weight_vector,0)[np.newaxis]
        cluster_alpha_np = NK / len(weight_vector)
        #print('the shape of nk is',np.shape(NK))
        #print('the NK is',NK,'the sum is Nk',np.sum(NK))
        
        cluster_mean_np = np.transpose(np.dot(np.transpose(weight_vector),train)) / (NK)
        #print('the final of mean_new is',np.shape(cluster_mean_np))
        
        #cluster_variance_np = np.sum(np.dot(np.transpose(weight_vector),np.square(np.subtract(train,cluster_mean_np))),0) / NK
        #cluster_variance_np = np.sum(np.dot(np.transpose(weight_vector),np.square(np.subtract(train,cluster_mean_np))) / NK,0)[np.newaxis]
        cluster_variance_np = (np.sum(weight_vector * np.square(np.subtract(train,cluster_mean_np)),0)[np.newaxis]) / NK
        #print('the shape of variance_new is',np.shape(cluster_variance_np))
        
        
        gaussian_value = gaussian_distribution(train,cluster_mean_np,cluster_variance_np)
        new_max_log_likelihood = np.sum(np.log(np.dot(gaussian_value , np.transpose(cluster_alpha_np))))
        #print('new_max_log_likelihood is', new_max_log_likelihood)
        
         
        
        if max_log_likelihood == new_max_log_likelihood: 
            print('final_max_log_likelihood is', new_max_log_likelihood)
            print('the final variance is',cluster_variance_np)
            print('the final  mean is ',cluster_mean_np)
            break
        else:
            max_log_likelihood = new_max_log_likelihood
        
    
    
    
    
    
if __name__ == "__main__" : main()