#!usr/bin/python3
import numpy as np
import copy
import random

def Perceptron(list_of_words):
    '''
    Create a probability vector and weight vector
    '''
    

    eps = 0.3 # step size

    o_py = np.empty((len(data_matrix_py)))
    global weight_py
    weight_py = np.empty(list_of_words-1,float)
    response = 0
    for i in range(len(weight_py)):
        weight_py[i] = random.random()
    dw_py = np.empty(list_of_words-1,object)
    dw_py.fill(0)
    k = 0
    value = 0
    while True:
        correct_counter = 0
        for i in range(len(data_matrix_py)):

            o_py = np.dot(data_matrix_py[i,:-1],weight_py)
            
            if o_py >= 0:
                response = 1
                value = 1
                
                if data_matrix_py[i,-1] == 1:
                    correct_counter += 1
            elif o_py < 0:
                response = -1
                value = 0
                if data_matrix_py[i,-1] == 0:
                    correct_counter += 1
            if  data_matrix_py[i,-1] != value:
            
                s_py = data_matrix_py[i,-1] - response
                weight_py = weight_py + np.dot(eps,(np.dot(s_py,data_matrix_py[i,:-1])))
        
        k += 1
    
        if correct_counter == (len(data_matrix_py)):

            break

        
def read_file(filename):
    infile = open(filename,"r")
    i = 0
    matrix_list = []
    main_matrix_list = []
   
    for line in infile:
        for word in line.split():
            
            letter = word.split(":")
            
            if len(letter) is 1:
                
                letter = list(map(int,letter))
                c_v = letter
                i_c_v = letter[0]
                  
            else:
                
                matrix_list.append(int(letter[1]))
        
        matrix_list.append(i_c_v)
        main_matrix_list.append(copy.deepcopy(matrix_list))
       
        matrix_list[:] = []

    global data_matrix_py
    data_matrix_py = np.empty((len(main_matrix_list),len(main_matrix_list[0])),float)
    data_matrix_py = np.array(main_matrix_list)



def main():
    read_file("training.train")
    Perceptron(len(data_matrix_py[0]))

    read_file("validation.new")



        

    
    incorrect_spam = 0
    correct_spam = 0
    incorrect_ham = 0
    correct_ham = 0
    final_output = np.empty((len(data_matrix_py),len(data_matrix_py[0])-1),float)

    final_output = np.dot(data_matrix_py[:,:-1],weight_py)
    
    for i in range(len(final_output)):
        
        
        if final_output[i] < 0:
            
            if data_matrix_py[i,-1] == 1:
                incorrect_spam += 1
            else:
                correct_spam += 1
        elif final_output[i] > 0:
           
            if data_matrix_py[i,-1] == 0:
                
                incorrect_ham += 1
            else:
                correct_ham += 1

    ham_accuracy = (correct_ham/(correct_ham+incorrect_spam))*100

    spam_accuracy = (correct_spam/(correct_spam+incorrect_ham))*100
    print('The overall accuracy of linear perceptron is',format((ham_accuracy + spam_accuracy)/2))
   
if __name__ == "__main__" : main()