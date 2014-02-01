#!usr/bin/python3

from PIL import Image
import numpy as np
from collections import defaultdict
import random
import copy
import operator
import sys

def k_means(image_np,rgb_values,k):
    cluster_list = {}

    for i in range(k):
        random_rgb = random.choice(list(rgb_values.keys()))
        cluster_list[random_rgb] = np.mean(random_rgb)
        #print('shape of image_np is',np.shape(image_np))
        #print('the cluster_list[',random_rgb,'] and mean value is',cluster_list[random_rgb])

    cluster_list_keys = list(cluster_list.keys()) 

    while(True):
        cluster_points = {}
        cluster_mean= {}
        cluster_distance = {}
        si_cluster_points = {}
        sii_cluster_points = {}
        #print('inside while loop')
        for i in range(len(cluster_list)):
            print('cluster list is',cluster_list)
            print('image np is',np.shape(image_np))
            #cluster_distance[cluster_list_keys[i]] = np.square(np.subtract(np.sum(image_np,-1),cluster_list[cluster_list_keys[i]]))
            cluster_distance[cluster_list_keys[i]] = np.sum(np.square(np.subtract(image_np,cluster_list[cluster_list_keys[i]])),-1) * 0.5
            print('cluster distance is ',np.shape(cluster_distance[cluster_list_keys[i]]))
        for i in range(len(cluster_list)):
            #print("================================================================================")
            #print("The currenet rgb value is ", cluster_list_keys[i],"and mean is",cluster_list[cluster_list_keys[i]])
            cluster_flag = False
            temp_list = []
            for j in range(len(cluster_list)):
                if i != j:
                    #print(i,j)
                    cluster_1_points = cluster_distance[cluster_list_keys[i]] < cluster_distance[cluster_list_keys[j]]
                    temp_list.append(copy.deepcopy(cluster_1_points))
            list_len = len(temp_list)
            if list_len >= 2:
                cluster_flag = True
                cluster_1_points = temp_list[0]
                new_cluster_1_points = np.where(cluster_1_points == True, 1, 0)
                #print('reassuring true points cluster_1_points - 0 ',np.shape(image_np[cluster_1_points]))
                for k in range(1,len(temp_list)):
                    false_value = random.randint(1,1000)
                    #print('the false value is', false_value)
                    #print('reassuring the true cluster_1_points -', k ,' ',np.shape(image_np[temp_list[k]]))
                    new_temp_list = np.where(temp_list[k] == True, 1, false_value)
                    new_cluster_1_points = new_temp_list == new_cluster_1_points
                    #print('After comparison the shape of cluster_1_points is',np.shape(image_np[new_cluster_1_points]))
                    new_cluster_1_points = np.where(new_cluster_1_points == True, 1, 0)
            if cluster_flag == True:
                cluster_1_points = np.where(new_cluster_1_points == 1, True, False)
            #print('Finally the shape of cluster_1_points is',np.shape(image_np[cluster_1_points]))
            si_cluster_1_points = copy.deepcopy(cluster_1_points)
            cluster_1_points = image_np[np.nonzero(cluster_1_points)]
            cluster_points[cluster_list_keys[i]] = copy.deepcopy(cluster_1_points)
            si_cluster_points[cluster_list_keys[i]] = copy.deepcopy(si_cluster_1_points)
            sii_cluster_points[cluster_list_keys[i]] = np.dstack((si_cluster_points[cluster_list_keys[i]],si_cluster_points[cluster_list_keys[i]],si_cluster_points[cluster_list_keys[i]]))
            cluster_mean[cluster_list_keys[i]] = np.mean(cluster_points[cluster_list_keys[i]])
        print('old cluster_mean is',sorted(cluster_list.values()),'new cluster mean is ',sorted(cluster_mean.values()))
        if sorted(cluster_list.values()) == sorted(cluster_mean.values()):
            break
        elif 0.0 in list(cluster_mean.values()):
            break
        else:
            cluster_list = cluster_mean
    image_np_copy = copy.deepcopy(image_np)
    for i in range(len(si_cluster_points)):
        image_np = np.where(sii_cluster_points[cluster_list_keys[i]], cluster_list_keys[i], image_np)
    new_image_np = np.array(image_np, dtype=np.uint8)
    outfile = open('out.txt','w')
    lista = new_image_np.tolist()
    for row in lista:
        outfile.write(str(row))
        outfile.write('\n')
    pil_im = Image.fromarray(new_image_np)
    pil_im.save(sys.argv[2])
    #pil_im.save('pengu-5.jpg')
def main():
    sys.argv = input('Enter the command line arguments: usage : <input_image.jpg k output_image.jpg>').split()
    rgb_values = defaultdict(int)
    image = Image.open(sys.argv[0])
    #image = Image.open('Images\Penguins.jpg')
    image_np = np.array(image, dtype=np.uint8)
    pil_im = Image.fromarray(image_np)
    pil_im.save('pil.jpg')
    for pixel in image.getdata():
        rgb_values[pixel] += 1
    print('length of rgb_value is',len(rgb_values))
    k_means(image_np,rgb_values,int(sys.argv[1]))
    
    
if __name__ == "__main__" : main()