# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:53:32 2019

@author: Hemaxi
"""

#import necessary packages
import pickle
import os
import cv2
import numpy as np
import model
#from skimage.morphology import dilation, erosion
#from scipy.ndimage.morphology import binary_fill_holes
#import matplotlib.pyplot as plt

#unet model name
modelname = 'models_folder/model7.h5'

def compute_predictions(bbox_pickle, image, imgname):

        #load the model 
        model_unet = model.loading_model(modelname)
        
        #print('segmentation')
        #print(np.shape(image))

        #image = image[:,:,0]
        segmentation_mask = segmentation_masks(bbox_pickle, image, model_unet)
        
        prediction = np.zeros(np.shape(segmentation_mask[:,:,0]))
        
        for i in range(len(segmentation_mask[0,0,:])):
        		prediction[segmentation_mask[:,:,i]==1] = i + 1
        

        #pickle_out = open(os.path.splitext(imgname)[0] + ".pickle", "wb")
        #pickle.dump(prediction, pickle_out)
        #pickle_out.close()

        #plt.imshow(prediction)


def segmentation_masks(bbox_pickle, image, model):
    #create an array to save all masks
    sgm_msk = np.zeros((np.shape(image)[0],np.shape(image)[1], len(bbox_pickle)))
    
    resized_img_list = []
    #for each object found by the yolo, obtain its mask
    for i in range(len(bbox_pickle)):
        bbox_aux = bbox_pickle[i]
        xmin = bbox_aux[0]
        ymin = bbox_aux[1]
        xmax = bbox_aux[2]
        ymax = bbox_aux[3]
        
        #print(i)
        #crop the image
        croped_image = image[ymin:ymax,xmin:xmax]
        
        #resize the image
        resized_img = cv2.resize(croped_image, (80, 80))
        resized_img = resized_img / 255.0
        
        resized_img_list.append(resized_img)
    
    resized_img_list = np.array(resized_img_list).reshape(-1, 80, 80, 1) 
    #feed the image to the model        
    pred = model.predict([resized_img_list])
        
    for i in range(len(bbox_pickle)):    
        bbox_aux = bbox_pickle[i]
        xmin = bbox_aux[0]
        ymin = bbox_aux[1]
        xmax = bbox_aux[2]
        ymax = bbox_aux[3]        
       
        
        pred_aux = pred[i,:,:,-1]
                
        #resize the prediction
        resized_pred = cv2.resize(pred_aux, (xmax-xmin, ymax-ymin))
        resized_pred = resized_pred > 0.5
        
        #put the predicted nuclei mask in the segmentation mask
        aux = sgm_msk[:,:,i]
        aux[ymin:ymax, xmin:xmax] = resized_pred
        sgm_msk[:,:,i] = aux
        
    return sgm_msk
    
#compute_predictions(path, modelname, path_pickle)

#print('Elapsed time', round((time.time() - start)/60, 1), 'minutes')