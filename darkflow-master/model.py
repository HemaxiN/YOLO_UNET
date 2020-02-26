# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 08:56:33 2019

@author: hemax
"""

import keras
#create custom metric
#import keras.backend as K


def dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return keras.backend.mean((2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth))


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice


def binary_crossentropy(y, p):
    return keras.backend.mean(keras.backend.binary_crossentropy(y, p))


def loading_model(path):
    model = keras.models.load_model(path, 
                                custom_objects={'dice_coef_loss_bce': dice_coef_loss_bce, 'dice_coef': dice_coef, 
                                                'dice_coef_loss': dice_coef_loss, 'binary_crossentropy': binary_crossentropy} )
    return model