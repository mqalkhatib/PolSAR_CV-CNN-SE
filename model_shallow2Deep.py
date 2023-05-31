''' 
Shallow to deep network
'''

import tensorflow as tf
import cvnn.layers as complex_layers
from tensorflow import keras
from SAR_utils import *
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input

###############################################################################
def cmplx_shallow2Deep(X_cmplx, num_classes):
    
    
    cmplx_inputs = complex_layers.complex_input(shape=(X_cmplx.shape[1:]))
    
    # Shallow Path
    c0 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(cmplx_inputs)
    c0_flat = complex_layers.ComplexFlatten()(c0)
    
    # Mid Path
    c1 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(cmplx_inputs)
    c1 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(c1)
    c1_flat = complex_layers.ComplexFlatten()(c1)

    # Deep Path
    c2 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(cmplx_inputs)
    c2 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(c2)
    c2 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(c2)
    c2_flat = complex_layers.ComplexFlatten()(c2)
    
    features_cat = tf.concat([c0_flat, c1_flat, c2_flat], axis = 1)

    
    
    
    c3 = complex_layers.ComplexDense(128, activation='cart_relu')(features_cat)
    c3 = complex_layers.ComplexDropout(0.25)(c3)
    c4 = complex_layers.ComplexDense(64, activation='cart_relu')(c3)
    
    predict = complex_layers.ComplexDense(num_classes,activation="softmax_real_with_abs")(c4)


    model = tf.keras.Model(inputs=[cmplx_inputs], outputs=predict)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model
###############################################################################



###############################################################################
def real_shallow2Deep(X_cmplx, num_classes):
    
    
    cmplx_inputs = Input(shape=(X_cmplx.shape[1:]))
    
    # Shallow Path
    c0 = Conv2D(23, activation='relu', kernel_size=(3,3), padding="same")(cmplx_inputs)
    c0_flat = Flatten()(c0)
    
    # Mid Path
    c1 = Conv2D(23, activation='relu', kernel_size=(3,3), padding="same")(cmplx_inputs)
    c1 = Conv2D(23, activation='relu', kernel_size=(3,3), padding="same")(c1)
    c1_flat = Flatten()(c1)

    # Deep Path
    c2 = Conv2D(23, activation='relu', kernel_size=(3,3), padding="same")(cmplx_inputs)
    c2 = Conv2D(23, activation='relu', kernel_size=(3,3), padding="same")(c2)
    c2 = Conv2D(23, activation='relu', kernel_size=(3,3), padding="same")(c2)
    c2_flat = Flatten()(c2)
    
    features_cat = tf.concat([c0_flat, c1_flat, c2_flat], axis = 1)

    
    
    
    c3 = Dense(181, activation='relu')(features_cat)
    c3 = Dropout(0.25)(c3)
    c4 = Dense(91, activation='relu')(c3)
    
    predict = Dense(num_classes,activation="softmax_real_with_abs")(c4)


    model = tf.keras.Model(inputs=[cmplx_inputs], outputs=predict)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model
###############################################################################
from SAR_utils import cmplx_SE_Block


def cmplx_shallow2Deep_SE(X_cmplx, num_classes):
    
    
    cmplx_inputs = complex_layers.complex_input(shape=(X_cmplx.shape[1:]))
    
    # Shallow Path
    c0 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(cmplx_inputs)
    
    
    # Mid Path
    c1 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(cmplx_inputs)
    c1 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(c1)
    

    # Deep Path
    c2 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(cmplx_inputs)
    c2 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(c2)
    c2 = complex_layers.ComplexConv2D(16, activation='cart_relu', kernel_size=(3,3), padding="same")(c2)
   
    
    features_concat = tf.concat([c0, c1, c2], axis = 3)
    se = cmplx_SE_Block(features_concat, se_ratio = 8)
    se = cmplx_SE_Block(se, se_ratio = 8)
    se = cmplx_SE_Block(se, se_ratio = 8)







    features_concat_flat = complex_layers.ComplexFlatten()(se)

    
    
    c3 = complex_layers.ComplexDense(128, activation='cart_relu')(features_concat_flat)
    c3 = complex_layers.ComplexDropout(0.25)(c3)
    c4 = complex_layers.ComplexDense(64, activation='cart_relu')(c3)
    
    predict = complex_layers.ComplexDense(num_classes,activation="softmax_real_with_abs")(c4)


    model = tf.keras.Model(inputs=[cmplx_inputs], outputs=predict)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model









