#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:32:59 2018

@author: Colorbaum
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import copy

#%%
"""
Instructions:
    
    The images will be in a file: /training_ready_np.npz
    
    #get training dict
    
    from DSB2018_Unet import load_train_images,hyperparams,nuclei_u_net,iou_metric
    import numpy as np
    #for postprocessing
    from datasciencebowl2018 import fill_int_holes, boundary_knn_batch
    import os
    
    PATH=...
    
    train_images=load_train_images()
    
    params=hyperparams(bdy_cost=100,
                 int_cost=5,
                 bgd_cost=1,
                 fat_bdy_on=True,
                 use_bias=np.array([[1,10,1],[10,1,1],[1,100,1]]),
                 batch_size=1,
                 training_steps=15,
                 learning_rate=0.01,
                 epsilon=0.1,
                 clip_norm=10**5,
                 regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)),
                 rates=[0,0,0,0,0,0,0,0,0]      
                 activation=tf.nn.elu,
                 checkpoint_path=None,
                 save_checkpoints_steps=3,
                 keep_checkpoint_max=3,
                 save_summary_steps=1,
                 logging_step=10
                 
                 )
    
    #If you want to predict data at the end, add "predict_images" to train_images
    train_images["predict_images"]=train_images["eval_images"]
    
    predicted_masks=nuclei_u_net("tboard_unet",train_images,params=params)
    length=predicted_masks.shape[0]
    
    #follow up to evaluate iou metric on predicted masks

    test_masks_true=train_images["eval_masks"]
    
    int_pred_masks=boundary_knn_batch(predicted_masks)
    int_true_masks=boundary_knn_batch)(test_masks)
    
    fill=partial(fill_int_holes,area=25,smoothing=1,remove_objects=4)
    
    int_pred_masks=np.concatenate([fill(int_pred_masks[i,...])[np.newaxis,...]\
    for i in range(length)])
    
    mean=np.mean([iou_metric(int_true_masks[i,...],int_pred_masks[i,...]) for i in \
    range(length)])
    
    print("Mean for test batch:{}".format(mean))
        
    
    ##############################################################
    #exmple of data format 
    train_x=np.zeros((1,128,128,1),dtype=np.float32)
    train_x[0,0:50,0:50,0]=1    
    train_y=np.zeros((1,128,128,3),dtype=np.float32) 
    train_y[0,0:49,0:49,1]=1
    train_y[0,50:,:,0]=train_y[0,:,50:,0]=1
    train_y[0,49,:50,2]=train_y[0,:50,49,2]=1
       
    eval_x=np.ones((1,256,256,1),dtype=np.float32)     
    eval_y=np.ones((1,256,256,3),dtype=np.float32)    
    
    image_data={"train_images":train_x,
                "train_labels":train_y,
                "eval_images":eval_x,
                "eval_labels":eval_y,
                "predict_images": train_x}   
    output=nuclei_u_net("/Users/Colorbaum/Desktop/u_net",params,image_data)
    ###############################################################
   
    """



def iou_metric(y_true_in, y_pred_in,print_table=False):
    """iou for y_true_in and y_pred_in. Both must already have regions integer labeled."""
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(),
                                  bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec) 



"""U-net model. """
def conv(x,filters,kernel_size=(3,3),name=None,params=None):
    return tf.layers.conv2d(x,
                            filters,
                            kernel_size,
                            strides=(1,1),
                            padding="same",
                            activation=params.activation,
                            use_bias=True,
                            kernel_regularizer=params.regularizer,
                            bias_regularizer=params.regularizer,
                            name=name)




def pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],name=None):
    return tf.nn.max_pool(x,
                          ksize,
                          strides,
                          padding="SAME",
                          data_format='NHWC',
                          name=name)

def convT(x,filters,kernel_size=(2,2),strides=(2,2),name=None,params=None):
    return tf.layers.conv2d_transpose(x,
                                        filters,
                                        kernel_size,
                                        strides=strides,
                                        padding='valid',
                                        data_format='channels_last',
                                        activation=params.activation,
                                        use_bias=True,
                                        kernel_initializer=None,
                                        bias_initializer=tf.zeros_initializer(),
                                        kernel_regularizer=params.regularizer,
                                        bias_regularizer=params.regularizer,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None,
                                        trainable=True,
                                        name=None,
                                        reuse=None)


                                    
def cat(x,y,name):
    return tf.concat([x,y],-1,name=name)


def window_adjustment(n):
    def g(x):
        return tf.ceil(x/2), tf.mod(x,2)
    l=[]
    for i in range(4):
        n,r=g(n)
        l.append(tf.cast(r,tf.int32))
    return l


        
def u_net_fn(features,labels,mode,params):
    
    
    c_00=features["x"]
    bat=tf.shape(c_00)[0]
    
    v_adj=window_adjustment(tf.shape(c_00)[1])
    h_adj=window_adjustment(tf.shape(c_00)[2])
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        training=True
    else:
        training=False
    #0th level of convolutions
    c_01=conv(c_00,params.layers[0],name="conv01_down",params=params)
    c_01=tf.layers.dropout(c_01,rate=params.rates[0],noise_shape=[bat,1,1,params.layers[0]],
                           training=training)
    c_02=conv(c_01,params.layers[0],name="conv02_down",params=params)
    c_02=tf.layers.dropout(c_02,rate=params.rates[0],noise_shape=[bat,1,1,params.layers[0]],
                           training=training)
    c_10=pool(c_02,name="pool10")
    #1st level down
    c_11=conv(c_10,params.layers[1],name="conv11_down",params=params)
    c_11=tf.layers.dropout(c_11,rate=params.rates[1],noise_shape=[bat,1,1,params.layers[1]],
                           training=training)
    c_12=conv(c_11,params.layers[1],name="conv12_down",params=params)
    c_12=tf.layers.dropout(c_12,rate=params.rates[1],noise_shape=[bat,1,1,params.layers[1]],
                           training=training)
    
    c_20=pool(c_12,name="pool20")
    #2nd level down
    c_21=conv(c_20,params.layers[2],name="conv21_down",params=params)
    c_21=tf.layers.dropout(c_21,rate=params.rates[2],noise_shape=[bat,1,1,params.layers[2]],
                           training=training)
    c_22=conv(c_21,params.layers[2],name="conv22_down",params=params)
    c_22=tf.layers.dropout(c_22,rate=params.rates[2],noise_shape=[bat,1,1,params.layers[2]],
                           training=training)
    
    c_30=pool(c_22,name="pool30")
    #3rd level down
    c_31=conv(c_30,params.layers[3],name="conv31_down",params=params)
    c_31=tf.layers.dropout(c_31,rate=params.rates[3],noise_shape=[bat,1,1,params.layers[3]],
                           training=training)
    c_32=conv(c_31,params.layers[3],name="conv32_down",params=params)
    c_32=tf.layers.dropout(c_32,rate=params.rates[3],noise_shape=[bat,1,1,params.layers[3]],
                           training=training)
    
    c_40=pool(c_32,name="pool40")
    #4th level down
    c_41=conv(c_40,params.layers[4],name="conv41_down",params=params)
    c_41=tf.layers.dropout(c_41,rate=params.rates[4],noise_shape=[bat,1,1,params.layers[4]],
                           training=training)
    c_42=conv(c_41,params.layers[4],name="conv42_down",params=params)
    c_42=tf.layers.dropout(c_42,rate=params.rates[4],noise_shape=[bat,1,1,params.layers[4]],
                           training=training)
    
    cT_30=convT(c_42,params.layers[3],name="convT30_up",params=params)
    #3rd level up
    v=tf.shape(cT_30)[1]-v_adj[3]
    h=tf.shape(cT_30)[2]-h_adj[3]
    c_31=conv(cat(cT_30[:,:v,:h,:],c_32,name="cat31_up"),params.layers[3], name="conv31_up",params=params)
    c_31=tf.layers.dropout(c_31,rate=params.rates[5],noise_shape=[bat,1,1,params.layers[3]],
                           training=training)
    c_32=conv(c_31,params.layers[3],name="conv32_up",params=params)
    c_32=tf.layers.dropout(c_32,rate=params.rates[5],noise_shape=[bat,1,1,params.layers[3]],
                           training=training)
    cT_20=convT(c_32,params.layers[2],name="convT20_up",params=params)
    #2nd level up
    v=tf.shape(cT_20)[1]-v_adj[2]
    h=tf.shape(cT_20)[2]-h_adj[2]
    c_21=conv(cat(cT_20[:,:v,:h,:],c_22,name="cat21_up"),params.layers[2], name="conv21_up",params=params)
    c_21=tf.layers.dropout(c_21,rate=params.rates[6],noise_shape=[bat,1,1,params.layers[2]],
                           training=training)
    c_22=conv(c_21,params.layers[2],name="conv22_up",params=params)
    c_22=tf.layers.dropout(c_22,rate=params.rates[6],noise_shape=[bat,1,1,params.layers[2]],
                           training=training)
    
    cT_10=convT(c_22,params.layers[1],name="convT10_up",params=params)
    #1st level up
    v=tf.shape(cT_10)[1]-v_adj[1]
    h=tf.shape(cT_10)[2]-h_adj[1]
    c_11=conv(cat(cT_10[:,:v,:h,:],c_12,name="cat11_up"),params.layers[1], name="conv11_up",params=params)
    c_11=tf.layers.dropout(c_11,rate=params.rates[7],noise_shape=[bat,1,1,params.layers[1]],
                           training=training)
    c_12=conv(c_11,params.layers[1],name="conv12_up",params=params)
    c_12=tf.layers.dropout(c_12,rate=params.rates[7],noise_shape=[bat,1,1,params.layers[1]],
                           training=training)
    
    cT_00=convT(c_12,params.layers[0],name="convT10_up",params=params)
    #0th level up
    v=tf.shape(cT_00)[1]-v_adj[0]
    h=tf.shape(cT_00)[2]-h_adj[0]
    c_01=conv(cat(cT_00[:,:v,:h,:],c_02,name="cat01_up"),params.layers[0], name="conv01_up",params=params)
    c_01=tf.layers.dropout(c_01,rate=params.rates[8],noise_shape=[bat,1,1,params.layers[0]],
                           training=training)
    c_02=conv(c_01,params.layers[0],name="conv02_up",params=params)
    c_02=tf.layers.dropout(c_02,rate=params.rates[8],noise_shape=[bat,1,1,params.layers[0]],
                           training=training)
    
    #end. concatonate with original image
    logit_params=copy.copy(params)
    logit_params.activation=None
    logits=conv(cat(c_02,c_00,name="last_image_concat"),3,
                                                        kernel_size=(1,1),
                                                        name="logits_out",
                                                        params=logit_params)
   
    def loss_with_parameters(y_true,y_pred):
        #give error bias to logits
        #error bias should be a 3D matrix whose rows are the weights for ext,int,bdy 
        #in that order. The diagonal should always be 1. 
        try:
            params.use_bias.shape==(3,3)
            bias=tf.convert_to_tensor(params.use_bias)
            ln=tf.log(bias)
            w=tf.einsum("...b,ba->...a",y_true,ln)
            logits=tf.add(w,y_pred)
                
        except:
            logits=y_pred
            
        bdy_cost=params.bdy_cost
        int_cost=params.int_cost
        bgd_cost=params.bgd_cost
        fat_bdy_on=params.fat_bdy_on
        weights_notbdy=tf.convert_to_tensor([bgd_cost,int_cost],dtype=tf.float32)
        
        weighted_notbdy=tf.reduce_sum(
                tf.multiply(weights_notbdy,y_true[:,:,:,0:2]),
                axis=-1,
                keepdims=True)
        
        boundary=y_true[:,:,:,2:3]
        if fat_bdy_on:
            fat_bdy = tf.nn.max_pool(boundary,ksize=[1,3,3,1],
                                 strides=[1,1,1,1],
                                 padding="SAME",
                                 data_format="NHWC")
        else:
            fat_bdy= boundary
        
        
        
        weighted_bdy=tf.scalar_mul(bdy_cost,fat_bdy)
        intersect=tf.multiply(fat_bdy,weighted_notbdy)
       
        weights=tf.add(weighted_bdy,weighted_notbdy)
        weights=tf.subtract(weights,intersect)
        cost=tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=logits)
        cost=tf.expand_dims(cost,-1)
        
        cost=tf.multiply(weights,cost)
        
        return tf.reduce_mean(cost)
    
    predictions = {"probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
                "labels":tf.one_hot(tf.argmax(logits,axis=-1),3,axis=-1,
                                    name="output_labels")}
    
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)


    
    loss=loss_with_parameters(labels,logits)
    if params.regularizer!=None:
        reg_loss=tf.losses.get_regularization_losses()
        loss=loss+tf.reduce_sum(reg_loss)
        
    
    eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels=tf.argmax(labels,axis=-1),
                            predictions=tf.argmax(predictions["labels"],axis=-1),
                            name="accuracy_1")}
    
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions["labels"],axis=-1),
                                          tf.argmax(labels,axis=-1)),tf.float32),
                                    name="accuracy_2")
    pred=predictions["labels"]
    bgd=tf.divide(tf.reduce_sum(tf.multiply(pred[:,:,:,0:1],labels[:,:,:,0:1])),
                           tf.reduce_sum(labels[:,:,:,0:1]),name="bgd_true_pos")
    intr=tf.divide(tf.reduce_sum(tf.multiply(pred[:,:,:,1:2],labels[:,:,:,1:2])),
                       tf.reduce_sum(labels[:,:,:,1:2]),name="int_true_pos")
    bdy=tf.divide(tf.reduce_sum(tf.multiply(pred[:,:,:,2:3],labels[:,:,:,2:3])),
                       tf.reduce_sum(labels[:,:,:,2:3]),name="bdy_true_pos")
    
    tf.summary.scalar("pixel_wise_average_accuracy",acc)
    tf.summary.scalar("weighted_loss",loss)
    tf.summary.scalar("background_true_pos",bgd)
    tf.summary.scalar("interior_true_pos",intr)
    tf.summary.scalar("boundary_true_pos",bdy)
    if mode == tf.estimator.ModeKeys.TRAIN:
        
#    summary_hook = tf.train.SummarySaverHook(2,
#                                                 output_dir="u_net",
#                                                 summary_op=tf.summary.merge_all())
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate,
                                           epsilon=params.epsilon)
        grads,varis=[*zip(*optimizer.compute_gradients(loss=loss))]
        clip_grads=[tf.clip_by_average_norm(grad,params.clip_norm) for grad in grads]
        
        train_op = optimizer.apply_gradients([*zip(clip_grads,varis)],
                                      global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op
                                         )
        
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)
   
def nuclei_u_net(model_dir,image_data,params):
    #all input data is in a dict of np.ndarray
    #image_data is either for training a dict: {train_images:,
    #train_labels=,eval_images:,eval_labels:,predict_images= }
    #Or, for predictions only dict: {predict_images:}
    my_config=tf.estimator.RunConfig(
            save_checkpoints_steps=params.save_checkpoints_steps,
            keep_checkpoint_max=params.keep_checkpoint_max,
            save_summary_steps=params.save_summary_steps
            )
    u_net = tf.estimator.Estimator(model_fn=u_net_fn,
                                             model_dir=model_dir,
                                             config=my_config,
                                             params=params)   
    if "train_images" in image_data.keys():
        tensors_to_log = {"pixel_wise_ave_accuracy": "accuracy_2","bgd_true_pos":"bgd_true_pos",
                          "int_true_pos":"int_true_pos","bdy_true_pos":"bdy_true_pos"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                                  every_n_iter=params.logging_step)
        
       
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
              x={"x": image_data["train_images"]},
              y=image_data["train_labels"],
              batch_size=params.batch_size,
              num_epochs=None,
              shuffle=True)
        u_net.train(input_fn=train_input_fn,
                    steps=params.training_steps,hooks=[logging_hook])
        
    if "eval_images" in image_data.keys():
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
             x={"x": image_data["eval_images"]},
             y=image_data["eval_labels"],
             num_epochs=1,
             shuffle=False)
        
     
       
            
        eval_results = u_net.evaluate(input_fn=eval_input_fn,checkpoint_path=params.checkpoint_path)
        print(eval_results)
       
        #eval_results = u_net.evaluate(input_fn=train_input_fn,steps=1)
        #print(eval_results)    

    if "predict_images" in image_data.keys():
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": image_data["predict_images"]},
                shuffle=False,
                batch_size=params.predict_batch_size,
                num_epochs=1)
        return [x for x in u_net.predict(input_fn=predict_input_fn,
                                         checkpoint_path=params.checkpoint_path)]
    
"""End of U-net model"""  
#%%

class hyperparams(object):
    def __init__(self,
                 layers=[64,128,256,512,1028],
                 bdy_cost=1,
                 int_cost=1,
                 bgd_cost=1,
                 fat_bdy_on=False,
                 use_bias=False,
                 batch_size=1,
                 training_steps=1,
                 learning_rate=0.001,
                 epsilon=0.1,
                 clip_norm=10**5,
                 regularizer=None,
                 activation=tf.nn.relu,
                 rates=None,
                 checkpoint_path=None,
                 predict_batch_size=30,
                 save_checkpoints_steps=50,
                 keep_checkpoint_max=20,
                 save_summary_steps=10,
                 logging_step=10
                 ):
        #Model feature values for u-net. Starting at 'top' of U to 'bottom' of U. 
        try:
            if len(layers)!=5:
                print("There must be values for 5 layers.")
            else:
                self.layers=layers
        except:
            print("Layers must be a list of 5 integer values.")
       
        #Cost function params
        self.bdy_cost=bdy_cost
        self.int_cost=int_cost
        self.bgd_cost=bgd_cost
        self.fat_bdy_on=fat_bdy_on
        #Training params
        self.batch_size=batch_size
        self.predict_batch_size=predict_batch_size
        self.training_steps=training_steps
        #ADAM opt params
        self.learning_rate=learning_rate
        self.epsilon=epsilon
        self.clip_norm=clip_norm
        #regularization and activation functions
        self.regularizer=regularizer
        try:
            if len(rates)==9 and np.all(np.array(rates)>=0) and np.all(np.array(rates)<=1):
                self.rates=rates
            else:
                print("Rates must be a list of 9 values between 0 and 1.")
        except:
            self.rates=[0,0,0,0,0,0,0,0,0]
        self.activation=activation
        #checkpoint to be used in prediction. if None, last checkpoint used.
        self.checkpoint_path=checkpoint_path
        self.save_checkpoints_steps=save_checkpoints_steps
        self.keep_checkpoint_max=keep_checkpoint_max
        self.save_summary_steps=save_summary_steps
        self.logging_step=logging_step #step size for printing accuracy
        #give error bias to logits
        #error bias should be a 3D matrix whose rows are the weights for ext,int,bdy 
        #in that order. The diagonal should always be 1. 
        #The intension is to give greater bias towards certain types of misclassifications
        #For example, misclassifying an exterior point as an interior should be worse
        #than classifying it as a boundary point. It could also be viewed as adding 
        #leniency to misclassifying boundary as exterior, for example (by putting more
        #weight on misclassifying the boundary as interior.)
        # ext int bdy <--incorrect categorization bias
        # (1  b   c) ext true
        # (d  1   f) int true
        # (g  h   1) bdy true
        #
        try:
            if use_bias.shape==(3,3):
                if np.all(use_bias>0):
                    if not np.all(use_bias.diagonal()==[1,1,1]):
                        use_bias = use_bias-np.diag(np.diag(use_bias))+np.eye(3)
                        print("I changed the diagonal of the bias matrix to ones.")
                    self.use_bias = use_bias.astype(np.float32)
                else:
                    self.use_bias=False
                    print("Bias matrix must have all positive elements. Set use_bias to False.")
            else: 
                self.use_bias=False
                print("You didn't input a 3x3 bias matrix. Set use_bias to False.")
            
        except:
            self.use_bias=use_bias


def load_train_images(npz_file_path,invert=False,eval_images=100):
    npz_file=np.load(npz_file_path)   
    keys=sorted(npz_file.keys()) 
    images=np.concatenate([npz_file[key] for key in keys],axis=0)
    if invert:
        images_inv=images[...]
        ones=np.ones(images.shape[0:3])
        images_inv[...,0]=ones-images_inv[...,0]
        images=np.concatenate([images,images_inv],axis=0)
             
    np.random.shuffle(images)
    evals=images[:eval_images,...]
    
    
    images=images[eval_images:,...]
    images=np.concatenate([images,np.rot90(images,1,(1,2))],axis=0)
    np.random.shuffle(images)
    
    image_dict={"train_images":images[...,0:1],
            "train_labels":images[...,1:],
            "eval_images":evals[...,0:1],
            "eval_labels":evals[...,1:]}   

    return image_dict

#%%
