import tensorflow as tf
import numpy as np
from data_loader_direct import DataLoader
from my_losses import *
from model import *
import time
import math
import os
import cv2
from estimator_rui import estimator_rui
import utils_lr as utlr


class DH_estimate:
    '''
    A wrapper function which create data, model and loss according to input type
    '''
    def __init__(self,trainer):
        self.trainer = trainer


    def est_pose(self,landmark1,landmark2):

        input_geo = tf.concat([landmark1,landmark2],axis=3)
        pred_pose = disp_net_pose(input_geo, num_encode=7,is_training=self.is_training)

        return pred_pose


    def process_pose_est(self,landmark1,landmark2,pixel_coords1,pixel_coords2,data_dict,pose_weight):


        pts1 = tf.concat([tf.expand_dims(pixel_coords1[:,:,0],axis=2),
                          tf.expand_dims(pixel_coords1[:,:,10],axis=2),
                          tf.expand_dims(pixel_coords1[:,:,19],axis=2),
                          tf.expand_dims(pixel_coords1[:,:,26],axis=2)
                         ],axis=2)
        pts2 = tf.concat([tf.expand_dims(pixel_coords2[:,:,0],axis=2),
                          tf.expand_dims(pixel_coords2[:,:,10],axis=2),
                          tf.expand_dims(pixel_coords2[:,:,19],axis=2),
                          tf.expand_dims(pixel_coords2[:,:,26],axis=2)
                         ],axis=2)

        
        B,H,W,D = landmark1.get_shape().as_list()
        fixed_points = tf.concat(
                                 [tf.concat([tf.ones([B,1,1])*W*0.25, tf.ones([B,1,1])*H*0.25],axis=1),
                                  tf.concat([tf.ones([B,1,1])*W*0.75, tf.ones([B,1,1])*H*0.25],axis=1),
                                  tf.concat([tf.ones([B,1,1])*W*0.75, tf.ones([B,1,1])*H*0.75],axis=1),
                                  tf.concat([tf.ones([B,1,1])*W*0.25, tf.ones([B,1,1])*H*0.75],axis=1)],
                                  axis=2
                                ) 
        H_gt,H_flat = utlr.solve_DLT(pts1,pts2)

        #import pdb;pdb.set_trace()
        ones = tf.ones([tf.shape(fixed_points)[0], 1, tf.shape(fixed_points)[2]])
        fixed_points_pad = tf.concat([tf.cast(fixed_points,tf.float32),ones],axis=1)        
        warped_points = tf.matmul(H_gt,fixed_points_pad)
        warped_points = warped_points/tf.tile(tf.expand_dims(warped_points[:,-1,:],axis=1),[1,3,1])

        gt_shift = tf.contrib.layers.flatten(fixed_points-warped_points[:,0:2,:])

        input_geo = tf.concat([landmark1,landmark2],axis=3)
        pred_shift = disp_net_pose(input_geo, num_encode=7,is_training=self.is_training)

        pred_points = fixed_points-tf.reshape(gt_shift,[-1,2,4])
        H_pred,H_flat_pred = utlr.solve_DLT(fixed_points,pred_points)

        if self.trainer.opt.proj_img:

            t_image=tf.contrib.image.transform(tf.expand_dims(data_dict['image'][1,:,:,:],axis=0),H_flat[:,:,0])
            t_image_pred=tf.contrib.image.transform(tf.expand_dims(data_dict['image'][1,:,:,:],axis=0),H_flat_pred[:,:,0])
            tf.summary.image('proj' , \
                            t_image)
            tf.summary.image('proj_pred' , \
                            t_image_pred)     
            tf.summary.image('tgt' , \
                            tf.expand_dims(data_dict['image'][0,:,:,:],axis=0))  


        transformation_loss = l2loss_mean(gt_shift,pred_shift)*pose_weight
        

        return transformation_loss
    


    def forward_wrapper(self,output,data_dict,pose_weight,scope_name="pose",is_training=True):
        '''
        A wrapper function for domain transfer.
        Generate dataloader, model, loss and its own summary
        '''
        self.is_training = is_training
        #Landmark heat maps
        landmark1 = tf.expand_dims(output[0,:,:,:],axis=0)
        landmark2 = tf.expand_dims(output[1,:,:,:],axis=0)
    
        #GT Landmark heat maps
        gtlandmark1 = tf.expand_dims(data_dict["points2D"][0,:,:,:],axis=0)
        gtlandmark2 = tf.expand_dims(data_dict["points2D"][1,:,:,:],axis=0)

        pixel_coords1 = tf.expand_dims(data_dict['pixel_coords'][0,:,:],axis=0)
        pixel_coords2 = tf.expand_dims(data_dict['pixel_coords'][1,:,:],axis=0)

        
        transformation_loss = self.process_pose_est(landmark1,
                                                    landmark2,
                                                    pixel_coords1,
                                                    pixel_coords2,
                                                    data_dict,
                                                    pose_weight
                                                    )



        coord_pair = [data_dict['image'][1,:,:,:]]

        return transformation_loss,coord_pair
