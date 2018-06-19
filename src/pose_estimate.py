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

class pose_estimate:
    '''
    A wrapper function which create data, model and loss according to input type
    '''
    def __init__(self,trainer):
        self.trainer = trainer

    
    def forward_wrapper(self,output,data_dict):
        '''
        A wrapper function for domain transfer.
        Generate dataloader, model, loss and its own summary
        '''
        landmark1 = tf.expand_dims(output[1][0,:,:,:],axis=0)
        landmark2 = tf.expand_dims(output[1][1,:,:,:],axis=0)

        depth1 = tf.expand_dims(data_dict["depth"][0,:,:,:]*100.0,axis=0)
        depth2 = tf.expand_dims(data_dict["depth"][1,:,:,:]*100.0,axis=0)

        input_pose = tf.concat([landmark1,depth1,landmark2,depth2],axis=3)

        pose_final = disp_net_pose(tf.cast(input_pose,tf.float32))
        quat_est = tfq.Quaternion(pose_final[:,0:4])

        visibility1 = tf.expand_dims(data_dict['visibility'][0,:],axis=0)
        visibility2 = tf.expand_dims(data_dict['visibility'][1,:],axis=0)

        # quaternion1 = tf.expand_dims(data_dict['quaternion'][0,:],axis=0)
        # quaternion2 = tf.expand_dims(data_dict['quaternion'][1,:],axis=0)

        # quat_gt1 = tfq.Quaternion(quaternion1)
        # quat_gt2 = tfq.Quaternion(quaternion2)

        # translation1 = tf.expand_dims(data_dict['translation'][0,:],axis=0)
        # translation2 = tf.expand_dims(data_dict['translation'][0,:],axis=0)


       #Get gt and estimate landmark locations
        pred_lm_coord = tf.reverse(argmax_2d(landmark2),[1])
        gt_lm_coord = tf.reverse(argmax_2d(landmark1),[1])

        batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(landmark1)[0]), 1), [1, tf.shape(landmark1)[3]])
        index_gt = tf.concat([tf.expand_dims(batch_index,axis=2), tf.transpose(tf.reverse(gt_lm_coord,[1]),[0,2,1])], axis=2)
        index_pred = tf.concat([tf.expand_dims(batch_index,axis=2), tf.transpose(tf.reverse(pred_lm_coord,[1]),[0,2,1])], axis=2)


        gt_depth_val = tf.gather_nd(depth1,index_gt)
        pred_depth_val = tf.gather_nd(depth2,index_pred)
        
        ones = tf.ones([tf.shape(landmark2)[0], 1, tf.shape(landmark2)[3]])
        pred_lm_coord = tf.concat([tf.cast(pred_lm_coord,tf.float32),ones],axis=1)
        gt_lm_coord = tf.concat([tf.cast(gt_lm_coord,tf.float32),ones],axis=1)

        gt_cam_coord = pixel2cam(gt_depth_val,gt_lm_coord,tf.expand_dims(data_dict["matK"][0,:,:],axis=0))
        pred_cam_coord = pixel2cam(pred_depth_val,pred_lm_coord,tf.expand_dims(data_dict["matK"][1,:,:],axis=0))

        #import pdb;pdb.set_trace()
        # gt_lm_3D = tf.matmul(quat_gt.as_rotation_matrix(), gt_cam_coord)+tf.tile(tf.expand_dims(translation[:,0:3]*tf.expand_dims(translation[:,-1],axis=1),axis=2),[1,1,tf.shape(gt_cam_coord)[2]])
        
        
        pred_lm_3D = tf.matmul(quat_est.as_rotation_matrix(), pred_lm_coord)+tf.tile(tf.expand_dims(pose_final[:,4:-1]*tf.expand_dims(pose_final[:,-1],axis=1),axis=2),[1,1,tf.shape(gt_cam_coord)[2]])

        #Loss
        lm3d_weights = tf.clip_by_value(visibility1,0.0,1.5)
        lm3d_weights = tf.clip_by_value(visibility2,0.0,1.5)*lm3d_weights
        lm3d_weights = tf.tile(tf.expand_dims(lm3d_weights,axis=1),[1,3,1])
        transformation_loss = l2loss(gt_cam_coord,pred_lm_3D,lm3d_weights)        

        #Construct summaries
        tf.summary.scalar('losses/transformation', transformation_loss)
        
        return transformation_loss