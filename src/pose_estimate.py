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


    def rigid_transform_3D(self,A, B):
        
        centroid_A = tf.expand_dims(tf.reduce_mean(A, axis=2),axis=2)
        centroid_B = tf.expand_dims(tf.reduce_mean(B, axis=2),axis=2)
        
        # centre the points
        AA = A - centroid_A
        BB = B - centroid_B

        # dot is matrix multiplication for array
        H = tf.matmul(AA , tf.transpose(BB,[0,2,1]))

        S, U, V = tf.svd(H)

        R = tf.matmul(V,tf.transpose(U,[0,2,1]))
        # R = Vt.T * U.T

        R_det = tf.matrix_determinant(R)

        def reflection(R,V):
            V_new = tf.concat([V[:,:,0:2],tf.expand_dims(V[:,:,-1]*-1.0,axis=2)],axis=2)
            R = tf.matmul(V_new,tf.transpose(U,[0,2,1]))
            return R

        
        R = tf.cond(tf.less(R_det[0],tf.zeros([],tf.float32)),lambda: reflection(R,V),lambda:R)
        # # special reflection case
        # if linalg.det(R) < 0:
        # print "Reflection detected"
        # Vt[2,:] *= -1
        # R = Vt.T * U.T

        t = -tf.matmul(R,centroid_A) + centroid_B
        #import pdb;pdb.set_trace()
        # print t

        return R, t,R_det
    
    def forward_wrapper(self,output,data_dict,is_training=True, is_reuse=False):
        '''
        A wrapper function for domain transfer.
        Generate dataloader, model, loss and its own summary
        '''
        landmark1 = tf.expand_dims(output[0][0,:,:,:],axis=0)
        landmark2 = tf.expand_dims(output[0][1,:,:,:],axis=0)

        depth1 = tf.expand_dims(data_dict["depth"][0,:,:,:],axis=0)
        depth2 = tf.expand_dims(data_dict["depth"][1,:,:,:],axis=0)
        
        #input_pose = tf.concat([landmark1,depth1,landmark2,depth2],axis=3)

        # pose_final = disp_net_pose(tf.cast(input_pose,tf.float32))
        # quat_est = tfq.Quaternion(pose_final[:,0:4])

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


        #Visibility
        #import pdb;pdb.set_trace()
        if self.trainer.opt.with_vis:
            pred_vis1 = tf.to_float(tf.expand_dims(output[2][0,:],axis=0)>0.5)
            pred_vis2 = tf.to_float(tf.expand_dims(output[2][1,:],axis=0)>0.5)
            lm3d_weights = pred_vis1
            lm3d_weights = lm3d_weights*pred_vis2
        else:
            lm3d_weights = tf.clip_by_value(visibility1,0.0,1.0)
            lm3d_weights = lm3d_weights*tf.clip_by_value(visibility2,0.0,1.0)

        vis_ind = tf.expand_dims(tf.where(tf.equal(lm3d_weights[0],tf.ones([],tf.float32))),axis=0)
        
        batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(landmark1)[0]), 1), [1, tf.shape(vis_ind)[1]])
        vis_ind = tf.concat([tf.expand_dims(batch_index,axis=2), tf.cast(vis_ind,tf.int32)], axis=2)

        gt_vis = tf.transpose(tf.gather_nd(tf.transpose(gt_cam_coord,[0,2,1]),vis_ind),[0,2,1])
        pred_vis=tf.transpose(tf.gather_nd(tf.transpose(pred_cam_coord,[0,2,1]),vis_ind),[0,2,1])


        if self.trainer.opt.with_geo:
            #import pdb;pdb.set_trace()
            input_geo = tf.concat([landmark1,landmark2],axis=3)
            pred_pose = disp_net_pose(input_geo, num_encode=7,is_training=is_training)

            quat_est = tfq.Quaternion(pred_pose[:,0:4])
            R = quat_est.as_rotation_matrix()
            T = tf.expand_dims(pred_pose[:,4:-1]*tf.expand_dims(pred_pose[:,-1],axis=1),axis=2)
        else:
            R,T,R_det = self.rigid_transform_3D(pred_vis,gt_vis)


        pred_lm_3D = tf.matmul(R,pred_vis)+tf.tile(T,[1,1,tf.shape(pred_vis)[2]])

        #Loss
        #lm3d_weights = tf.tile(tf.expand_dims(lm3d_weights,axis=1),[1,3,1])
        transformation_loss = l2loss(gt_vis,pred_lm_3D)

        transformation_loss = tf.cond(tf.less(tf.reduce_sum(tf.cast(lm3d_weights,tf.float32)),tf.ones([],tf.float32)*3.0),lambda:0.0,lambda:transformation_loss)

        coord_pair = [gt_vis,pred_lm_3D]

        #Construct summarie
        
        return transformation_loss,coord_pair