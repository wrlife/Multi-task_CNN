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
    
    def forward_wrapper(self,output,data_dict,pose_weight,is_training=True):
        '''
        A wrapper function for domain transfer.
        Generate dataloader, model, loss and its own summary
        '''

        #Landmark heat maps
        landmark1 = tf.expand_dims(output[0,:,:,:],axis=0)
        landmark2 = tf.expand_dims(output[1,:,:,:],axis=0)

        #Depth maps
        depth1 = tf.expand_dims(data_dict["depth"][0,:,:,:],axis=0)
        depth2 = tf.expand_dims(data_dict["depth"][1,:,:,:],axis=0)

        #Get GT visibility
        visibility1 = tf.expand_dims(data_dict['visibility'][0,:],axis=0)
        visibility2 = tf.expand_dims(data_dict['visibility'][1,:],axis=0)

        #Get GT rotation and translation
        quaternion = data_dict['quaternion']
        translation = data_dict['translation']

        rotation_loss = 0.0
        translation_loss = 0.0

        min_thresh = tf.constant(5000.0)
        
        #Soft arg-max operation
        if self.trainer.opt.with_geo:
            norm_to_regular = tf.concat([tf.ones([1,28,1])*480, tf.ones([1,28,1])*640],axis=2)
            lm1_coord = tf.reverse(tf.transpose(tf.reshape((tf.contrib.layers.spatial_softmax(landmark1,temperature=1.0)+1)/2.0,[1,28,2])*norm_to_regular,[0,2,1]),[1])
            lm2_coord = tf.reverse(tf.transpose(tf.reshape((tf.contrib.layers.spatial_softmax(landmark2,temperature=1.0)+1)/2.0,[1,28,2])*norm_to_regular,[0,2,1]),[1])

            gt_lm_coord = lm1_coord
            pred_lm_coord = lm2_coord
        else:
            pred_lm_coord = tf.reverse(argmax_2d(landmark2),[1])
            gt_lm_coord = tf.reverse(argmax_2d(landmark1),[1])

        #Extract depth value at landmark locations
        batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(landmark1)[0]), 1), [1, tf.shape(landmark1)[3]])
        index_gt = tf.concat([tf.expand_dims(batch_index,axis=2), tf.transpose(tf.reverse(tf.to_int32(gt_lm_coord),[1]),[0,2,1])], axis=2)
        index_pred = tf.concat([tf.expand_dims(batch_index,axis=2), tf.transpose(tf.reverse(tf.to_int32(pred_lm_coord),[1]),[0,2,1])], axis=2)
        gt_depth_val = tf.gather_nd(depth1,index_gt)
        pred_depth_val = tf.gather_nd(depth2,index_pred)

        
        #Filter out invisible and depth zero points

        #Invisible points
        lm1_val = tf.gather_nd(landmark1,index_gt)
        lm2_val = tf.gather_nd(landmark2,index_pred)

        lm1_val_sup = tf.expand_dims(lm1_val[:,0,0],axis=1)
        lm2_val_sup = tf.expand_dims(lm2_val[:,0,0],axis=1)
        for ii in range(1,28):
            lm1_val_sup = tf.concat([lm1_val_sup,tf.expand_dims(lm1_val[:,ii,ii],axis=1)],axis=1)
            lm2_val_sup = tf.concat([lm2_val_sup,tf.expand_dims(lm2_val[:,ii,ii],axis=1)],axis=1)                

        pred_vis1 = tf.to_float(lm1_val_sup>(tf.maximum(tf.reduce_max(landmark1)/10.0,min_thresh)))
        pred_vis2 = tf.to_float(lm2_val_sup>(tf.maximum(tf.reduce_max(landmark2)/10.0,min_thresh)))
        lm3d_weights = pred_vis1
        lm3d_weights = lm3d_weights*pred_vis2

        #mutual invis and depth zero
        mutualdepth = gt_depth_val*pred_depth_val
        zero_depth = tf.expand_dims(
                        tf.where(
                            tf.logical_and(
                                tf.greater(mutualdepth[0,:,0],tf.ones([],tf.float32)*10.0),
                                tf.equal(lm3d_weights[0],tf.ones([],tf.float32))
                            )
                        ),
                    axis=0)
        zero_index = tf.tile(tf.expand_dims(tf.range(tf.shape(landmark1)[0]), 1), [1, tf.shape(zero_depth)[1]])
        zero_depth = tf.concat([tf.expand_dims(zero_index, axis=2), tf.cast(zero_depth,tf.int32)], axis=2)
        gt_depth_val = tf.gather_nd(gt_depth_val,zero_depth)
        gt_lm_coord = tf.transpose(tf.gather_nd(tf.transpose(gt_lm_coord,[0,2,1]),zero_depth),[0,2,1])
        pred_depth_val = tf.gather_nd(pred_depth_val,zero_depth)
        pred_lm_coord = tf.transpose(tf.gather_nd(tf.transpose(pred_lm_coord,[0,2,1]),zero_depth),[0,2,1])


        #Project 2D to 3D
        ones = tf.ones([tf.shape(landmark2)[0], 1, tf.shape(zero_depth)[1]])
        pred_lm_coord = tf.concat([tf.cast(pred_lm_coord,tf.float32),ones],axis=1)
        gt_lm_coord = tf.concat([tf.cast(gt_lm_coord,tf.float32),ones],axis=1)
        gt_cam_coord = pixel2cam(gt_depth_val,gt_lm_coord,tf.expand_dims(data_dict["matK"][0,:,:],axis=0))
        pred_cam_coord = pixel2cam(pred_depth_val,pred_lm_coord,tf.expand_dims(data_dict["matK"][1,:,:],axis=0))

        pred_vis = pred_cam_coord
        gt_vis = gt_cam_coord
        #Visibility
        #import pdb;pdb.set_trace()
        # if self.trainer.opt.with_vis:
        #     pred_vis1 = tf.to_float(tf.expand_dims(output[2][0,:],axis=0)>0.5)
        #     pred_vis2 = tf.to_float(tf.expand_dims(output[2][1,:],axis=0)>0.5)
        #     lm3d_weights = pred_vis1
        #     lm3d_weights = lm3d_weights*pred_vis2
        # else:

        # lm3d_weights = tf.clip_by_value(visibility1,0.0,1.0)
        # lm3d_weights = lm3d_weights*tf.clip_by_value(visibility2,0.0,1.0)
        # vis_ind = tf.expand_dims(tf.where(tf.equal(lm3d_weights[0],tf.ones([],tf.float32))),axis=0)
        # batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(landmark1)[0]), 1), [1, tf.shape(vis_ind)[1]])
        # vis_ind = tf.concat([tf.expand_dims(batch_index,axis=2), tf.cast(vis_ind,tf.int32)], axis=2)
        # gt_vis = tf.transpose(tf.gather_nd(tf.transpose(gt_cam_coord,[0,2,1]),vis_ind),[0,2,1])
        # pred_vis = tf.transpose(tf.gather_nd(tf.transpose(pred_cam_coord,[0,2,1]),vis_ind),[0,2,1])

        if self.trainer.opt.with_geo:
            
            input_geo = tf.concat([landmark1,landmark2],axis=3)
            pred_pose = disp_net_pose(input_geo, num_encode=7,is_training=is_training)
            quat_est = tfq.Quaternion(pred_pose[:,0:4])
            R = quat_est.as_rotation_matrix()
            T = tf.expand_dims(pred_pose[:,4:-1]*tf.expand_dims(pred_pose[:,-1],axis=1),axis=2)

        else:
            #Get predicted landmark probability
            R,T,R_det = self.rigid_transform_3D(pred_vis,gt_vis)


        if self.trainer.opt.proj_img:
            filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
            filler = tf.tile(filler, [1, 1, 1])
            transform_mat = tf.concat([R, T], axis=2)
            transform_mat = tf.concat([transform_mat, filler], axis=1)
            output_img,_,_,_,_ = utlr.projective_inverse_warp(tf.expand_dims(data_dict['image'][0,:,:,:],axis=0), depth2[:,:,:,0], transform_mat, tf.expand_dims(data_dict["matK"][1,:,:],axis=0),format='matrix')



        pred_lm_3D = tf.matmul(R,pred_vis)+tf.tile(T,[1,1,tf.shape(pred_vis)[2]])

        #Loss
        num_vis_points = tf.reduce_sum(tf.cast(lm3d_weights,tf.float32))
        tepm = tf.shape(zero_index)[1]
        transformation_loss = l2loss(gt_vis,pred_lm_3D)*pose_weight#+tf.ones([])

        #if not self.trainer.opt.with_geo:
        transformation_loss = tf.cond(tf.less(tf.reduce_sum(tf.cast(lm3d_weights,tf.float32)),tf.ones([],tf.float32)*3.0),lambda:tf.zeros([]),lambda:transformation_loss)
            
        coord_pair = [transformation_loss,tf.shape(zero_depth)[1]]

        #Construct summarie
        
        return transformation_loss,coord_pair