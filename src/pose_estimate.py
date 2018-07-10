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

    def proj_img(self,R,T,image,depth,matK):

        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [1, 1, 1])
        transform_mat = tf.concat([R, T], axis=2)
        transform_mat = tf.concat([transform_mat, filler], axis=1)
        output_img,_,_,_,_ = utlr.projective_inverse_warp(image,depth, transform_mat, matK ,format='matrix')
        image = tf.summary.image('proj' , \
                        output_img)        
        image = tf.summary.image('tgt' , \
                        image)

    def est_pose(self,landmark1,landmark2):

        input_geo1 = tf.concat([landmark1,landmark2],axis=3)
        pred_pose1 = disp_net_pose(input_geo1, num_encode=7,is_training=is_training)
        quat_est1 = tfq.Quaternion(pred_pose1[:,0:4])
        R1 = quat_est1.as_rotation_matrix()
        T1 = tf.expand_dims(pred_pose1[:,4:-1]*tf.expand_dims(pred_pose1[:,-1],axis=1),axis=2)

        #Project right to left
        input_geo2 = tf.concat([landmark2,landmark1],axis=3)
        pred_pose2 = disp_net_pose(input_geo2, num_encode=7,is_training=is_training)
        quat_est2 = tfq.Quaternion(pred_pose2[:,0:4])
        R2 = quat_est2.as_rotation_matrix()
        T2 = tf.expand_dims(pred_pose2[:,4:-1]*tf.expand_dims(pred_pose2[:,-1],axis=1),axis=2)


    def process_pose_est(self,landmark1,landmark2,pred_cam_coord1,gt_cam_coord2,pred_cam_coord2,gt_cam_coord1,depth1,depth2,data_dict,pose_weight):

        rotation_loss = 0.0
        if self.trainer.opt.with_geo:
            #Project left to right
            R1,T1 = est_pose(landmark1,landmark2)
            R2,T2 = est_pose(landmark2,landmark1)
        else:
            #Get predicted landmark probability
            R1,T1,R_det1 = self.rigid_transform_3D(pred_cam_coord2,gt_cam_coord1)
            R2,T2,R_det2 = self.rigid_transform_3D(pred_cam_coord1,gt_cam_coord2)

        if self.trainer.opt.proj_img:
            proj_img(R1,T1,
                    tf.expand_dims(data_dict['image'][0,:,:,:],axis=0),
                    depth2[:,:,:,0],
                    tf.expand_dims(data_dict["matK"][1,:,:],axis=0))

            proj_img(R2,T2,
                    tf.expand_dims(data_dict['image'][1,:,:,:],axis=0),
                    depth1[:,:,:,0],
                    tf.expand_dims(data_dict["matK"][0,:,:],axis=0))


        pred_cam_coord2_tran = tf.matmul(R1,pred_cam_coord2)+tf.tile(T1,[1,1,tf.shape(pred_cam_coord2)[2]])
        pred_cam_coord1_tran = tf.matmul(R2,pred_cam_coord1)+tf.tile(T2,[1,1,tf.shape(pred_cam_coord1)[2]])
        #Loss
        num_vis_points = tf.reduce_sum(tf.cast(lm3d_weights,tf.float32))
        #tepm = tf.shape(zero_index)[1]
        transformation_loss = l2loss_mean(gt_cam_coord1,pred_cam_coord2_tran)*pose_weight
        transformation_loss = transformation_loss+l2loss_mean(gt_cam_coord2,pred_cam_coord1_tran)*pose_weight

        #Cycle consist
        if self.trainer.opt.cycle_consist:
            pred_cam_coord2_cycle = tf.matmul(R2,pred_cam_coord2_tran)+tf.tile(T2,[1,1,tf.shape(pred_cam_coord2_tran)[2]])
            pred_cam_coord1_cycle = tf.matmul(R1,pred_cam_coord1_tran)+tf.tile(T1,[1,1,tf.shape(pred_cam_coord1_tran)[2]])

            rotation_loss = l2loss_mean(pred_cam_coord2_cycle,pred_cam_coord2)*pose_weight
            rotation_loss = l2loss_mean(pred_cam_coord1_cycle,pred_cam_coord1)*pose_weight+rotation_loss

        transformation_loss = rotation_loss+transformation_loss

        return transformation_loss

    
    def forward_wrapper(self,output,data_dict,pose_weight,is_training=True):
        '''
        A wrapper function for domain transfer.
        Generate dataloader, model, loss and its own summary
        '''

        #Landmark heat maps
        landmark1 = tf.expand_dims(output[0,:,:,:],axis=0)
        landmark2 = tf.expand_dims(output[1,:,:,:],axis=0)
    
        #GT Landmark heat maps
        gtlandmark1 = tf.expand_dims(data_dict["points2D"][0,:,:,:],axis=0)
        gtlandmark2 = tf.expand_dims(data_dict["points2D"][1,:,:,:],axis=0)

        #Depth maps
        depth1 = tf.expand_dims(data_dict["depth"][0,:,:,:],axis=0)
        depth2 = tf.expand_dims(data_dict["depth"][1,:,:,:],axis=0)

        #Get GT visibility
        visibility1 = tf.expand_dims(data_dict['visibility'][0,:],axis=0)
        visibility2 = tf.expand_dims(data_dict['visibility'][1,:],axis=0)

        matK1 = tf.expand_dims(data_dict['matK'][0,:],axis=0)
        matK2 = tf.expand_dims(data_dict['matK'][1,:],axis=0)        

        #Get GT rotation and translation
        quaternion = data_dict['quaternion']
        translation = data_dict['translation']

        
        translation_loss = 0.0

        #min_thresh = tf.constant(5000.0)
        #import pdb;pdb.set_trace()
        pred_cam_coord1,gt_cam_coord2,usable_points1 = project_2Dlm_to_3D(landmark1,gtlandmark2,depth1,depth2,visibility1,visibility2,matK1,matK2,self.trainer.opt)

        gt_cam_coord1,pred_cam_coord2,usable_points2 = project_2Dlm_to_3D(gtlandmark1,landmark2,depth1,depth2,visibility1,visibility2,matK1,matK2,self.trainer.opt)

        # pred_vis = pred_cam_coord
        # gt_vis = gt_cam_coord

        transformation_loss = tf.cond(tf.logical_and(tf.greater(usable_points1,tf.ones([],tf.float32)*3.0),
                                                    tf.greater(usable_points2,tf.ones([],tf.float32)*3.0)
                                                    ),
                                      lambda:process_pose_est(landmark1,landmark2,pred_cam_coord1,gt_cam_coord2,pred_cam_coord2,gt_cam_coord1,depth1,depth2,data_dict,pose_weight),
                                      lambda:tf.zeros([])
                               )

        # lm3d_weights = tf.clip_by_value(visibility1,0.0,1.0)
        # lm3d_weights = lm3d_weights*tf.clip_by_value(visibility2,0.0,1.0)
        # vis_ind = tf.expand_dims(tf.where(tf.equal(lm3d_weights[0],tf.ones([],tf.float32))),axis=0)
        # batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(landmark1)[0]), 1), [1, tf.shape(vis_ind)[1]])
        # vis_ind = tf.concat([tf.expand_dims(batch_index,axis=2), tf.cast(vis_ind,tf.int32)], axis=2)
        # gt_vis = tf.transpose(tf.gather_nd(tf.transpose(gt_cam_coord,[0,2,1]),vis_ind),[0,2,1])
        # pred_vis = tf.transpose(tf.gather_nd(tf.transpose(pred_cam_coord,[0,2,1]),vis_ind),[0,2,1])


        #if not self.trainer.opt.with_geo:
        # transformation_loss = tf.cond(tf.less(tf.reduce_sum(tf.cast(lm3d_weights,tf.float32)),tf.ones([],tf.float32)*3.0),lambda:tf.zeros([]),lambda:transformation_loss)
        # rotation_loss = tf.cond(tf.less(tf.reduce_sum(tf.cast(lm3d_weights,tf.float32)),tf.ones([],tf.float32)*3.0),lambda:tf.zeros([]),lambda:rotation_loss)


        coord_pair = [transformation_loss]


        #Construct summarie
        
        return transformation_loss,coord_pair
