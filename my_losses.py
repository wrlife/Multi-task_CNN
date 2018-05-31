from __future__ import division
import tensorflow as tf
import numpy as np


def l2loss(label,pred,v_weight=None):
    diff = label-pred
    if v_weight is not None:
        diff = tf.multiply(diff,v_weight)
    loss = tf.reduce_mean(diff**2)
    return loss


def compute_loss(pred,pred_landmark,data_dict,FLAGS):

    #=======
    #Depth loss
    #=======
    total_loss = 0
    depth_loss = 0
    landmark_loss = 0
    vis_loss = 0

    depth_weight = 100
    landmark_weight = 1
    vis_weight=10

    label_batch = data_dict['label']
    landmark = data_dict['points2D']
    visibility = data_dict['visibility']


    #Segmentation loss
    for s in range(FLAGS.num_scales):
        curr_label = tf.image.resize_area(label_batch, 
            [int(FLAGS.img_height/(2**s)), int(FLAGS.img_width/(2**s))])
        depth_loss+=l2loss(curr_label,pred[s])#depth_loss += tf.nn.l2_loss(pred[s]-curr_label)/(2**s)#tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=curr_label,output=pred[s]))/(2**s)

    depth_loss = depth_weight*depth_loss

    #Landmark loss
    #v_weight = tf.concat([tf.expand_dims(visibility,2),tf.expand_dims(visibility,2)],axis=2)
    #landmark_loss = l2loss(landmark,pred_landmark[:,:,0:2],v_weight)*landmark_weight
    landmark_weight = 30000.0/tf.reduce_max(landmark)
    landmark_loss = l2loss(landmark*landmark_weight,pred_landmark)
    #vis_loss = l2loss(visibility,pred_landmark[:,:,-1])*vis_weight
    #vis_loss = l2loss(visibility,pred_landmark)*vis_weight


    total_loss = depth_loss+landmark_loss+vis_loss

    return total_loss,depth_loss,landmark_loss,vis_loss

