from __future__ import division
import tensorflow as tf
import numpy as np
import tfquaternion as tfq


def l2loss(label,pred,v_weight=None):
    diff = label-pred
    if v_weight is not None:
        diff = tf.multiply(diff,v_weight)
    loss = tf.reduce_mean(diff**2)#/tf.cast(tf.shape(diff)[0]*tf.shape(diff)[1]*tf.shape(diff)[2]*tf.shape(diff)[3],tf.float32)
    return loss


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  batch, height, width = depth.get_shape().as_list()
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords


def compute_loss(output,data_dict,FLAGS):

    pred = output[0]
    pred_landmark = output[1]
    if FLAGS.model=="pose":
        pose = output[2]
    elif FLAGS.model=="hourglass":
        pre_landmark_init = output[0][1]
        pre_landmark = output[1][1]

    #=======
    #Depth loss
    #=======
    total_loss = 0
    depth_loss = 0
    landmark_loss = 0
    vis_loss = 0
    quaternion_loss = 0
    translation_loss = 0

    depth_weight = 10000
    landmark_weight = 1
    vis_weight=10
    translation_weight = 100
    quaternion_weight = 5000
    scale_weight = 0.1

    label_batch = data_dict['label']
    landmark = data_dict['points2D']
    visibility = data_dict['visibility']
    quaternion = data_dict['quaternion']
    translation = data_dict['translation']


    if FLAGS.with_seg:
        #Segmentation loss
        for s in range(FLAGS.num_scales):
            curr_label = tf.image.resize_area(label_batch, 
                [int(FLAGS.img_height/(2**s)), int(FLAGS.img_width/(2**s))])
            depth_loss+=l2loss(curr_label,pred[s])/(2**s)
        depth_loss = depth_weight*depth_loss

    if FLAGS.model=="pose":
        quat_est = tfq.Quaternion(pose[:,0:4])
        quat_gt = tfq.Quaternion(quaternion)

         
    if FLAGS.model=="multiscale":
        for s in range(FLAGS.num_scales):
            curr_landmark = tf.image.resize_area(landmark, 
                [int(FLAGS.img_height/(2**s)), int(FLAGS.img_width/(2**s))])
            landmark_loss+=l2loss(curr_landmark,pred_landmark[s])/(2**s)*landmark_weight        
    
    elif FLAGS.model=="hourglass":
        landmark_loss = l2loss(data_dict["landmark_init"],pre_landmark_init)*landmark_weight
        landmark_loss = l2loss(landmark,pred_landmark)*landmark_weight + landmark_loss
    
    elif:    
        landmark_loss = l2loss(landmark,pred_landmark)*landmark_weight
    



    total_loss = depth_loss+landmark_loss+vis_loss+quaternion_loss+translation_loss

    return total_loss,depth_loss,landmark_loss,vis_loss,quaternion_loss,translation_loss

