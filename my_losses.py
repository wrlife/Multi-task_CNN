from __future__ import division
import tensorflow as tf
import numpy as np
import tfquaternion as tfq


def rank(tensor):

    # return the rank of a Tensor
    return len(tensor.get_shape())
    
def argmax_2d(tensor):

    # input format: BxHxWxD
    assert rank(tensor) == 4
    
    # flatten the Tensor along the height and width axes
    flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))
    
    # argmax of the flat tensor
    argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)
    
    # convert indexes into 2D coordinates
    argmax_x = argmax // tf.shape(tensor)[2]
    argmax_y = argmax % tf.shape(tensor)[2]
    
    # stack and return 2D coordinates
    return tf.stack((argmax_x, argmax_y), axis=1)

def l2loss(label,pred,v_weight=None):
    diff = label-pred
    if v_weight is not None:
        diff = tf.multiply(diff,v_weight)
    loss = tf.reduce_mean(diff**2)#/tf.cast(tf.shape(diff)[0]*tf.shape(diff)[1]*tf.shape(diff)[2]*tf.shape(diff)[3],tf.float32)
    return loss


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, -1]
    pixel_coords: homogeneous pixel coordinates [batch, 3]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  batch = tf.shape(depth)[0]
  depth = tf.reshape(depth, [batch, 1, -1])
  #pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
#   if is_homogeneous:
#     ones = tf.ones([batch, 1, height*width])
#     cam_coords = tf.concat([cam_coords, ones], axis=1)
  #cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords


def compute_loss(output,data_dict,FLAGS):


    if FLAGS.model=="pose":
        pose = output[2]

    if FLAGS.model=="hourglass":
        pre_landmark_init = output[0][1]
        pred_landmark = output[1][1]
    else:
        pred = output[0]
        pred_landmark = output[1]

    #=======
    #Depth loss
    #=======
    total_loss = 0
    depth_loss = 0
    landmark_loss = 0
    vis_loss = 0
    transformation_loss = 0

    depth_weight = 10000
    landmark_weight = 1
    vis_weight=10
    translation_weight = 100
    quaternion_weight = 5000
    scale_weight = 0.1

    label_batch = data_dict['label']
    landmark = data_dict['points2D']
    visibility = data_dict['visibility']

    visibility

    quaternion = data_dict['quaternion']
    translation = data_dict['translation']
    depth = data_dict["depth"]

    
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

        #Get gt and estimate landmark locations
        pred_lm_coord = tf.reverse(argmax_2d(pred_landmark),[1])
        gt_lm_coord = tf.reverse(argmax_2d(landmark),[1])
        
        #import pdb;pdb.set_trace()

        batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(pred_landmark)[0]), 1), [1, tf.shape(pred_landmark)[3]])
        index_gt = tf.concat([tf.expand_dims(batch_index,axis=2), tf.transpose(tf.reverse(gt_lm_coord,[1]),[0,2,1])], axis=2)
        index_pred = tf.concat([tf.expand_dims(batch_index,axis=2), tf.transpose(tf.reverse(pred_lm_coord,[1]),[0,2,1])], axis=2)


        gt_depth_val = tf.gather_nd(depth[:,:,:,0],index_gt)*100.0
        pred_depth_val = tf.gather_nd(depth[:,:,:,0],index_pred)*100.0
        
        ones = tf.ones([tf.shape(pred_landmark)[0], 1, tf.shape(pred_landmark)[3]])
        pred_lm_coord = tf.concat([tf.cast(pred_lm_coord,tf.float32),ones],axis=1)
        gt_lm_coord = tf.concat([tf.cast(gt_lm_coord,tf.float32),ones],axis=1)

        gt_cam_coord = pixel2cam(gt_depth_val,gt_lm_coord,data_dict["matK"])
        pred_cam_coord = pixel2cam(pred_depth_val,pred_lm_coord,data_dict["matK"])

        #import pdb;pdb.set_trace()
        gt_lm_3D = tf.matmul(quat_gt.as_rotation_matrix(), gt_cam_coord)+tf.tile(tf.expand_dims(translation[:,0:3]*tf.expand_dims(translation[:,-1],axis=1),axis=2),[1,1,tf.shape(gt_cam_coord)[2]])#  +tf.tile(tf.expand_dims(translation[:,0:3]*translation[:,3],axis=2),[1,1,tf.shape(gt_cam_coord)[2]])
        pred_lm_3D = tf.matmul(quat_est.as_rotation_matrix(), pred_lm_coord)+tf.tile(tf.expand_dims(pose[:,4:-1]*tf.expand_dims(pose[:,-1],axis=1),axis=2),[1,1,tf.shape(gt_cam_coord)[2]])#   +tf.tile(tf.expand_dims(pose[:,4:-1]*translation[:,-1],axis=2),[1,1,tf.shape(pred_lm_coord)[2]])

        lm3d_weights = tf.clip_by_value(visibility,0.0,1.5)
        lm3d_weights = tf.tile(tf.expand_dims(lm3d_weights,axis=1),[1,3,1])
        gt_lm_3D = gt_lm_3D*lm3d_weights
        #import pdb;pdb.set_trace()
        transformation_loss = l2loss(gt_lm_3D,pred_lm_3D,lm3d_weights)
        

         
    if FLAGS.model=="multiscale":
        for s in range(FLAGS.num_scales):
            curr_landmark = tf.image.resize_area(landmark, 
                [int(FLAGS.img_height/(2**s)), int(FLAGS.img_width/(2**s))])
            landmark_loss+=l2loss(curr_landmark,pred_landmark[s])/(2**s)*landmark_weight        
    
    elif FLAGS.model=="hourglass":
        landmark_loss = l2loss(data_dict["landmark_init"],pre_landmark_init)*landmark_weight
        landmark_loss = l2loss(landmark,pred_landmark)*landmark_weight + landmark_loss
    
    else:
        landmark_loss = l2loss(landmark,pred_landmark)*landmark_weight
    



    total_loss = depth_loss+landmark_loss+vis_loss+transformation_loss

    return total_loss,depth_loss,landmark_loss,vis_loss,transformation_loss

