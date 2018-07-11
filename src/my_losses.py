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
    loss = tf.reduce_sum(diff**2)#/tf.cast(tf.shape(diff)[0]*tf.shape(diff)[1]*tf.shape(diff)[2]*tf.shape(diff)[3],tf.float32)
    return loss

def l2loss_mean(label,pred,v_weight=None):
    diff = label-pred
    if v_weight is not None:
        diff = tf.multiply(diff,v_weight)
    loss = tf.reduce_mean(diff**2)#/tf.cast(tf.shape(diff)[0]*tf.shape(diff)[1]*tf.shape(diff)[2]*tf.shape(diff)[3],tf.float32)
    return loss

def l1loss(label,pred,v_weight=None):
    diff = label-pred
    if v_weight is not None:
        diff = tf.multiply(diff,v_weight)
    loss = tf.reduce_sum(tf.abs(diff))#/tf.cast(tf.shape(diff)[0]*tf.shape(diff)[1]*tf.shape(diff)[2]*tf.shape(diff)[3],tf.float32)
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
        pred_landmark = output[0]
        pred_seg = output[1]

    #=======
    #Depth loss
    #=======
    total_loss = 0
    depth_loss = 0
    landmark_loss = 0
    vis_loss = 0
    geo_loss = 0.0
    dist_loss = 0.0

    depth_weight = 10000
    landmark_weight = FLAGS.img_height*FLAGS.img_width
    dist_weight = 100#FLAGS.img_height*FLAGS.img_width/100.0#1000
    vis_weight=1000
    translation_weight = 100
    quaternion_weight = 5000
    scale_weight = 0.1

    label_batch = data_dict['label']
    landmark = data_dict['points2D']
    visibility = data_dict['visibility']

    quaternion = data_dict['quaternion']
    translation = data_dict['translation']
    depth = data_dict["depth"]

    
    if FLAGS.with_seg:
        depth_loss = depth_weight*l2loss(label_batch,pred_seg)
        
    if FLAGS.model=="multiscale":
        for s in range(FLAGS.num_scales):
            curr_landmark = tf.image.resize_area(landmark, 
                [int(FLAGS.img_height/(2**s)), int(FLAGS.img_width/(2**s))])
            landmark_loss+=l2loss(curr_landmark,pred_landmark[s])/(2**s)*landmark_weight        
    
    elif FLAGS.model=="hourglass":
        landmark_loss = l2loss(data_dict["landmark_init"],pre_landmark_init)*landmark_weight
        landmark_loss = l2loss(landmark,pred_landmark)*landmark_weight + landmark_loss

    else:
        #import pdb;pdb.set_trace()
        lm3d_weights = tf.clip_by_value(visibility,0.0,1.0)
        lm3d_weights = tf.expand_dims(lm3d_weights,axis=1)
        lm3d_weights = tf.expand_dims(lm3d_weights,axis=2)
        #import pdb;pdb.set_trace()
        lm3d_weights = tf.tile(lm3d_weights,[1,FLAGS.img_height,FLAGS.img_width,1])

        landmark = landmark*lm3d_weights
        #landmark_loss = l2loss_mean(landmark,pred_landmark)*landmark_weight
    

    #Geometric loss
    # if FLAGS.with_geo:
    #     geo_loss,gt_landmarkdist = geometric_loss(pred_landmark,landmark,depth,visibility,data_dict["matK"])

    if FLAGS.with_vis:
        vis_loss = compute_vis_loss(visibility,output[2])*vis_weight
        if FLAGS.evaluation:
            pred_vis = tf.to_float(tf.expand_dims(output[2],axis=0)>0.5)

            visibility = tf.clip_by_value(visibility,0.0,1.0)     

            vis_loss = tf.reduce_sum(tf.abs(visibility-pred_vis))


    #Local constrain
    #import pdb;pdb.set_trace()
    if FLAGS.with_dist:
        _,H,W,D = pred_landmark.get_shape().as_list()
        pred_landmark.set_shape([FLAGS.batch_size,H,W,D])
        landmark.set_shape([FLAGS.batch_size,H,W,D])
        matK = tf.expand_dims(data_dict['matK'][0,:],axis=0)
        gt_cam_coord,pred_cam_coord,_ = project_2Dlm_to_3D(landmark,pred_landmark,depth,depth,visibility,visibility,matK,matK,FLAGS)
        gt_cam_coord_shift = tf.concat([tf.expand_dims(gt_cam_coord[:,:,-1],axis=2),gt_cam_coord[:,:,0:-1]],axis=2)
        pred_cam_coord_shift = tf.concat([tf.expand_dims(pred_cam_coord[:,:,-1],axis=2),pred_cam_coord[:,:,0:-1]],axis=2)
        gt_landmarkdist = gt_cam_coord-gt_cam_coord_shift
        pred_landmarkdist = pred_cam_coord-pred_cam_coord_shift
        dist_loss = l2loss(gt_landmarkdist,pred_landmarkdist)*dist_weight
        tf.summary.scalar('losses/dist_loss', dist_loss) 

    total_loss = depth_loss+landmark_loss+vis_loss+geo_loss+dist_loss

    return total_loss,depth_loss,landmark_loss,vis_loss,geo_loss



def project_2Dlm_to_3D(landmark1,landmark2,depth1,depth2,visibility1,visibility2,matK1,matK2,FLAGS,min_thresh=0.1,with_gtvis=True,with_pose=True):

    B,H,W,D = landmark1.get_shape().as_list()#tf.shape(landmark1)

    visibility1.set_shape([B,D])
    visibility2.set_shape([B,D])
    #Soft arg-max operation
    #import pdb;pdb.set_trace()
    if with_pose:
        norm_to_regular = tf.concat([tf.ones([B,D,1])*H, tf.ones([B,D,1])*W],axis=2)
        lm1_coord = tf.reverse(tf.transpose(tf.reshape((tf.contrib.layers.spatial_softmax(landmark1,temperature=1.0/(FLAGS.img_height*FLAGS.img_width),trainable=False)+1)/2.0,[B,D,2])*norm_to_regular,[0,2,1]),[1])
        lm2_coord = tf.reverse(tf.transpose(tf.reshape((tf.contrib.layers.spatial_softmax(landmark2,temperature=1.0/(FLAGS.img_height*FLAGS.img_width),trainable=False)+1)/2.0,[B,D,2])*norm_to_regular,[0,2,1]),[1])

        gt_lm_coord = lm1_coord
        pred_lm_coord = lm2_coord
    else:
        pred_lm_coord = tf.reverse(argmax_2d(landmark2),[1])
        gt_lm_coord = tf.reverse(argmax_2d(landmark1),[1])

    #Extract depth value at landmark locations
    batch_index = tf.tile(tf.expand_dims(tf.range(B), 1), [1, D])
    index_gt = tf.concat([tf.expand_dims(batch_index,axis=2), tf.transpose(tf.reverse(tf.to_int32(gt_lm_coord),[1]),[0,2,1])], axis=2)
    index_pred = tf.concat([tf.expand_dims(batch_index,axis=2), tf.transpose(tf.reverse(tf.to_int32(pred_lm_coord),[1]),[0,2,1])], axis=2)
    gt_depth_val = tf.gather_nd(depth1,index_gt)
    pred_depth_val = tf.gather_nd(depth2,index_pred)

    #Get mutually visible points
    if with_gtvis:
        lm3d_weights = tf.clip_by_value(visibility1,0.0,1.0)
        lm3d_weights = lm3d_weights*tf.clip_by_value(visibility2,0.0,1.0)
    else:
        lm1_val = tf.gather_nd(landmark1,index_gt)
        lm2_val = tf.gather_nd(landmark2,index_pred)

        lm1_val_sup = tf.expand_dims(lm1_val[:,0,0],axis=1)
        lm2_val_sup = tf.expand_dims(lm2_val[:,0,0],axis=1)
        for ii in range(1,D):
            lm1_val_sup = tf.concat([lm1_val_sup,tf.expand_dims(lm1_val[:,ii,ii],axis=1)],axis=1)
            lm2_val_sup = tf.concat([lm2_val_sup,tf.expand_dims(lm2_val[:,ii,ii],axis=1)],axis=1)                

        pred_vis1 = tf.to_float(lm1_val_sup>(tf.maximum(tf.reduce_max(landmark1)/5.0,min_thresh)))
        pred_vis2 = tf.to_float(lm2_val_sup>(tf.maximum(tf.reduce_max(landmark2)/5.0,min_thresh)))
        lm3d_weights = pred_vis1
        lm3d_weights = lm3d_weights*pred_vis2

    #import pdb;pdb.set_trace()
    #mutual invis and depth zero
    mutualdepth = gt_depth_val*pred_depth_val

    usable_points = tf.logical_and(
                            tf.greater(mutualdepth[:,:,0],tf.ones([],tf.float32)*10.0),
                            tf.equal(lm3d_weights[:],tf.ones([],tf.float32)))

    zero_depth =  tf.where(usable_points)
    usable_points = tf.reduce_sum(tf.to_int32(usable_points))
                                
               
    #zero_index = tf.tile(tf.expand_dims(tf.range(B), 1), [1, tf.shape(zero_depth)[1]])
    #zero_depth = tf.concat([tf.expand_dims(zero_index, axis=2), tf.cast(zero_depth,tf.int32)], axis=2)
    gt_depth_val = tf.expand_dims(tf.gather_nd(gt_depth_val,zero_depth),axis=0)
    gt_lm_coord = tf.transpose(tf.expand_dims(tf.gather_nd(tf.transpose(gt_lm_coord,[0,2,1]),zero_depth),axis=0),[0,2,1])
    pred_depth_val = tf.expand_dims(tf.gather_nd(pred_depth_val,zero_depth),axis=0)
    pred_lm_coord = tf.transpose(tf.expand_dims(tf.gather_nd(tf.transpose(pred_lm_coord,[0,2,1]),zero_depth),axis=0),[0,2,1])


    #Project 2D to 3D
    ones = tf.ones([1, 1, tf.shape(zero_depth)[0]])
    pred_lm_coord = tf.concat([tf.cast(pred_lm_coord,tf.float32),ones],axis=1)
    gt_lm_coord = tf.concat([tf.cast(gt_lm_coord,tf.float32),ones],axis=1)
    gt_cam_coord = pixel2cam(gt_depth_val,gt_lm_coord,matK1)
    pred_cam_coord = pixel2cam(pred_depth_val,pred_lm_coord,matK2)

    return gt_cam_coord,pred_cam_coord,usable_points


def geometric_loss(pred_landmark,landmark,depth,visibility,matK):

        #Get gt and estimate landmark locations

        norm_to_regular = tf.concat([tf.ones([1,28,1])*480, tf.ones([1,28,1])*640],axis=2)
        gt_lm_coord = tf.reverse(tf.transpose(tf.reshape((tf.contrib.layers.spatial_softmax(landmark,temperature=1.0)+1)/2.0,[1,28,2])*norm_to_regular,[0,2,1]),[1])
        pred_lm_coord = tf.reverse(tf.transpose(tf.reshape((tf.contrib.layers.spatial_softmax(pred_landmark,temperature=1.0)+1)/2.0,[1,28,2])*norm_to_regular,[0,2,1]),[1])

        batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(pred_landmark)[0]), 1), [1, tf.shape(pred_landmark)[3]])
        index_gt = tf.concat([tf.expand_dims(batch_index,axis=2), tf.transpose(tf.reverse(gt_lm_coord,[1]),[0,2,1])], axis=2)
        index_pred = tf.concat([tf.expand_dims(batch_index,axis=2), tf.transpose(tf.reverse(pred_lm_coord,[1]),[0,2,1])], axis=2)

        gt_depth_val = tf.gather_nd(depth[:,:,:,0],index_gt)
        pred_depth_val = tf.gather_nd(depth[:,:,:,0],index_pred)
        
        ones = tf.ones([tf.shape(pred_landmark)[0], 1, tf.shape(pred_landmark)[3]])
        pred_lm_coord = tf.concat([tf.cast(pred_lm_coord,tf.float32),ones],axis=1)
        gt_lm_coord = tf.concat([tf.cast(gt_lm_coord,tf.float32),ones],axis=1)

        gt_cam_coord = pixel2cam(gt_depth_val,gt_lm_coord,matK)
        pred_cam_coord = pixel2cam(pred_depth_val,pred_lm_coord,matK)

        #import pdb;pdb.set_trace()
        gt_cam_coord_shift = tf.concat([tf.expand_dims(gt_cam_coord[:,:,-1],axis=2),gt_cam_coord[:,:,0:-1]],axis=2)
        pred_cam_coord_shift = tf.concat([tf.expand_dims(pred_cam_coord[:,:,-1],axis=2),pred_cam_coord[:,:,0:-1]],axis=2)

        lm3d_weights = tf.clip_by_value(visibility,0.0,1.5)
        lm3d_weights = tf.tile(tf.expand_dims(lm3d_weights,axis=1),[1,3,1])

        gt_landmarkdist = tf.sqrt(tf.reduce_sum((gt_cam_coord-gt_cam_coord_shift)**2,axis=1))
        pred_landmarkdist = tf.sqrt(tf.reduce_sum((pred_cam_coord-pred_cam_coord_shift)**2,axis=1))
        geoloss = l2loss(gt_landmarkdist,pred_landmarkdist,lm3d_weights)


        return geoloss,gt_landmarkdist

def compute_vis_loss(visibility,pred_visibility):
    vis_loss = l2loss(visibility,pred_visibility)
    return vis_loss



def camera_loss(landmark1,landmark2,pose,data_dict):

    visibility = data_dict['visibility']

    quaternion = data_dict['quaternion']
    translation = data_dict['translation']
    depth = data_dict["depth"]


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

