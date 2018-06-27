from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np


# Range of disparity/inverse depth values
DISP_SCALING = 100.0
MIN_DISP = 0

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()
  
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias


def conv_encoder(num_encode,input_,num_features,max_features=512,with_b = True):
    '''
    Convolutional encoder
    '''

    cnv_layers = []
    #import pdb;pdb.set_trace()
    for i in range(num_encode):

        #Upper bound for max number of features
        if num_features*(2**i)>max_features:
            curr_features = max_features
        else:
            curr_features = num_features*(2**i)
        cnv = slim.conv2d(input_, curr_features,  [3, 3], stride=2, scope='cnv'+str(i+1))
        if with_b:
            cnvb = slim.conv2d(cnv, curr_features,  [3, 3], stride=1, scope='cnv'+str(i+1)+'b')
            input_ = cnvb
            cnv_layers.append(cnvb)
        else:
            input_ = cnv
            cnv_layers.append(cnv)
        
    
    return cnv_layers

def conv_decoder(num_encode,cnv_layers,num_features,num_out_channel=1,max_features=512,min_features=32):
    '''
    Convolutional decoder
    num_out_channel is the final output channel
    '''

    input_ = cnv_layers[-1]

    decnv_layers = []
    for i in range(num_encode-1,0,-1):

        if num_features*(2**i)>max_features:
            curr_features = max_features
        elif num_features*(2**i)<min_features:
            curr_features = min_features
        else:
            curr_features = num_features*(2**i)            

        upcnv = slim.conv2d_transpose(input_, curr_features, [3, 3], stride=2, scope='upcnv'+str(i+1))
        upcnv = resize_like(upcnv, cnv_layers[i-1])
        i_in  = tf.concat([upcnv, cnv_layers[i-1]], axis=3)
        icnv  = slim.conv2d(i_in, curr_features, [3, 3], stride=1, scope='icnv'+str(i+1))
        input_ = icnv
        decnv_layers.append(icnv)

    upcnv = slim.conv2d_transpose(input_, np.maximum(num_features,min_features), [3, 3], stride=2, scope='upcnv1')
    icnv  = slim.conv2d(upcnv, np.maximum(num_features,min_features), [3, 3], stride=1, scope='icnv1')
    disp  = slim.conv2d(icnv, num_out_channel,   [3, 3], stride=1, 
        activation_fn=None, normalizer_fn=None, scope='disp1')
    decnv_layers.append(icnv)
    
    return disp,decnv_layers


def disp_net(tgt_image, is_training=True, is_reuse=False):
    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net',reuse = is_reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

            # with tf.variable_scope('landmark'):
            #     landmark_cnv7  = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cam_cnv7')
            #     landmark_pred = slim.conv2d(landmark_cnv7, 28*3, [1, 1], scope='pred', 
            #         stride=1, normalizer_fn=None, activation_fn=None)
            #     landmark_avg = tf.reduce_mean(landmark_pred, [1, 2])
            #     # Empirically we found that scaling by a small constant 
            #     # facilitates training.
            #     landmark_final = tf.reshape(landmark_avg, [-1, 28, 3])            

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4')# + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            #upcnv4_hm = slim.conv2d_transpose(icnv5, 128,  [3, 3], stride=2, scope='upcnv4_hm')
            #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
            #icnv4_hm  = slim.conv2d(upcnv4_hm, 64,  [3, 3], stride=1, scope='icnv4_hm')
            #landmark4_hm  = slim.conv2d(icnv4_hm, 28, [3, 3], stride=1, 
            #    activation_fn=None, normalizer_fn=None, scope='disp4_hm')# + MIN_DISP

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3')# + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            #upcnv3_hm = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3_hm')
            #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
            #icnv3_hm  = slim.conv2d(upcnv3_hm, 32,  [3, 3], stride=1, scope='icnv3_hm')
            #landmark3_hm  = slim.conv2d(icnv3_hm, 28, [3, 3], stride=1, 
            #    activation_fn=None, normalizer_fn=None, scope='disp3_hm')# + MIN_DISP

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2')# + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])
            
            #upcnv2_hm = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2_hm')
            #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
            #icnv2_hm  = slim.conv2d(upcnv2_hm, 28,  [3, 3], stride=1, scope='icnv2_hm')
            #landmark2_hm  = slim.conv2d(icnv2_hm, 28, [3, 3], stride=1, 
            #    activation_fn=None, normalizer_fn=None, scope='disp2_hm')# + MIN_DISP

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')# + MIN_DISP


            
            
            for i in range(28):
              upcnv1_hm = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1_hm'+str(i))
              #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
              icnv1_hm  = slim.conv2d(upcnv1_hm, 16,  [3, 3], stride=1, scope='icnv1_hm'+str(i))
              landmark1_hm  = slim.conv2d(icnv1_hm, 1, [3, 3], stride=1, 
                  activation_fn=None, normalizer_fn=None, scope='disp1_hm'+str(i))# + MIN_DISP
              
              if i==0:
                land_mark=landmark1_hm
              else:
                land_mark = tf.concat([land_mark,landmark1_hm],axis=3)
              

            
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], land_mark, end_points #,landmark2_hm,landmark3_hm,landmark4_hm]



def disp_net_single(tgt_image, num_encode, num_features=32,num_out_channel=28, is_training=True, is_reuse=False,with_vis=False,with_seg=False):
    batch_norm_params = {'is_training': is_training,'decay':0.9}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    max_features=512
    with tf.variable_scope('depth_net',reuse = tf.AUTO_REUSE) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.leaky_relu,
                            outputs_collections=end_points_collection):
            input_ = tgt_image
            cnv_layers = conv_encoder(num_encode,input_,num_features,max_features=max_features)

            output = []
            if with_vis:
                cnv_flat = tf.reduce_mean(cnv_layers[-1], [1, 2])
                #pose_final = tf.reshape(pose_avg, [-1, 8])              
                #cnv_flat = tf.reshape(cnv_layers[-1], [-1, int((H/2**(num_encode))*(W/2**(num_encode))*np.maximum(num_features*(2**(num_encode-1)),max_features))])

                fc1 = tf.layers.dense(inputs=cnv_flat, units=max_features, activation=tf.nn.leaky_relu)
                fc = tf.layers.dense(inputs=fc1, units=28, activation=tf.sigmoid)

            if with_seg:
                num_out_channel = num_out_channel+1

            landmark,decnv_layers = conv_decoder(num_encode,cnv_layers,num_features,num_out_channel=num_out_channel,min_features=256)

            if with_seg:
                landmark = landmark[:,:,:,0:num_out_channel-1]
                pred_seg = tf.expand_dims(landmark[:,:,:,-1],axis=3)
                
            output.append(landmark)
            if with_seg:
                output.append(pred_seg)
            else:
                output.append(decnv_layers[-1])
            
            if with_vis:
                output.append(fc)
            
            return output

def disp_net_pose(tgt_image, num_encode, num_features=32, is_training=True, is_reuse=False):
    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    max_features=512
    with tf.variable_scope('pose_net',reuse = tf.AUTO_REUSE) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
             
            cnv_layers = conv_encoder(num_encode,tgt_image,num_features,max_features=max_features)
            cnv_flat = tf.reduce_mean(cnv_layers[-1], [1, 2])
            #pose_final = tf.reshape(pose_avg, [-1, 8])              
            #cnv_flat = tf.reshape(cnv_layers[-1], [-1, int((H/2**(num_encode))*(W/2**(num_encode))*np.maximum(num_features*(2**(num_encode-1)),max_features))])
            fc1 = tf.layers.dense(inputs=cnv_flat, units=max_features, activation=tf.nn.leaky_relu)
            pose_final = tf.layers.dense(inputs=fc1, units=8, activation=None)

            return pose_final


def disp_net_initial(tgt_image, is_training=True, is_reuse=False):
    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net_initial',reuse = is_reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')         

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(cnv4b, 256, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 256, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4')
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 256,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 256,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3')
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 256,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 256,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2')
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])
            
            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')

            upcnv1_hm = slim.conv2d_transpose(icnv2, 256,  [3, 3], stride=2, scope='upcnv1_hm')
            icnv1_hm  = slim.conv2d(upcnv1_hm, 256,  [3, 3], stride=1, scope='icnv1_hm')
            land_mark  = slim.conv2d(icnv1_hm, 4, [3, 3], stride=1, 
                  activation_fn=None, normalizer_fn=None, scope='disp1_hm')            
            
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], land_mark, end_points 




def disp_net_refine(tgt_image, is_training=True, is_reuse=False):
    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net_refine',reuse = is_reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')         

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(cnv4b, 256, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 256, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4')
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 256,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 256,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3')
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 256,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 256,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2')
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])
            
            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')

            upcnv1_hm = slim.conv2d_transpose(icnv2, 256,  [3, 3], stride=2, scope='upcnv1_hm')
            icnv1_hm  = slim.conv2d(upcnv1_hm, 256,  [3, 3], stride=1, scope='icnv1_hm')
            land_mark  = slim.conv2d(icnv1_hm, 28, [3, 3], stride=1, 
                  activation_fn=None, normalizer_fn=None, scope='disp1_hm')            
            
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], land_mark, end_points




def disp_net_single_pose(tgt_image, is_training=True, is_reuse=False):
    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net',reuse = is_reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

            with tf.variable_scope('pose'):
                pose_cnv7  = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cam_cnv7')
                pose_pred = slim.conv2d(pose_cnv7, 8, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = tf.reshape(pose_avg, [-1, 8])            

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(cnv4b, 256, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 256, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4')# + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            #upcnv4_hm = slim.conv2d_transpose(icnv5, 128,  [3, 3], stride=2, scope='upcnv4_hm')
            #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
            #icnv4_hm  = slim.conv2d(upcnv4_hm, 64,  [3, 3], stride=1, scope='icnv4_hm')
            #landmark4_hm  = slim.conv2d(icnv4_hm, 28, [3, 3], stride=1, 
            #    activation_fn=None, normalizer_fn=None, scope='disp4_hm')# + MIN_DISP

            upcnv3 = slim.conv2d_transpose(icnv4, 256,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 256,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3')# + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            #upcnv3_hm = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3_hm')
            #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
            #icnv3_hm  = slim.conv2d(upcnv3_hm, 32,  [3, 3], stride=1, scope='icnv3_hm')
            #landmark3_hm  = slim.conv2d(icnv3_hm, 28, [3, 3], stride=1, 
            #    activation_fn=None, normalizer_fn=None, scope='disp3_hm')# + MIN_DISP

            upcnv2 = slim.conv2d_transpose(icnv3, 256,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 256,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2')# + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])
            
            #upcnv2_hm = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2_hm')
            #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
            #icnv2_hm  = slim.conv2d(upcnv2_hm, 28,  [3, 3], stride=1, scope='icnv2_hm')
            #landmark2_hm  = slim.conv2d(icnv2_hm, 28, [3, 3], stride=1, 
            #    activation_fn=None, normalizer_fn=None, scope='disp2_hm')# + MIN_DISP

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')# + MIN_DISP



            upcnv1_hm = slim.conv2d_transpose(icnv2, 256,  [3, 3], stride=2, scope='upcnv1_hm')
            #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
            icnv1_hm  = slim.conv2d(upcnv1_hm, 256,  [3, 3], stride=1, scope='icnv1_hm')
            land_mark  = slim.conv2d(icnv1_hm, 28, [3, 3], stride=1, 
                  activation_fn=None, normalizer_fn=None, scope='disp1_hm')# + MIN_DISP
              
            
            
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], land_mark,pose_final, end_points #,landmark2_hm,landmark3_hm,landmark4_hm]


def disp_net_multi_decoder(tgt_image, is_training=True, is_reuse=False):
    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net',reuse = is_reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            #cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            #cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            #cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            #cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

            # with tf.variable_scope('landmark'):
            #     landmark_cnv7  = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cam_cnv7')
            #     landmark_pred = slim.conv2d(landmark_cnv7, 28*3, [1, 1], scope='pred', 
            #         stride=1, normalizer_fn=None, activation_fn=None)
            #     landmark_avg = tf.reduce_mean(landmark_pred, [1, 2])
            #     # Empirically we found that scaling by a small constant 
            #     # facilitates training.
            #     landmark_final = tf.reshape(landmark_avg, [-1, 28, 3])            

            for i in range(28):

              #upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7'+str(i))
              # There might be dimension mismatch due to uneven down/up-sampling
              #upcnv7 = resize_like(upcnv7, cnv6b)
              #i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
              #icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7'+str(i))
  
              #upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6'+str(i))
              #upcnv6 = resize_like(upcnv6, cnv5b)
              #i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
              #icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6'+str(i))
  
              upcnv5 = slim.conv2d_transpose(cnv5b, 256, [3, 3], stride=2, scope='upcnv5'+str(i))
              upcnv5 = resize_like(upcnv5, cnv4b)
              i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
              icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5'+str(i))
  
              upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4'+str(i))
              i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
              icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4'+str(i))
              disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                  activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4'+str(i))# + MIN_DISP
              disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])
  
              #upcnv4_hm = slim.conv2d_transpose(icnv5, 128,  [3, 3], stride=2, scope='upcnv4_hm')
              #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
              #icnv4_hm  = slim.conv2d(upcnv4_hm, 64,  [3, 3], stride=1, scope='icnv4_hm')
              #landmark4_hm  = slim.conv2d(icnv4_hm, 28, [3, 3], stride=1, 
              #    activation_fn=None, normalizer_fn=None, scope='disp4_hm')# + MIN_DISP
  
              upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3'+str(i))
              i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
              icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3'+str(i))
              disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                  activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3'+str(i))# + MIN_DISP
              disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])
  
              #upcnv3_hm = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3_hm')
              #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
              #icnv3_hm  = slim.conv2d(upcnv3_hm, 32,  [3, 3], stride=1, scope='icnv3_hm')
              #landmark3_hm  = slim.conv2d(icnv3_hm, 28, [3, 3], stride=1, 
              #    activation_fn=None, normalizer_fn=None, scope='disp3_hm')# + MIN_DISP
  
              upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2'+str(i))
              i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
              icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2'+str(i))
              disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                  activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2'+str(i))# + MIN_DISP
              disp2_up = tf.image.resize_bilinear(disp2, [H, W])
              
              #upcnv2_hm = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2_hm')
              #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
              #icnv2_hm  = slim.conv2d(upcnv2_hm, 28,  [3, 3], stride=1, scope='icnv2_hm')
              #landmark2_hm  = slim.conv2d(icnv2_hm, 28, [3, 3], stride=1, 
              #    activation_fn=None, normalizer_fn=None, scope='disp2_hm')# + MIN_DISP
  
              upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1'+str(i))
              i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
              icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1'+str(i))
              disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                  activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1'+str(i))# + MIN_DISP

              upcnv1_hm = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1_hm'+str(i))
              #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
              icnv1_hm  = slim.conv2d(upcnv1_hm, 16,  [3, 3], stride=1, scope='icnv1_hm'+str(i))
              landmark1_hm  = slim.conv2d(icnv1_hm, 1, [3, 3], stride=1, 
                  activation_fn=None, normalizer_fn=None, scope='disp1_hm'+str(i))# + MIN_DISP
              
              if i==0:
                land_mark=landmark1_hm
              else:
                land_mark = tf.concat([land_mark,landmark1_hm],axis=3)
              

            
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], land_mark, end_points #,landmark2_hm,landmark3_hm,landmark4_hm]





def disp_net_single_multiscale(tgt_image, is_training=True, is_reuse=False):
    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net',reuse = is_reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

            # with tf.variable_scope('landmark'):
            #     landmark_cnv7  = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cam_cnv7')
            #     landmark_pred = slim.conv2d(landmark_cnv7, 28*3, [1, 1], scope='pred', 
            #         stride=1, normalizer_fn=None, activation_fn=None)
            #     landmark_avg = tf.reduce_mean(landmark_pred, [1, 2])
            #     # Empirically we found that scaling by a small constant 
            #     # facilitates training.
            #     landmark_final = tf.reshape(landmark_avg, [-1, 28, 3])            

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(cnv4b, 256, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 256, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 28,   [3, 3], stride=1, 
                activation_fn=None, normalizer_fn=None, scope='disp4')# + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            landmark4 = disp4

            #upcnv4_hm = slim.conv2d_transpose(icnv5, 128,  [3, 3], stride=2, scope='upcnv4_hm')
            #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
            #icnv4_hm  = slim.conv2d(upcnv4_hm, 64,  [3, 3], stride=1, scope='icnv4_hm')
            #landmark4_hm  = slim.conv2d(icnv4_hm, 28, [3, 3], stride=1, 
            #    activation_fn=None, normalizer_fn=None, scope='disp4_hm')# + MIN_DISP

            upcnv3 = slim.conv2d_transpose(icnv4, 256,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 256,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 28,   [3, 3], stride=1, 
                activation_fn=None, normalizer_fn=None, scope='disp3')# + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            landmark3 = disp3

            #upcnv3_hm = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3_hm')
            #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
            #icnv3_hm  = slim.conv2d(upcnv3_hm, 32,  [3, 3], stride=1, scope='icnv3_hm')
            #landmark3_hm  = slim.conv2d(icnv3_hm, 28, [3, 3], stride=1, 
            #    activation_fn=None, normalizer_fn=None, scope='disp3_hm')# + MIN_DISP

            upcnv2 = slim.conv2d_transpose(icnv3, 256,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 256,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 28,   [3, 3], stride=1, 
                activation_fn=None, normalizer_fn=None, scope='disp2')# + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            landmark2 = disp2
            
            #upcnv2_hm = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2_hm')
            #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
            #icnv2_hm  = slim.conv2d(upcnv2_hm, 28,  [3, 3], stride=1, scope='icnv2_hm')
            #landmark2_hm  = slim.conv2d(icnv2_hm, 28, [3, 3], stride=1, 
            #    activation_fn=None, normalizer_fn=None, scope='disp2_hm')# + MIN_DISP

            upcnv1 = slim.conv2d_transpose(icnv2, 256,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 256,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 28,   [3, 3], stride=1, 
                activation_fn=None, normalizer_fn=None, scope='disp1')# + MIN_DISP

            landmark1 = disp1

            # upcnv1_hm = slim.conv2d_transpose(icnv2, 256,  [3, 3], stride=2, scope='upcnv1_hm')
            # #i1_in_hm  = tf.concat([upcnv1_hm], axis=3)
            # icnv1_hm  = slim.conv2d(upcnv1_hm, 256,  [3, 3], stride=1, scope='icnv1_hm')
            # land_mark1  = slim.conv2d(icnv1_hm, 28, [3, 3], stride=1, 
            #       activation_fn=None, normalizer_fn=None, scope='disp1_hm')# + MIN_DISP
              
            
            
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], [landmark1,landmark2,landmark3,landmark4], end_points #,landmark2_hm,landmark3_hm,landmark4_hm]



def discriminator(tgt_image, num_encode, num_features=32, is_training=True, is_reuse=False,max_features=512):
    batch_norm_params = {'is_training': is_training,'decay':0.9}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value

    with tf.variable_scope('disc_net',reuse = is_reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.leaky_relu,
                            outputs_collections=end_points_collection):
            
            #import pdb;pdb.set_trace()
            input_ = tgt_image
            cnv_layers = conv_encoder(num_encode,input_,num_features,with_b=False)
            #fc = linear(tf.reshape(cnv_layers[-1], [-1, int((H/2**(num_encode))*(W/2**(num_encode))*np.maximum(num_features*(2**(num_encode-1)),max_features))]), 1, 'd_h4_lin')
            #import pdb;pdb.set_trace()
            cnvout = slim.conv2d(cnv_layers[-1], 1,  [3, 3], stride=1, scope='cnv'+str(num_encode+1)+'b',activation_fn=None,normalizer_fn=None)            
            #cnv_flat = tf.reshape(cnv_layers[-1], [-1, int((H/2**(num_encode))*(W/2**(num_encode))*np.maximum(num_features*(2**(num_encode-1)),max_features))])
            #fc1 = tf.layers.dense(inputs=cnv_flat, units=1024, activation=tf.nn.relu)
            #fc = tf.layers.dense(inputs=fc1, units=1, activation=None)
            return cnvout#tf.nn.sigmoid(fc)


def discriminator_bn(tgt_image, is_training=True, is_reuse=False):
    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value

    with tf.variable_scope('depth_net',reuse = is_reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):

            cnv1_flat = tf.reshape(tgt_image, [-1, 2 * 2 * 512])
            fc1 = tf.layers.dense(inputs=cnv1_flat, units=2048, activation=tf.nn.relu)
            #cnv1_flat = tf.reshape(tgt_image, [-1, 2 * 2 * 512])
            fc2 = tf.layers.dense(inputs=fc1, units=1024, activation=tf.nn.relu)
            fc3 = tf.layers.dense(inputs=fc2, units=1, activation=None)

            return tf.nn.sigmoid(fc3)








