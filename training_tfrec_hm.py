import tensorflow as tf
import numpy as np
from data_loader_direct import DataLoader
from my_losses import *
from model import *
import time
import math
import os
from smoother import Smoother

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints_l2_IR_color_depth_landmark_hm_varGauss_v3/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momenm term of adam")
flags.DEFINE_float("smooth_weight", 0.5, "Weight for smoothness")
flags.DEFINE_float("explain_reg_weight", 0.0, "Weight for explanability regularization")
flags.DEFINE_integer("batch_size", 2, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 224, "Image height")
flags.DEFINE_integer("img_width", 224, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("max_steps", 60, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 1, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 1000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_integer("num_scales", 4, "Num scales")
flags.DEFINE_integer("change_gauss", 2000, "Change gauss scale after num steps")

opt = flags.FLAGS

# Basic Constants
FILTER_SIZE = 25
SIGMA = 0.3*((FILTER_SIZE-1)*0.5 - 1) + 0.8

CUDA_VISIBLE_DEVICES=1

def save(sess, checkpoint_dir, step, saver):
    model_name = 'model'
    print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
    if step == 'latest':
        saver.save(sess, 
                        os.path.join(checkpoint_dir, model_name + '.latest'))
    else:
        saver.save(sess, 
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

def gauss_smooth(mask):
    smoother = Smoother({'data':mask}, FILTER_SIZE, SIGMA)
    return smoother.get_output()



#Initialize data loader
imageloader = DataLoader('/home/z003xr2y/data/tfrecords/',  #'D:\\Exp_data\\data\\2017_0216_DetectorDetection\\tfrecords'
                            5,
                            224, 
                            224,
                            'train')

# Load training data
data_dict = imageloader.inputs(opt.batch_size,opt.max_steps)  # batch_size, num_epochs


#Construct model
#Concatenate color and depth for model input
input_ts = tf.concat([data_dict['IR'],data_dict['image'],data_dict['depth']],axis=3)
pred, pred_landmark, _ = disp_net(tf.cast(input_ts,tf.float32))


#Use larger Gaussian mask in the first few thousand iterations of training
use_large_gauss = tf.placeholder(tf.float32,name="condition")
smoothed = gauss_smooth(data_dict['points2D'])
data_dict['points2D'] = tf.cond(use_large_gauss>0,lambda:smoothed, lambda:data_dict['points2D'])


#Compute loss
total_loss,depth_loss,landmark_loss,vis_loss = compute_loss(pred,pred_landmark,data_dict,opt)
#val_loss = compute_loss(pred_val,label_val_batch,opt)


with tf.name_scope("train_op"):

    #Optimization
    train_vars = [var for var in tf.trainable_variables()]
    optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
    train_op = slim.learning.create_train_op(total_loss, optim)
    global_step = tf.Variable(0,
                                name = 'global_step',
                                trainable = False)
    incr_global_step = tf.assign(global_step,global_step+1)

    #Summary
    tf.summary.scalar('losses/total_loss', total_loss)
    tf.summary.scalar('losses/depth_loss', depth_loss)
    tf.summary.scalar('losses/landmark_loss', landmark_loss)
    tf.summary.scalar('losses/vis_loss', vis_loss)
    
    
    
    tf.summary.image('train_image' , \
                        data_dict['image'])
    tf.summary.image('gt_label' , \
                        data_dict['label'])
    tf.summary.image('pred_label' , \
                        pred[0])

    gt_landmark = tf.expand_dims(tf.reduce_sum(data_dict['points2D'],3),axis=3)
    pred_landmark = tf.expand_dims(tf.reduce_sum(pred_landmark[0],3),axis=3)


    # sp1,sp2,sp3=tf.split(pred_landmark,3,2)
    # offset1 = tf.ones_like(sp1)/opt.img_width
    # offset2 = tf.ones_like(sp1)/opt.img_height
    # pred_bbox = tf.concat([sp2,sp1,sp2+offset2,sp1+offset1],axis=2)


    # gt_landmark = data_dict['points2D']
    # sp1_gt, sp2_gt = tf.split(gt_landmark, 2, 2)
    # gt_bbox = tf.concat([sp2_gt,sp1_gt,sp2_gt+offset2,sp1_gt+offset1],axis=2)


    # gt_lm_img = tf.image.draw_bounding_boxes(data_dict['image'],gt_bbox)
    # pred_lm_img = tf.image.draw_bounding_boxes(data_dict['image'],pred_bbox)
    tf.summary.image('gt_lm_img' , \
                        gt_landmark)
    tf.summary.image('pred_lm_img' , \
                        pred_landmark)


    # Test summary
    # tf.summary.scalar('losses/total_loss', val_loss)
    # tf.summary.image('val_image' , \
    #                     image_test_batch)    
    # tf.summary.image('pred_val_label', \
    #                     pred_test[0])


#Start training
with tf.name_scope("parameter_count"):
    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                    for v in tf.trainable_variables()])

saver = tf.train.Saver([var for var in tf.model_variables()] + \
                            [global_step],
                            max_to_keep=10)

sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                            save_summaries_secs=0, 
                            saver=None)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Session start
with sv.managed_session(config=config) as sess:

    # Load parameters
    print('Trainable variables: ')
    for var in tf.trainable_variables():
        print(var.name)
    print("parameter_count =", sess.run(parameter_count))
    if opt.continue_train:
        if opt.init_checkpoint_file is None:
            checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
        else:
            checkpoint = opt.init_checkpoint_file
        print("Resume training from previous checkpoint: %s" % checkpoint)
        saver.restore(sess, checkpoint)

    try:
        step=0
        while True:
            start_time = time.time()

            # Essential fetches for training
            fetches = {
                "train":train_op,
                "global_step": global_step,
                "incr_global_step": incr_global_step
            }

            # Fetch summary
            if step % opt.summary_freq == 0:
                fetches["loss"] = total_loss
                fetches["summary"] = sv.summary_op

            #===============
            #Run fetch
            #===============
            use_gauss = 1.0
            if(step>opt.change_gauss):
                use_gauss=0
                
            results = sess.run(fetches,feed_dict={use_large_gauss:use_gauss})
            # Save and print log
            duration = time.time() - start_time
            gs = results["global_step"]
            if step % opt.summary_freq == 0:
                sv.summary_writer.add_summary(results["summary"], gs)
                print('Step %d: loss = %.2f (%.3f sec)' % (step, results["loss"],
                                                        duration))
            if step % opt.save_latest_freq == 0:
                save(sess, opt.checkpoint_dir, gs,saver)
            # if step % steps_per_epoch == 0:
            #     save(sess, opt.checkpoint_dir, gs,saver)

            step += 1
            
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (opt.max_steps,
                                                        step))

