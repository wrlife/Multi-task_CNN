import tensorflow as tf
import numpy as np
from data_loader_direct import DataLoader
from my_losses import *
from model import *
import time
import math
import os
from smoother import Smoother
import cv2
from estimator_rui import *
from domain_trans import *
from pose_estimate import *
from training import *
from evaluate import *
from prediction import *
from cyclegan_training import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "/home/z003xr2y/data/data/tfrecords_hr_filldepth/", "Dataset directory")
flags.DEFINE_string("evaluation_dir", "None", "Dataset directory")
flags.DEFINE_string("domain_transfer_dir", "None", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints_IR_depth_color_landmark_hm_lastdecode_sm/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("learning_rate2", 0.001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momenm term of adam")
flags.DEFINE_integer("num_scales", 4, "number of scales")
flags.DEFINE_integer("num_encoders", 5, "number of encoders")
flags.DEFINE_integer("num_features", 32, "number of starting features")
flags.DEFINE_integer("batch_size", 5, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 480, "Image height")
flags.DEFINE_integer("img_width", 640, "Image width")
flags.DEFINE_integer("max_steps", 120, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 1000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_string("inputs", "all", "all IR_depth depth_color IR_color IR color depth")
flags.DEFINE_string("model", "lastdecode", "lastdecode sinlge")
flags.DEFINE_boolean("downsample", False, "Data augment")
flags.DEFINE_boolean("data_aug", False, "Data augment")
flags.DEFINE_boolean("with_seg", False, "with seg")
flags.DEFINE_boolean("with_pose", False, "with pose estimation")
flags.DEFINE_boolean("with_noise", False, "if False, start prediction")
flags.DEFINE_boolean("with_geo", False, "with geometry estimation")
flags.DEFINE_boolean("with_dom", False, "with domain transform")
flags.DEFINE_boolean("with_vis", False, "with visibility loss")
flags.DEFINE_boolean("training", True, "if False, start prediction")
flags.DEFINE_boolean("evaluation", False, "if False, start prediction")
flags.DEFINE_boolean("prediction", False, "if False, start prediction")
flags.DEFINE_boolean("cycleGAN", False, "if False, start cyclegan")


opt = flags.FLAGS

opt.checkpoint_dir="./checkpoints/"+opt.inputs+"_"+opt.model
if opt.with_seg:
    opt.checkpoint_dir = opt.checkpoint_dir+"_seg"
if opt.data_aug:
    opt.checkpoint_dir = opt.checkpoint_dir+"_dataaug"
if opt.with_pose:
    opt.checkpoint_dir = opt.checkpoint_dir+"_pose"
if opt.with_noise:
    opt.checkpoint_dir = opt.checkpoint_dir+"_noise"
if opt.with_vis:
    opt.checkpoint_dir = opt.checkpoint_dir+"_vis"
if opt.with_geo:
    opt.checkpoint_dir = opt.checkpoint_dir+"_geo"
if opt.domain_transfer_dir!="None" and opt.with_dom:
    opt.checkpoint_dir = opt.checkpoint_dir+"_dom"

evaluate_name = opt.checkpoint_dir[14:]

opt.checkpoint_dir = opt.checkpoint_dir+"/lr1_"+str(opt.learning_rate)+"_lr2_"+str(opt.learning_rate2)+"_numEncode"+str(opt.num_encoders)+"_numFeatures"+str(opt.num_features)
#import pdb;pdb.set_trace()
if not os.path.exists(opt.checkpoint_dir):
    os.makedirs(opt.checkpoint_dir)



write_params(opt)
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#==========================
#Define a estimator instance
#Estimator wraps dataloader, 
#model and tensorboard summary
#==========================
scope_name="DBNet"
m_trainer = estimator_rui(opt,scope_name)


#==========================
#Forward path for training
# /testing data
#==========================
global_step = tf.Variable(0,
                            name = 'global_step',
                            trainable = False)
incr_global_step = tf.assign(global_step,global_step+1)

pose_weight = (tf.cast(global_step,tf.float32)-5000.0)/50000.0
if opt.training:
    losses, output, data_dict,_ = m_trainer.forward_wrapper(
                                            opt.dataset_dir,
                                            scope_name,
                                            opt.max_steps,
                                            with_dataaug=opt.data_aug)
    losses = list(losses)

    #==========================
    #Forward path for pose 
    # estimation
    #==========================
    if opt.with_pose:

        m_pose_est = pose_estimate(m_trainer)
        
        pose_loss,coord_pair = m_pose_est.forward_wrapper(
                                                output,
                                                data_dict,
                                                pose_weight
                                                )

            #return pose_loss#tf.cond(est, lambda:pose_loss,lambda:pose_loss)

        pose_loss = tf.cond(tf.greater(global_step,tf.ones([],tf.int32)*5000), lambda:pose_loss,lambda:0.0)#est_pose(tf.greater(global_step,tf.ones([],tf.int32)*5000),m_trainer,output,data_dict)
        losses[0] = losses[0]+pose_loss
        losses[4] = pose_loss



#==========================
#Forward path for evaluation
#During testing, just set None
#==========================
if opt.evaluation_dir != "None":
    losses_eval, output_eval, data_dict_eval,_ = m_trainer.forward_wrapper(
                                                                        opt.evaluation_dir,
                                                                        scope_name,
                                                                        opt.max_steps,
                                                                        is_training=opt.training,
                                                                        is_reuse=opt.training)
    losses_eval = list(losses_eval)


    #==========================
    #Forward path for pose 
    # estimation
    #==========================

    if opt.with_pose or opt.evaluation:
        m_pose_est_eval = pose_estimate(m_trainer)
        pose_loss_eval,_ = m_pose_est_eval.forward_wrapper(
                                                output_eval,
                                                data_dict_eval,
                                                pose_weight,
                                                is_training=opt.training
                                                )
        pose_loss_eval = tf.cond(tf.greater(global_step,tf.ones([],tf.int32)*5000), lambda:pose_loss_eval,lambda:0.0)
        losses_eval[0] = losses_eval[0]+pose_loss_eval
        losses_eval[4] = pose_loss_eval

else:
    losses_eval=0
    data_dict_eval=0
    output_eval=0


#==========================
#Forward path for domain transfer
#Set to None during testing
#==========================
#import pdb;pdb.set_trace()
if opt.domain_transfer_dir != "None" and opt.with_dom:
    
    #Forward mapping
    _, output_fix, data_dict_fix,input_fix = m_trainer.forward_wrapper(
                                                            opt.dataset_dir,
                                                            scope_name,
                                                            opt.max_steps,
                                                            is_training=True,
                                                            is_reuse=False,
                                                            with_loss=False,
                                                            network_type="G")


    #Backward mapping
    _, output_bw, data_dict_bw,input_bw = m_trainer.forward_wrapper(
                                                            opt.domain_transfer_dir,
                                                            scope_name+"_bw",
                                                            opt.max_steps,
                                                            is_training=True,
                                                            is_reuse=False,
                                                            with_loss=False,
                                                            test_input=True,
                                                            network_type="G")

    m_domain_tans = domain_trans(m_trainer)
    gen_loss,disc_loss,gen_loss_bw,disc_loss_bw = m_domain_tans.forward_wrapper(
                                            scope_name,
                                            output_fix[0],
                                            data_dict_fix,
                                            input_fix,
                                            output_bw[0],
                                            data_dict_bw,
                                            input_bw)

    data_dict = data_dict_fix
    output = output_fix
    #losses[0] = gen_loss#losses[0]+gen_loss
else:
    train_adv=0

    
#==========================
#Forward path
#==========================

if opt.prediction:
    
    _, output_dom, data_dict_dom = m_trainer.forward_wrapper(
                                                            opt.domain_transfer_dir,
                                                            scope_name,
                                                            opt.max_steps,
                                                            is_training=False,
                                                            is_reuse=False,
                                                            with_loss=False,
                                                            test_input=True)


#==========================
#Start training
#==========================
if opt.training:
    training(
        opt,
        m_trainer,
        losses,
        losses_eval,
        data_dict, 
        data_dict_eval,
        output, 
        output_eval,
        global_step,
        incr_global_step
        )

#==========================
#Start evaluation
#==========================
elif opt.evaluation:
    evaluate(
        opt,
        evaluate_name,
        m_trainer,
        losses_eval,
        data_dict_eval,
        output_eval,
        global_step,
        incr_global_step)


#==========================
#Start Prediction
#==========================
elif opt.prediction:
    prediction(
        opt,
        m_trainer,
        data_dict_dom,
        output_dom        
    )


#==========================
#Start training cycle GAN
#==========================
elif opt.cycleGAN:
    cycleGAN_training(
        opt,
        m_trainer,
        gen_loss,
        disc_loss,
        gen_loss_bw,
        disc_loss_bw
        )