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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "/home/z003xr2y/data/data/tfrecords_hr_filldepth/", "Dataset directory")
flags.DEFINE_string("evaluation_dir", "None", "Dataset directory")
flags.DEFINE_string("domain_transfer_dir", "None", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints_IR_depth_color_landmark_hm_lastdecode_sm/", "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("learning_rate2", 0.00002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momenm term of adam")
flags.DEFINE_integer("batch_size", 2, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 480, "Image height")
flags.DEFINE_integer("img_width", 640, "Image width")
flags.DEFINE_integer("max_steps", 120, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 1000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_string("inputs", "all", "all IR_depth depth_color IR_color IR color depth")
flags.DEFINE_string("model", "lastdecode", "lastdecode sinlge")
flags.DEFINE_boolean("data_aug", False, "Data augment")
flags.DEFINE_boolean("with_seg", False, "with seg")
flags.DEFINE_boolean("with_pose", False, "with pose estimation")
flags.DEFINE_boolean("training", True, "if False, start prediction")

opt = flags.FLAGS

opt.checkpoint_dir="./checkpoints/"+opt.inputs+"_"+opt.model
if opt.data_aug:
    opt.checkpoint_dir = opt.checkpoint_dir+"_dataaug"
if opt.with_pose:
    opt.checkpoint_dir = opt.checkpoint_dir+"_pose"
if opt.domain_transfer_dir!="None":
    opt.checkpoint_dir = opt.checkpoint_dir+"_dom"

opt.checkpoint_dir = opt.checkpoint_dir+"/lr1_"+str(opt.learning_rate)+"_lr2_"+str(opt.learning_rate2)

if not os.path.exists(opt.checkpoint_dir):
    os.makedirs(opt.checkpoint_dir)

write_params(opt)
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
losses, output, data_dict = m_trainer.forward_wrapper(
                                           opt.dataset_dir,
                                           scope_name,
                                           opt.max_steps,
                                           with_dataaug=True)


#==========================
#Forward path for pose 
# estimation
#==========================
if opt.with_pose:
    m_pose_est = pose_estimate(m_trainer)
    pose_loss = m_pose_est.forward_wrapper(
                                            output,
                                            data_dict)
    losses = list(losses)
    losses[0] = losses[0]+pose_loss


#==========================
#Forward path for evaluation
#During testing, just set None
#==========================
if opt.evaluation_dir != "None":
    losses_eval, output_eval, data_dict_eval = m_trainer.forward_wrapper(
                                                                        opt.evaluation_dir,
                                                                        scope_name,
                                                                        is_training=False,
                                                                        is_reuse=True)
    losses_eval = list(losses_eval)


#==========================
#Forward path for domain transfer
#Set to None during testing
#==========================
if opt.domain_transfer_dir != "None":
    m_domain_tans = domain_trans(m_trainer)
    gen_loss, disc_loss, train_adv = m_domain_tans.forward_wrapper(
                                            opt.domain_transfer_dir,
                                            scope_name,
                                            output)
    losses = list(losses)
    losses[0] = losses[0]+gen_loss
else:
    train_adv=0


#==========================
#Start training
#==========================
pred_landmark = m_trainer.parse_output_landmark(output)
pred_landmark_eval = m_trainer.parse_output_landmark(output_eval)

training(
    opt,
    m_trainer,
    losses,
    losses_eval,data_dict, 
    data_dict_eval,
    pred_landmark, 
    pred_landmark_eval,
    train_adv)



#==========================
#Start Prediction
#==========================
