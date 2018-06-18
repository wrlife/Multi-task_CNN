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
from domain_trans import domain_trans


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "/home/z003xr2y/data/data/tfrecords_hr_filldepth/", "Dataset directory")
flags.DEFINE_string("evaluation_dir", "None", "Dataset directory")
flags.DEFINE_string("domain_transfer_dir", "None", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints_IR_depth_color_landmark_hm_lastdecode_sm/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("learning_rate2", 0.00002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momenm term of adam")
flags.DEFINE_float("smooth_weight", 0.5, "Weight for smoothness")
flags.DEFINE_float("explain_reg_weight", 0.0, "Weight for explanability regularization")
flags.DEFINE_integer("batch_size", 2, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 480, "Image height")
flags.DEFINE_integer("img_width", 640, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("max_steps", 120, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 1000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_integer("num_scales", 4, "Num scales")
flags.DEFINE_integer("change_gauss", 2000, "Change gauss scale after num steps")

flags.DEFINE_string("inputs", "all", "all IR_depth depth_color IR_color IR color depth")
flags.DEFINE_string("model", "lastdecode", "lastdecode sinlge")
flags.DEFINE_boolean("data_aug", False, "Data augment")
flags.DEFINE_boolean("with_seg", False, "with seg")

opt = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]="1"

scope_name="DBNet"
#Define a trainer instance
m_trainer = estimator_rui(opt)

#Forward path for training data
losses, pred_landmark, output, data_dict = m_trainer.forward_wrapper(opt.dataset_dir,scope_name,opt.max_steps,with_dataaug=True)

#Forward path for pose estimation


#Forward path for evaluation
if opt.evaluation_dir != "None":
    losses_eval, pred_landmark_eval, output_eval, data_dict_eval = m_trainer.forward_wrapper(opt.evaluation_dir,scope_name,is_training=False,is_reuse=True)
    losses_eval = list(losses_eval)

#Forward path for domain transfer
if opt.domain_transfer_dir != "None":
    m_domain_tans = domain_trans(m_trainer)
    gen_loss,disc_loss,train_adv = m_domain_tans.forward_wrapper(opt.domain_transfer_dir,scope_name,output)

    losses = list(losses)
    losses[0] = losses[0]+gen_loss




#Summaries
share_loss = tf.placeholder(tf.bool, shape=())
inputloss = tf.cond(share_loss, lambda: losses, lambda: losses_eval)
input_landmark = tf.cond(share_loss, lambda: pred_landmark, lambda: pred_landmark_eval)
input_data = tf.cond(share_loss, lambda: data_dict, lambda: data_dict_eval)
m_trainer.construct_summary(inputloss,input_data,input_landmark)


with tf.name_scope("train_op"):

    #Optimization
    
    #train_mapping = slim.learning.create_train_op(gen_loss, optim_mapping,variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'tgt_model'))
    #train_vars = [var for var in tf.trainable_variables()]

    optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
    train_op = slim.learning.create_train_op(losses[0], optim,variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name))

    global_step = tf.Variable(0,
                                name = 'global_step',
                                trainable = False)
    incr_global_step = tf.assign(global_step,global_step+1)


#Start training
with tf.name_scope("parameter_count"):
    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                    for v in tf.trainable_variables()])

model_vars = collect_vars(scope_name)
model_vars['global_step'] = global_step
saver = tf.train.Saver(model_vars, #+ \
                            #[global_step],
                            max_to_keep=10)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Session start
with tf.Session(config=config) as sess:

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(opt.checkpoint_dir + '/logs/train',
                                            sess.graph)
    eval_writer = tf.summary.FileWriter(opt.checkpoint_dir + '/logs/eval')

    merged = tf.summary.merge_all()

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
        m_f_size = opt.img_height/2.0#159.0
        while True:
            start_time = time.time()
            #import pdb;pdb.set_trace()
            # Essential fetches for training
            fetches = {
                "train":train_op,
                "global_step": global_step,
                "incr_global_step": incr_global_step
            }

            # Fetch summary
            if step % opt.summary_freq == 0:
                fetches["loss"] = losses[0]
                fetches["summary"] = merged
                # fetches["gt3d"] = gt_lm_3D
                # fetches["pred3d"]= pred_lm_3D


                if opt.evaluation_dir != "None":
                    fetches2 = {"summary":merged}
                    results2 = sess.run(fetches2,feed_dict={share_loss:False})
                

            
                

            #===============
            #Run fetch
            #===============
            #use_gauss = 1.0
            #if(step>opt.change_gauss):
            #    use_gauss=0
            #if m_f_size>9.0:
            #  m_f_size = m_f_size-m_f_size/8000.0
            #else:
            #  m_f_size=9.0
            m_f_size = 9.0
                
            results = sess.run(fetches,feed_dict={share_loss:True}) #,feed_dict={kernel_size:m_f_size,random_landmark:np.random.randint(5)}
            # Save and print log
            duration = time.time() - start_time
            gs = results["global_step"]
            if step % opt.summary_freq == 0:
                train_writer.add_summary(results["summary"], gs)
                print('Step %d: loss = %.2f (%.3f sec), Filter_size: %f' % (step, results["loss"],
                                                        duration, m_f_size))

                if opt.evaluation_dir != "None":
                    eval_writer.add_summary(results2["summary"], gs)
                #import pdb;pdb.set_trace()
                # print(results["gt3d"][0,:,1])
                # print(results["pred3d"][0,:,1])
            if step % opt.save_latest_freq == 0:
                save(sess, opt.checkpoint_dir, gs,saver)
            step += 1
            
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (opt.max_steps,
                                                        step))

