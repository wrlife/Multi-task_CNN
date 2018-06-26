import tensorflow as tf
import numpy as np
import time
import math
import os
import cv2
import tensorflow.contrib.slim as slim
from estimator_rui import *


def cycleGAN_training(opt,
                    m_trainer,
                    gen_loss,
                    disc_loss,
                    gen_loss_bw,
                    disc_loss_bw):


    with tf.name_scope("train_op"):
        #Optimization
        optim_G = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
        train_op = slim.learning.create_train_op(gen_loss, optim_G,variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, m_trainer.scope_name))
        train_op_bw = slim.learning.create_train_op(gen_loss_bw, optim_G,variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, m_trainer.scope_name+"_bw"))

        optim_adv = tf.train.AdamOptimizer(opt.learning_rate2, opt.beta1)
        train_adv = slim.learning.create_train_op(disc_loss, optim_adv, variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc_model'))
        train_adv_bw = slim.learning.create_train_op(disc_loss_bw, optim_adv, variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc_model_bw'))


        global_step = tf.Variable(0,
                                    name = 'global_step',
                                    trainable = False)
        incr_global_step = tf.assign(global_step,global_step+1)

    #Start training
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                        for v in tf.trainable_variables()])

    #model_vars = collect_vars(m_trainer.scope_name)
    #model_vars['global_step'] = global_step
    saver = tf.train.Saver()

    # if opt.domain_transfer_dir!="None":
    #     model_fix = collect_vars("fixnet")
    #     saver_fix = tf.train.Saver(model_fix)        

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
            # if opt.domain_transfer_dir!="None":
            #     saver_fix.restore(sess,checkpoint)

        try:
            step=0
            while True:
                start_time = time.time()
                # Essential fetches for training
                fetches = {
                    "train": train_op,
                    "train_adv": train_adv,
                    "train_bw": train_op_bw,
                    "train_adv_bw": train_adv_bw,
                    "global_step": global_step,
                    "incr_global_step": incr_global_step
                }
                # Fetch summary
                if step % opt.summary_freq == 0:
                    fetches["loss"] = gen_loss
                    fetches["summary"] = merged

                    if opt.evaluation_dir != "None":
                        fetches2 = {"summary":merged}
                        results2 = sess.run(fetches2,feed_dict={share_loss:False})
                    
                results = sess.run(fetches) 
                # Save and print log
                duration = time.time() - start_time
                gs = results["global_step"]
                if step % opt.summary_freq == 0:
                    train_writer.add_summary(results["summary"], gs)
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, results["loss"],
                                                            duration))

                    if opt.evaluation_dir != "None":
                        eval_writer.add_summary(results2["summary"], gs)
                    #import pdb;pdb.set_trace()
                    #print(results["gt3d"][0])
                    # print(results["pred3d"][0,:,1])
                if step % opt.save_latest_freq == 0:
                    save(sess, opt.checkpoint_dir, gs,saver)
                step += 1
                
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (opt.max_steps,
                                                            step))

