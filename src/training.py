import tensorflow as tf
import numpy as np
import time
import math
import os
import cv2
import tensorflow.contrib.slim as slim
from estimator_rui import *


def training(opt,m_trainer,losses,losses_eval,
             data_dict,data_dict_eval,
             output,output_eval,global_step,
             coord_pair
             ,incr_global_step
             ):

    #import pdb;pdb.set_trace()
    #Summaries
    share_loss = tf.placeholder(tf.bool, shape=())
    if opt.evaluation_dir != "None":
        
        inputloss = tf.cond(share_loss, lambda: losses, lambda: losses_eval)
        input_visual = tf.cond(share_loss, lambda: output, lambda: output_eval)
        input_data = tf.cond(share_loss, lambda: data_dict, lambda: data_dict_eval)
    else:
        inputloss = losses
        input_visual = output
        input_data = data_dict
    if not opt.pretrain_pose:
        m_trainer.construct_summary(input_data,input_visual,inputloss)
    else:
        tf.summary.scalar('losses/total_loss', losses[0])

    #import pdb;pdb.set_trace()
    with tf.name_scope("train_op"):

        #Optimization
        optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)

        #import pdb;pdb.set_trace()
        # Get the gradient pairs (Tensor, Variable)
        grads = optim.compute_gradients(losses[0])
        # Update the weights wrt to the gradient
        #train_op = optim.apply_gradients(grads)
        # Save the grads with tf.summary.histogram
        #for index, grad in enumerate(grads):
        #import pdb;pdb.set_trace
        tf.summary.histogram("{}-grad".format(grads[-1][1].name), grads[-1])
        tf.summary.histogram("{}-grad".format(grads[-2][1].name), grads[-2])

        
        train_op = slim.learning.create_train_op(losses[0], optim)
        
    #Start training
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                        for v in tf.trainable_variables()])

    model_vars = collect_vars(m_trainer.scope_name)
    model_vars['global_step'] = global_step
    saver = tf.train.Saver()

    saver_lmnet = tf.train.Saver(model_vars)

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
            saver_lmnet.restore(sess, checkpoint)
            # if opt.domain_transfer_dir!="None":
            #     saver_fix.restore(sess,checkpoint)

        try:
            step=0
            while True:
                start_time = time.time()
                # Essential fetches for training
                fetches = {
                    "train":train_op,
                    "global_step": global_step,
                    #"incr_global_step": incr_global_step
                }

                if opt.domain_transfer_dir!="None" and opt.with_dom:
                    fetches["train_adv"] = train_adv

                # Fetch summary
                if step % opt.summary_freq == 0:
                    fetches["loss"] = losses[0]
                    fetches["summary"] = merged
                    #fetches["gt3d"] = losses[5]
                    # fetches["pred3d"]= pred_lm_3D

                    if opt.evaluation_dir != "None":
                        fetches2 = {"summary":merged}
                        results2 = sess.run(fetches2,feed_dict={share_loss:False})

                    # if opt.with_pose:
                    #     fetches["gt3d"] = coord_pair
                        
                results = sess.run(fetches,feed_dict={share_loss:True}) 
                # Save and print log
                duration = time.time() - start_time
                gs = results["global_step"]
                if step % opt.summary_freq == 0:
                    train_writer.add_summary(results["summary"], gs)
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, results["loss"],
                                                            duration))
                    
                    #import pdb;pdb.set_trace()
                    #print(results["gt3d"][0])
                    #print(results["gt3d"][1])
                    # print(results["gt3d"][2])
                    # print(results["gt3d"][3])
                    if opt.evaluation_dir != "None":
                        eval_writer.add_summary(results2["summary"], gs)
                    #import pdb;pdb.set_trace()
                    # if opt.with_pose:
                    #     #import pdb;pdb.set_trace()
                    #     print(results["gt3d"][0][0,0:5])
                    #     print(results["gt3d"][1][0,0:5])
                    #     print(results["gt3d"][2])
                    # print(results["pred3d"][0,:,1])
                if step % opt.save_latest_freq == 0:
                    save(sess, opt.checkpoint_dir, gs,saver)
                step += 1
                
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (opt.max_steps,
                                                            step))

