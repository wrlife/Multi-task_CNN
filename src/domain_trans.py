import tensorflow as tf
import numpy as np
from data_loader_direct import DataLoader
from my_losses import *
from model import *
import time
import math
import os
import cv2
from estimator_rui import estimator_rui

class domain_trans:
    '''
    A wrapper function which create data, model and loss according to input type
    '''
    def __init__(self,trainer):
        self.trainer = trainer

    def construct_src(self,output_src,data_dict_src):
        mask = 1.0-data_dict_src["label"]
        output_src = output_src*data_dict_src["label"]+data_dict_src["IR"]*mask

        return output_src
        
    
    def forward_wrapper(self,scope_name,output_src,data_dict_src,input_src,output_bw,data_dict_bw,input_bw):
        '''
        A wrapper function for domain transfer.
        Generate dataloader, model, loss and its own summary
        '''

        weight_gen = 1000
        weight_disc = 1000
        lam = 10

        #_, output_dom, data_dict_dom = self.trainer.forward_wrapper(domain_transfer_dir,scope_name,is_training=True,is_reuse=True,with_loss=False,test_input=True)
        
        #test_IR = self.trainer.input_wrapper(domain_transfer_dir,scope_name,is_training=True,is_reuse=True,with_loss=False,test_input=True)

        cycle_bw = self.trainer.construct_model(output_bw,is_training=True, is_reuse=True,scope_name=scope_name,num_out_channel=input_bw.get_shape()[3].value)
        cycle_src = self.trainer.construct_model(output_src,is_training=True, is_reuse=True,scope_name=scope_name+"_bw",num_out_channel=input_src.get_shape()[3].value)

        cycle_bw_loss = tf.reduce_mean(tf.abs(input_bw - cycle_bw[0]))*lam
        cycle_src_loss = tf.reduce_mean(tf.abs(input_src - cycle_src[0]))*lam

        #=========================
        #Post process produced
        #=========================
        #output_src = self.construct_src(output_src,data_dict_src)

        #=========================
        #GAN for domain adaptation
        #=========================
        
        with tf.variable_scope("disc_model") as scope:
            disc_real = discriminator(input_bw,num_encode=4)
            disc_fake = discriminator(output_src,num_encode=4,is_training=True, is_reuse=True)
        #Consturct loss
        #import pdb;pdb.set_trace()
        gen_loss = tf.reduce_mean((disc_fake-1)**2)+cycle_bw_loss+cycle_src_loss#-tf.reduce_mean(tf.log(disc_fake))  #
        disc_loss =    tf.reduce_mean((disc_real-1)**2+(disc_fake)**2)*0.5 #-tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))#*weight_disc 


        if self.trainer.opt.with_dom:
            optim_adv = tf.train.AdamOptimizer(self.trainer.opt.learning_rate2, self.trainer.opt.beta1)
            train_adv = slim.learning.create_train_op(disc_loss, optim_adv, variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc_model'))


        with tf.variable_scope("disc_model_bw") as scope:
            disc_real_bw = discriminator(input_src,num_encode=3)
            disc_fake_bw = discriminator(output_bw,num_encode=3,is_training=True, is_reuse=True)

        gen_loss_bw = tf.reduce_mean((disc_fake_bw-1)**2)+cycle_bw_loss+cycle_src_loss#-tf.reduce_mean(tf.log(disc_fake))  #
        disc_loss_bw =    tf.reduce_mean((disc_real_bw-1)**2+(disc_fake_bw)**2)*0.5

        #Construct discriminator
        #import pdb;pdb.set_trace()
        if self.trainer.opt.with_dom:
            #optim_adv = tf.train.AdamOptimizer(self.trainer.opt.learning_rate2, self.trainer.opt.beta1)
            train_adv_bw = slim.learning.create_train_op(disc_loss_bw, optim_adv, variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc_model_bw'))

        #Construct summaries
        tf.summary.image('domain/image_tgt' , \
                            input_bw)
        tf.summary.image('domain_lm_img' , \
                            output_src)
        tf.summary.scalar('losses/gen_loss', gen_loss)
        tf.summary.scalar('losses/disc_loss', disc_loss)

        tf.summary.image('domain/image_src' , \
                            input_src)
        tf.summary.image('domain_lm_img_bw' , \
                            output_bw)
        tf.summary.scalar('losses/gen_loss_bw', gen_loss_bw)
        tf.summary.scalar('losses/disc_loss_bw', disc_loss_bw)

        tf.summary.scalar('losses/cycle_bw_loss', cycle_bw_loss)
        tf.summary.scalar('losses/cycle_src_loss', cycle_src_loss)
        
        return gen_loss,disc_loss,gen_loss_bw,disc_loss_bw#train_adv,train_adv_bw