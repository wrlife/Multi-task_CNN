import tensorflow as tf
import numpy as np
from data_loader_direct import DataLoader
from my_losses import *
from model import *
import time
import math
import os
from smoother import Smoother
from collections import OrderedDict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "/home/z003xr2y/data/tfrecords/", "Dataset directory")
flags.DEFINE_string("evaluation_dir", "None", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints_IR_landmark_single_domain_trans_test2/", "Directory name to save the checkpoints")
flags.DEFINE_string("checkpoint_dir_src", "./checkpoints_IR_landmark_single/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate1", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("learning_rate2", 0.000002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momenm term of adam")
flags.DEFINE_float("smooth_weight", 0.5, "Weight for smoothness")
flags.DEFINE_float("explain_reg_weight", 0.0, "Weight for explanability regularization")
flags.DEFINE_integer("batch_size", 2, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 224, "Image height")
flags.DEFINE_integer("img_width", 224, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("max_steps", 1000, "Maximum number of training iterations")
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

# Basic Constants
#FILTER_SIZE = 9


os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

def gauss_smooth(mask,FILTER_SIZE):
    SIGMA = 0.3*((FILTER_SIZE-1)*0.5 - 1) + 0.8#0.3*(FILTER_SIZE-1) + 0.8
    smoother = Smoother({'data':mask}, FILTER_SIZE, SIGMA)
    new_mask = smoother.get_output()

    return new_mask
    
    
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
    
def rank(tensor):

    # return the rank of a Tensor
    return len(tensor.get_shape())

def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])


def collect_vars(scope, start=None, end=None, prepend_scope=None):
    #import pdb;pdb.set_trace()
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    #import pdb;pdb.set_trace()
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
    return var_dict



#------------------------------------------------
#Training from data loading to loss computation
#------------------------------------------------
#Initialize data loader
imageloader_src = DataLoader(opt.dataset_dir,  #'D:\\Exp_data\\data\\2017_0216_DetectorDetection\\tfrecords'
                            5,
                            opt.img_height, 
                            opt.img_width,
                            'train')
# Load training data
data_dict_src = imageloader_src.inputs(opt.batch_size,opt.max_steps,None)  # batch_size, num_epochs

imageloader_tgt = DataLoader("/home/z003xr2y/data/tfrecords_test/",  
                            5,
                            opt.img_height, 
                            opt.img_width,
                            'train')
# Load training data
data_dict_tgt = imageloader_tgt.inputs_test(opt.batch_size,None)  # batch_size, num_epochs

#==========================
#Construct input
#==========================
#Concatenate color and depth for model input

input_src = data_dict_src['IR']#tf.concat([data_dict_src['IR'],data_dict_src['depth']],axis=3)
input_tgt = data_dict_tgt['IR']#tf.concat([data_dict_tgt['IR'],data_dict_tgt['depth']],axis=3)


with tf.variable_scope("src_model") as scope:
    output_src = disp_net_single(tf.cast(input_src,tf.float32),is_training=False)

with tf.variable_scope("tgt_model") as scope:
    output_tgt = disp_net_single(tf.cast(input_tgt,tf.float32))

#=========================
#GAN for domain adaptation
#=========================

with tf.variable_scope("disc_model") as scope:
    disc_real = discriminator_bn(output_src[2])
    disc_fake = discriminator_bn(output_tgt[2],is_training=True, is_reuse=True)

#import pdb;pdb.set_trace()
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))


#=======================
#Construct output
#=======================
pred_landmark = output_tgt[1]
  
  

with tf.name_scope("train_op"):

    
    #Optimization
    src_vars = collect_vars('src_model')#src_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'src_model')
    tgt_vars = collect_vars('tgt_model')#tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'tgt_model')
    disc_vars = collect_vars('disc_model')#tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'disc_model')

    optim_mapping = tf.train.AdamOptimizer(opt.learning_rate1, opt.beta1)
    optim_adv = tf.train.AdamOptimizer(opt.learning_rate2, opt.beta1)


    train_mapping = slim.learning.create_train_op(gen_loss, optim_mapping,variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'tgt_model'))
    train_adv = slim.learning.create_train_op(disc_loss, optim_adv, variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc_model'))
    
    global_step = tf.Variable(0,
                                name = 'global_step',
                                trainable = False)
    incr_global_step = tf.assign(global_step,global_step+1)

    #==========================
    #Summary
    #==========================
    tf.summary.scalar('losses/gen_loss', gen_loss)
    tf.summary.scalar('losses/disc_loss', disc_loss)
    tf.summary.image('tgt/image' , \
                        data_dict_tgt['image'])      
    random_landmark = tf.placeholder(tf.int32)
    pred_landmark = tf.expand_dims(pred_landmark[:,:,:,random_landmark],axis=3)
    tf.summary.image('tgt/pred_landmark' , \
                        pred_landmark)


    tf.summary.image('src/image' , \
                        data_dict_src['image'])      

    gt_landmark = tf.expand_dims(data_dict_src['points2D'][:,:,:,random_landmark],axis=3)
    pred_landmark = tf.expand_dims(output_src[1][:,:,:,random_landmark],axis=3)
    tf.summary.image('src/gt_landmark' , \
                        gt_landmark)
    tf.summary.image('src/pred_landmark' , \
                        pred_landmark)



saver_src = tf.train.Saver(var_list=src_vars)

saver_tgt = tf.train.Saver(var_list=tgt_vars)

#Start training
sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                            save_summaries_secs=0, 
                            saver=None)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# Session start
with sv.managed_session(config=config) as sess:

    
    # Load parameters
    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir_src)
    saver_src.restore(sess, checkpoint)
    saver_tgt.restore(sess, checkpoint)


    try:
        step=0
        while True:
            start_time = time.time()

            # Essential fetches for training
            fetches = {
                "train_mapping":train_mapping,
                "train_adv":train_adv,
                "global_step": global_step,
                "incr_global_step": incr_global_step
            }

            # Fetch summary
            if step % opt.summary_freq == 0:
                fetches["gen_loss"] = gen_loss
                fetches["disc_loss"] = disc_loss
                fetches["summary"] = sv.summary_op

            #===============
            #Run fetch
            #===============
            results = sess.run(fetches,feed_dict={random_landmark:np.random.randint(5)})
            
            # Save and print log
            duration = time.time() - start_time
            gs = results["global_step"]
            if step % opt.summary_freq == 0:
                sv.summary_writer.add_summary(results["summary"], gs)
                print('Step %d: Generator Loss: %f, Discriminator Loss: %f' % (step, results["gen_loss"],
                                                        results["disc_loss"]))
                                        
            if step % opt.save_latest_freq == 0:
                save(sess, opt.checkpoint_dir, gs,saver_tgt)
            step += 1
            
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (opt.max_steps,
                                                        step))

