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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "/home/z003xr2y/data/data/tfrecords_hr_filldepth/", "Dataset directory")
flags.DEFINE_string("evaluation_dir", "None", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints_IR_depth_color_landmark_hm_lastdecode_sm/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
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

#==========================
#Construct input
#==========================

def construct_input(opt, data_dict):

    #Concatenate color and depth for model input
    if opt.inputs == "all":
        input_ts = tf.concat([data_dict['IR'],data_dict['depth'],data_dict['image']],axis=3) #data_dict['depth'],
    elif opt.inputs == "IR_depth":
        input_ts = tf.concat([data_dict['IR'],data_dict['depth']],axis=3)
    elif opt.inputs == "depth_color":
        input_ts = tf.concat([data_dict['depth'],data_dict['image']],axis=3)
    elif opt.inputs =="IR_color":
        input_ts = tf.concat([data_dict['IR'],data_dict['image']],axis=3)
    elif opt.inputs =="IR":
        input_ts = data_dict['IR']
    elif opt.inputs =="color":
        input_ts = data_dict['image']
    elif opt.inputs =="depth":
        input_ts = data_dict['depth']
    
    return input_ts



#=======================
#Construct model
#=======================
def construct_model(opt, data_dict, input_ts,is_training=True, is_reuse=False):
    if opt.model=="lastdecode":
        output = disp_net(tf.cast(input_ts,tf.float32),is_training,is_reuse)
    elif opt.model=="single":
        output = disp_net_single(tf.cast(input_ts,tf.float32),is_training,is_reuse)
    elif opt.model=="pose":
        output = disp_net_single_pose(tf.cast(input_ts,tf.float32),is_training,is_reuse)
    elif opt.model=="multiscale":
        output = disp_net_single_multiscale(tf.cast(input_ts,tf.float32),is_training,is_reuse)
    elif opt.model=="hourglass":
        initial_output = disp_net_initial(tf.cast(input_ts,tf.float32),is_training,is_reuse)
        input_ts = tf.concat([input_ts,initial_output[1]],axis=3)
        refine_output = disp_net_refine(tf.cast(input_ts,tf.float32),is_training,is_reuse)
        output = [initial_output,refine_output]
        data_dict["landmark_init"] = tf.concat([tf.expand_dims(data_dict["points2D"][:,:,:,0],axis=3),
                                                tf.expand_dims(data_dict["points2D"][:,:,:,4],axis=3),
                                                tf.expand_dims(data_dict["points2D"][:,:,:,10],axis=3),
                                                tf.expand_dims(data_dict["points2D"][:,:,:,14],axis=3)],axis=3)
    elif opt.model=="with_tp":
        #import pdb;pdb.set_trace()
        #template_mask = np.repeat(np.expand_dims(cv2.imread('template_mask.png').astype(np.float32),axis=0),opt.batch_size,0)/255.0
        template_image = np.repeat(np.expand_dims(cv2.imread('template_image.png').astype(np.float32),axis=0),opt.batch_size,0)/255.0
        #tp_ms = tf.constant(template_mask)
        tp_im = tf.constant(template_image)
        input_ts = tf.concat([input_ts,tp_im],axis=3)
        output = disp_net_single(tf.cast(input_ts,tf.float32))


    #=======================
    #Construct output
    #=======================
    #pred = output[0]
    if opt.model == "multiscale":
        pred_landmark = output[1][0]
    elif opt.model=="hourglass":
        pred_landmark = output[1][1]
    else:
        pred_landmark = output[1]
    
    return output,pred_landmark


#=============================
#Construct summaries
#=============================
def construct_summary(losses,data_dict,pred_landmark):

    #Summary
    total_loss = tf.summary.scalar('losses/total_loss', losses[0])
    seg_loss = tf.summary.scalar('losses/seg_loss', losses[1])
    landmark_loss = tf.summary.scalar('losses/landmark_loss', losses[2])
    transformation_loss = tf.summary.scalar('losses/transformation_loss', losses[3])
    vis_loss = tf.summary.scalar('losses/vis_loss', losses[4])
    image = tf.summary.image('image' , \
                        data_dict['image'])

    if opt.with_seg:
        tf.summary.image('gt_label' , \
                            data_dict['label'])
        tf.summary.image('pred_label' , \
                            pred[0])
                        
    # random_landmark = tf.placeholder(tf.int32)
    gt_landmark = tf.expand_dims(tf.reduce_sum(data_dict['points2D'],3),axis=3)#tf.expand_dims(data_dict['points2D'][:,:,:,random_landmark],axis=3)#
    pred_landmark = tf.expand_dims(tf.reduce_sum(pred_landmark,3),axis=3)#tf.expand_dims(pred_landmark[:,:,:,random_landmark],axis=3)#
    tf.summary.image('gt_lm_img' , \
                        gt_landmark)
    tf.summary.image('pred_lm_img' , \
                        pred_landmark)
    return tf.summary.merge([total_loss,seg_loss,landmark_loss,transformation_loss,vis_loss,image]) #

    
def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
        xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
        k = FLAGS.dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}


#------------------------------------------------
#Training from data loading to loss computation
#------------------------------------------------

#Initialize data loader
imageloader = DataLoader(opt.dataset_dir,  #'D:\\Exp_data\\data\\2017_0216_DetectorDetection\\tfrecords'
                            5,
                            opt.img_height, 
                            opt.img_width,
                            'train')
# Load training data
data_dict = imageloader.inputs(opt.batch_size,opt.max_steps,opt.data_aug)  # batch_size, num_epochs

#Construct input accordingly
input_ts = construct_input(opt, data_dict)

#Select model accordingly
output,pred_landmark = construct_model(opt, data_dict, input_ts)

#Use larger Gaussian mask in the first few thousand iterations of training
#use_large_gauss = tf.placeholder(tf.float32,name="condition")
#kernel_size = tf.placeholder(tf.float32,name="k_size")
#new_mask = gauss_smooth(data_dict['points2D'],kernel_size)
#data_dict['points2D'] = new_mask

#Compute loss
losses = compute_loss(output,data_dict,opt)
#total_loss,depth_loss,landmark_loss,vis_loss,transformation_loss







#------------------------------------------------
#Evaluation
#------------------------------------------------
if opt.evaluation_dir != "None":
    #Initialize evaluation
    imageloader_val = DataLoader(opt.evaluation_dir,  #'D:\\Exp_data\\data\\2017_0216_DetectorDetection\\tfrecords'
                                1,
                                opt.img_height, 
                                opt.img_width,
                                'val')
    # Load training data
    data_dict_val = imageloader_val.inputs(opt.batch_size,opt.max_steps)  # batch_size, num_epochs
    
    #Construct input accordingly
    input_ts_val = construct_input(opt, data_dict_val)
    
    #Select model accordingly
    output_val,pred_landmark_val = construct_model(opt, data_dict_val, input_ts_val,is_training=False,is_reuse=True)    

    #Compute loss
    losses_val = compute_loss(output_val,data_dict_val,opt)
    #total_loss_val,depth_loss_val,landmark_loss_val,vis_loss_val,transformation_loss_val



    # val_lm_coord = argmax_2d(pred_landmark_val)
    # gt_lm_coord = argmax_2d(data_dict_val['points2D'])
    # diff = val_lm_coord-gt_lm_coord
    # avg_dist = tf.reduce_mean(tf.sqrt(tf.cast(tf.reduce_sum(diff**2,1),tf.float32)))
  

#Summaries
share_loss = tf.placeholder(tf.bool, shape=())
inputloss = tf.cond(share_loss, lambda: losses, lambda: losses_val)
input_landmark = tf.cond(share_loss, lambda: pred_landmark, lambda: pred_landmark_val)
merged = construct_summary(inputloss,data_dict,input_landmark)
#Summaries
#eval_merged = construct_summary(losses_val,data_dict_val)



with tf.name_scope("train_op"):

    #Optimization
    train_vars = [var for var in tf.trainable_variables()]
    optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
    train_op = slim.learning.create_train_op(losses[0], optim)
    global_step = tf.Variable(0,
                                name = 'global_step',
                                trainable = False)
    incr_global_step = tf.assign(global_step,global_step+1)


#Start training
with tf.name_scope("parameter_count"):
    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                    for v in tf.trainable_variables()])

saver = tf.train.Saver([var for var in tf.model_variables()] + \
                            [global_step],
                            max_to_keep=10)

# sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
#                             save_summaries_secs=0, 
#                             saver=None)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Session start
with tf.Session(config=config) as sess:#sv.managed_session(config=config) as sess:


    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(opt.checkpoint_dir + '/logs/train',
                                            sess.graph)
    eval_writer = tf.summary.FileWriter(opt.checkpoint_dir + '/logs/eval')


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

