import tensorflow as tf
import numpy as np
import time
import math
import os
import cv2
import tensorflow.contrib.slim as slim
from estimator_rui import *
import xlsxwriter


def get_lanmark_loc_from_hm(mask,thresh):

    ind = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
    if mask[ind]<thresh:
        ind = [-1,-1]

    return ind

#---------------------------------------------
#Function to draw landmark points on image
#---------------------------------------------
def drawlandmark(image,points2D,outname,visibility):

    image_landmark = np.copy(image)#np.zeros(image.shape, np.uint8)
    for i in range(points2D.shape[1]):
        if visibility[i]==1:
            cv2.circle(image_landmark,(int(np.round(points2D[0,i])),int(np.round(points2D[1,i]))), 2, (0,255,0), -1)
        else:
            cv2.circle(image_landmark,(int(np.round(points2D[0,i])),int(np.round(points2D[1,i]))), 2, (0,0,255), -1)

    #image_landmark = cv2.resize(image_landmark,(640,480),interpolation = cv2.INTER_AREA)
    cv2.imwrite(outname,image_landmark)

def prediction(opt,
             m_trainer,
             data_dict,
             output):

    #import pdb;pdb.set_trace()
    eps = 0.000001
    #Summaries
    m_trainer.construct_summary(data_dict,output)

    global_step = tf.Variable(0,
                                name = 'global_step',
                                trainable = False)
    incr_global_step = tf.assign(global_step,global_step+1)


    with tf.Session() as sess:
    
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        test_writer = tf.summary.FileWriter(opt.checkpoint_dir + '/logs/predict',
                                                sess.graph)
        merged = tf.summary.merge_all()
        model_vars = collect_vars(m_trainer.scope_name)
        model_vars['global_step'] = global_step
        saver = tf.train.Saver(model_vars)

        checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
        saver.restore(sess, checkpoint)
        count = 0
        try:
            while True:
                fetches = {
                    "output": output,
                    "image": data_dict["image"],
                    "IR": data_dict["IR"],
                    "global_step": global_step,
                    "incr_global_step": incr_global_step
                }
                fetches["summary"] = merged
    
                if opt.model=="pose":
                    fetches["pose"] = pose
                    fetches["quaternion"] = data_dict["quaternion"]
                    fetches["translation"] = data_dict["translation"]
                    fetches["depth"] = data_dict["depth"]
                    fetches["matK"] = data_dict["matK"]
                results = sess.run(fetches)
                gs = results["global_step"]
                test_writer.add_summary(results["summary"],gs)
    
                if opt.with_seg:
                    #Quantitative evaluation
                    z = results["pred"][0][0,:,:,0]
                    z[z>0.5]=1.0
                    z[z<=0.5]=0.0         
    
                #Result dir
                points2D = np.zeros([3,28],dtype=np.float32)
                thresh = 3#np.max(results["gt_landmark"][0,:,:,:])/2.0
                
                for tt in range(28):
                    import pdb;pdb.set_trace()
                    ind = get_lanmark_loc_from_hm(results["output"][0][0,:,:,tt],thresh)
                    
                    points2D[0,tt]=ind[1]
                    points2D[1,tt]=ind[0]

    
                visibility=np.ones(points2D.shape[1],dtype=np.float64)
                drawlandmark(results["image"][0,:,:,:]*255.0,points2D, os.path.join('./test','landmark'+str(count)+'.png'),visibility)
                count = count+1
                print("The %s frame is processed"%(count))
    
        except tf.errors.OutOfRangeError:
            print('Done ')
