import tensorflow as tf
import random
import numpy as np
#import PIL.Image as pil
from glob import glob
import cv2
import os,sys

from model import *
from data_loader_direct import DataLoader
sys.path.insert(0,'/home/z003xr2y/data/Multi-task_CNN/py_img_seg_eval')
from eval_segm import *


img_height=224
img_width=224


#Initialize data loader
imageloader = DataLoader('/home/z003xr2y/data/tfrecords_val/',
                            5,
                            img_height, 
                            img_width,
                            'valid')

# Load training data
data_dict = imageloader.inputs(1,1)

#Concatenate color and depth for model input
input_ts = tf.concat([data_dict['IR'],data_dict['depth'],data_dict['image']],axis=3)
pred, pred_landmark,_ = disp_net(tf.cast(input_ts,tf.float32), is_training = False)

saver = tf.train.Saver([var for var in tf.model_variables()])
checkpoint = tf.train.latest_checkpoint("/home/z003xr2y/data/Multi-task_CNN/bk/checkpoints_IR_depth_color_landmark_hm_lastdecode/")
print(checkpoint)


def get_lanmark_loc_from_hm(mask):

    ind = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
    if mask[ind]<20000:
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

    image_landmark = cv2.resize(image_landmark,(640,480),interpolation = cv2.INTER_AREA)
    cv2.imwrite(outname,image_landmark)


with tf.Session() as sess:

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    
    pa=0;ma=0;mi=0;fwi=0;
    eu_dist = np.zeros(28,dtype=np.float32)
    saver.restore(sess, checkpoint)
    count=0
    pointcount = np.zeros(28,dtype=np.float32)
    try:
        while True:
            fetches = {
                "pred":pred,
                "pred_landmark": pred_landmark,
                "gt_seg": data_dict["label"],
                "gt_landmark": data_dict["points2D"],
                "image": data_dict["image"]
            }

            results = sess.run(fetches)


            #redimage = np.zeros_like(image,image.dtype)
            #redimage[:,:]=(0,255,0)
            # z = results["pred"][0][0,:,:,0]
            # #z = np.repeat(z,3,axis=2)

            # #z = cv2.resize(z,(image.shape[1],image.shape[0]),interpolation = cv2.INTER_AREA)
            # #image[z>0.5] = redimage[z>0.5]

            # #Quantitative evaluation
            # z[z>0.5]=1.0
            # z[z<=0.5]=0.0

            # mask = results["gt_seg"][0,:,:,0]
            # pa += pixel_accuracy(z,mask)
            # ma += mean_accuracy(z,mask)
            # mi += mean_IU(z,mask)
            # fwi += frequency_weighted_IU(z,mask)            


            #Result dir
            # directory = os.path.join(datadir,'segment')
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            
            # image = cv2.resize(image,(640,480),interpolation = cv2.INTER_AREA)
            # cv2.imwrite(os.path.join(directory,name+'segm_ir_d_rgb_lmhm.png'),image)

            points2D = np.zeros([2,28],dtype=np.float32)

            for tt in range(28):
                #import pdb;pdb.set_trace()
                ind = get_lanmark_loc_from_hm(results["pred_landmark"][0,:,:,tt])
                points2D[0,tt]=ind[1]
                points2D[1,tt]=ind[0]
                
                if ind[0]==-1:
                    continue
                pointcount[tt]=pointcount[tt]+1
                ind_gt = get_lanmark_loc_from_hm(results["gt_landmark"][0,:,:,tt])
                eu_dist[tt] =  eu_dist[tt]+np.sqrt(np.square(ind[1]-ind_gt[1])+np.square(ind[0]-ind_gt[0]))


            #visibility=np.ones(points2D.shape[1],dtype=np.float64)
            #drawlandmark(results["image"][0,:,:,:]*255.0,points2D, os.path.join('D:\\Exp_data\\data\\2017_0216_DetectorDetection\\test','landmark'+str(count)+'.png'),visibility)
            count = count+1
            #print("The %s frame is processed"%(count))

    except tf.errors.OutOfRangeError:
        print('Done ')

    #coord.request_stop()
    #coord.join(threads)

    pa = pa/count
    ma = ma/count
    mi = mi/count
    fwi = fwi/count
    print ("Pixel accuracy: %f"%pa)
    print ("Pixel accuracy: %f"%ma)
    print ("Pixel accuracy: %f"%mi)
    print ("Pixel accuracy: %f"%fwi)

    eu_dist = eu_dist/pointcount
    print (eu_dist)

    print(pointcount)
    import pdb;pdb.set_trace()
    print("Avg num point: %f"%(np.sum(pointcount)/count))

    print("Avg dist: %f"%(np.sum(eu_dist)/28.0))
