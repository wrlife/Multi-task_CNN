import tensorflow as tf
import random
import numpy as np
import PIL.Image as pil
from glob import glob
import cv2
import os

from model import *
#from data_loader_direct import DataLoader



#Load image and label
image_batch = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
depth_batch = tf.placeholder(shape=[None, 224, 224, 1], dtype=tf.float32)
ir_batch = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)

f = open('./test.txt', 'r')

# # Define the model:
#with tf.name_scope("Prediction"):

input_ts = tf.concat([ir_batch,image_batch,depth_batch],axis=3)
pred, pred_landmark,_ = disp_net(input_ts, is_training = False)

saver = tf.train.Saver([var for var in tf.model_variables()])
checkpoint = tf.train.latest_checkpoint("D:\\code\\depth_rnn\\checkpoints_l2_IR_color_depth_landmark_hm")
print(checkpoint)


def get_lanmark_loc_from_hm(mask):

    ind = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
    if mask[ind]<100:
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

    saver.restore(sess, checkpoint)

    for line in f:
        datadir=line[:-8]
        name = line[-8:-1]

        fh = os.path.join(datadir,'color', name +'color.png.color.png')
        # image = pil.open(fh)
        # image = np.array(image)
        image = cv2.imread(fh)
        I = cv2.resize(image,(224,224),interpolation = cv2.INTER_AREA)
        I = I/255.0

        dh = line[:-1]+'depth1.png'
        depth = cv2.imread(dh,-1)
        depth = cv2.resize(depth,(224,224),interpolation = cv2.INTER_AREA)
        depth = depth/16000.0
        depth = np.expand_dims(depth,axis=2)

        irh = line[:-1]+'ir.png'
        ir = cv2.imread(irh,-1)
        ir = cv2.resize(ir,(224,224),interpolation = cv2.INTER_AREA)
        ir = ir/255.0
        #ir = np.expand_dims(depth,axis=2)        

        pred_board,landmark = sess.run([pred,pred_landmark],feed_dict={ir_batch:ir[None,:,:,:],
                                                                image_batch:I[None,:,:,:],
                                                                depth_batch:depth[None,:,:,:]})


        redimage = np.zeros_like(image,image.dtype)
        redimage[:,:]=(0,255,0)
        z = pred_board[0][0,:,:,:]
        z = np.repeat(z,3,axis=2)
        #z[z<=0.5]=0
        #z[z>0.5] = redimage[z>0.5]
        z = cv2.resize(z,(image.shape[1],image.shape[0]),interpolation = cv2.INTER_AREA)
        image[z>0.5] = redimage[z>0.5]
        #cv2.addWeighted(z.astype(np.uint8), 1, image, 1, 0, image)

        #Result dir
        directory = os.path.join(datadir,'segment')
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        image = cv2.resize(image,(640,480),interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(directory,name+'segm_ir_d_rgb_lmhm.png'),image)

        points2D = np.zeros([2,28],dtype=np.float32)

        for tt in range(landmark[0].shape[2]):
            ind = get_lanmark_loc_from_hm(landmark[0][:,:,tt])
            points2D[0,tt]=ind[1]
            points2D[1,tt]=ind[0]

        visibility=np.ones(points2D.shape[1],dtype=np.float64)
        drawlandmark(I*255.0,points2D, os.path.join(directory,name+'.landmark.png'),visibility)

        print("The %s frame is processed"%(line))
