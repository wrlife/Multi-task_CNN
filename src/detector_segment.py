import tensorflow as tf
import random
import numpy as np
#import PIL.Image as pil
from glob import glob
import cv2
import os

from model import *
#from data_loader_direct import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"]="2"


img_height = 480
img_width = 640

colors = []
for cc in range(28):
    colors.append((np.random.randint(255),np.random.randint(255),np.random.randint(255)))


def compute_homography_warp(points2D,image,imagename):

    #Read target point locations
    
    with open('clinic_landmark.txt') as f:
        lines = f.readlines()

    points2D_tgt = np.zeros([2,28],dtype=np.float32)
    for j in range(28):
        points2D_tgt[:,j] = [float(x) for x in lines[j].split()]
        #import pdb;pdb.set_trace()
        points2D_tgt[0,j] = points2D_tgt[0,j]*640/224
        points2D_tgt[1,j] = points2D_tgt[1,j]*480/224

    src = []
    tgt = []

    for j in range(28):
        if points2D[0,j]!=-1:
            tgt.append(points2D_tgt[:,j])
            src.append(points2D[:,j])
    #import pdb;pdb.set_trace()
    src = np.asarray(src)
    tgt = np.asarray(tgt)
    if(src.shape[0]<5):
        return

    M, mask = cv2.findHomography(src,tgt, cv2.RANSAC)

    try:
        im_dst = cv2.warpPerspective(image, M, (img_width,img_height))
    except:
        return
    
    

    cv2.imwrite(imagename,im_dst)

    return M
       



def predict(inputs,model,checkpoint_dir,with_seg, method):

    tf.reset_default_graph()

    #Load image and label
    image_batch = tf.placeholder(shape=[None, img_height, img_width, 3], dtype=tf.float32)
    depth_batch = tf.placeholder(shape=[None, img_height, img_width, 1], dtype=tf.float32)
    ir_batch = tf.placeholder(shape=[None, img_height, img_width, 3], dtype=tf.float32)
    
    f = open('./test.txt', 'r')
    
    
    #Concatenate color and depth for model input
    if inputs == "all":
        input_ts = tf.concat([ir_batch,depth_batch,image_batch],axis=3) #depth_batch,
    elif inputs == "IR_depth":
        input_ts = tf.concat([ir_batch,depth_batch],axis=3)
    elif inputs == "depth_color":
        input_ts = tf.concat([depth_batch,image_batch],axis=3)
    elif inputs =="IR_color":
        input_ts = tf.concat([ir_batch,image_batch],axis=3)
    elif inputs =="IR":
        input_ts = ir_batch
    elif inputs =="color":
        input_ts = image_batch
    elif inputs =="depth":
        input_ts = depth_batch
    
    if model=="lastdecode":
        pred, pred_landmark, _ = disp_net(tf.cast(input_ts,tf.float32), is_training = False)
    elif model=="single":
        pred, pred_landmark, _ = disp_net_single(tf.cast(input_ts,tf.float32), is_training = False)
    elif model=="multiscale":
        pred, pred_landmarks, _ = disp_net_single_multiscale(tf.cast(input_ts,tf.float32))
        pred_landmark = pred_landmarks[0]
    elif model=="hourglass":
        initial_output = disp_net_initial(tf.cast(input_ts,tf.float32))
        input_ts = tf.concat([input_ts,initial_output[1]],axis=3)
        refine_output = disp_net_refine(tf.cast(input_ts,tf.float32))
        pred = refine_output[0]
        pred_landmark = refine_output[1] 
    
    saver = tf.train.Saver([var for var in tf.model_variables()])
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    print(checkpoint)
    
    
    def get_lanmark_loc_from_hm(mask):
    
        ind = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
        if mask[ind]<1000:
            ind = [-1,-1]
    
        return ind
    
    #---------------------------------------------
    #Function to draw landmark points on image
    #---------------------------------------------
    def drawlandmark(image,points2D,outname,visibility):
    
        image_landmark = np.copy(image)#np.zeros(image.shape, np.uint8)
        
        for i in range(points2D.shape[1]):
            if visibility[i]==1:
                cv2.circle(image_landmark,(int(np.round(points2D[0,i])),int(np.round(points2D[1,i]))), 5, colors[i], -1)
            else:
                cv2.circle(image_landmark,(int(np.round(points2D[0,i])),int(np.round(points2D[1,i]))), 5, (0,0,255), -1)
    
        #image_landmark = cv2.resize(image_landmark,(640,480),interpolation = cv2.INTER_AREA)
        cv2.imwrite(outname,image_landmark)
    
    
    with tf.Session() as sess:
    
        saver.restore(sess, checkpoint)
        
        pointscount=0.0
        totalcount=0.0
    
        for line in f:
        
            
            datadir=line[:-8]
            name = line[-8:-1]
            
            #import pdb;pdb.set_trace()
            fh = os.path.join(datadir,'color', name +'color.png.color.png')
            
            #print("start %s frame"%(fh))
            # image = pil.open(fh)
            # image = np.array(image)

            image = cv2.imread(fh)
            if image is None:
                continue
            
            #image = np.expand_dims(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),axis=2)
            #I = cv2.resize(image,(224,224),interpolation = cv2.INTER_AREA)
            I = image.astype(np.float32)/255.0
    
            dh = line[:-1]+'depth1.png'
            depth = cv2.imread(dh,-1)
            #depth = cv2.resize(depth,(224,224),interpolation = cv2.INTER_AREA)
            depth = depth/1600.0
            depth = np.expand_dims(depth,axis=2)
    
            irh = line[:-1]+'ir.png'
            ira = cv2.imread(irh,-1)
            #ir = cv2.resize(ir,(224,224),interpolation = cv2.INTER_AREA)
            ir = ira/255.0
            #ir = np.expand_dims(depth,axis=2)        
    
            #import pdb;pdb.set_trace()
            pred_board,landmark = sess.run([pred,pred_landmark],feed_dict={ir_batch:ir[None,:,:,:],
                                                                    image_batch:I[None,:,:,:],
                                                                    depth_batch:depth[None,:,:,:]})
            #Result dir
            directory = os.path.join('./segment',method)
            if not os.path.exists(directory):
                os.makedirs(directory)
                    
            if with_seg:
                redimage = np.zeros_like(image,image.dtype)
                redimage[:,:]=(0,255,0)
                z = pred_board[0][0,:,:,:]
                z = np.repeat(z,3,axis=2)
                #z[z<=0.5]=0
                #z[z>0.5] = redimage[z>0.5]
                z = cv2.resize(z,(image.shape[1],image.shape[0]),interpolation = cv2.INTER_AREA)
                image[z>0.5] = redimage[z>0.5]
                #cv2.addWeighted(z.astype(np.uint8), 1, image, 1, 0, image)
                
                image = cv2.resize(image,(640,480),interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(directory,name+'segm_ir_d_rgb_lmhm.png'),image)
    
            points2D = np.zeros([2,28],dtype=np.float32)
    
            for tt in range(landmark[0].shape[2]):
                #import pdb;pdb.set_trace()
                ind = get_lanmark_loc_from_hm(landmark[0][:,:,tt])
                points2D[0,tt]=ind[1]
                points2D[1,tt]=ind[0]
                
                if ind[1]!=-1:
                    pointscount = pointscount+1
    

            imagename = os.path.join(directory,name+'.warped.png')
            compute_homography_warp(points2D,image,imagename)

            visibility=np.ones(points2D.shape[1],dtype=np.float64)
            #drawlandmark(I*255.0,points2D, os.path.join(directory,name+'.landmark.png'),visibility)
            drawlandmark(ira,points2D, os.path.join(directory,name+'.landmark.png'),visibility)
            totalcount = totalcount+1
            
        avg_points = pointscount/totalcount    
        f = open(os.path.join(directory,'avgpoints.txt'),'w')
        f.write(np.str(avg_points)+'\n')
        f.close()
    
            
            

predict("IR","single","/home/z003xr2y/data/Multi-task_CNN/checkpoints_IR_single_hr/",False,"checkpoints_IR_single_hr")
# predict("IR_color","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_IR_color_landmark_multiscale/",False,"checkpoints_IR_color_landmark_multiscale")
# predict("IR_depth","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_IR_depth_landmark_multiscale/",False,"checkpoints_IR_depth_landmark_multiscale_test")
# predict("depth_color","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_depth_color_landmark_multiscale/",False,"checkpoints_depth_color_landmark_multiscale")
# predict("depth","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_depth_landmark_multiscale/",False,"checkpoints_depth_landmark_multiscale")
# predict("color","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_color_landmark_multiscale/",False,"checkpoints_color_landmark_multiscale")
# predict("IR","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_IR_landmark_multiscale/",False,"checkpoints_IR_landmark_multiscale")
#predict("all","single","/home/z003xr2y/data/Multi-task_CNN/checkpoints_IR_depth_color_landmark_dataaug_single/",False,"checkpoints_IR_depth_color_landmark_dataaug_single")
#predict("all","single","/home/z003xr2y/data/Multi-task_CNN/checkpoints_IR_depth_color_landmark_seg_single/",True,"checkpoints_IR_depth_color_landmark_seg_single")
#predict("IR_depth","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_IR_depth_landmark__multiscale/",False,"checkpoints_IR_depth_landmark_multiscale")