import tensorflow as tf
import random
import numpy as np
#import PIL.Image as pil
from glob import glob
import cv2
import os,sys

from model import *
from data_loader_direct import DataLoader
sys.path.insert(0,'/home/z003xr2y/data/Multi-task_CNN/py_img_seg_eval/')
from eval_segm import *
import xlsxwriter

img_height=224
img_width=224
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def evaluation(inputs,model,checkpoint_dir,with_seg,worksheet,row, method,thresh):

    tf.reset_default_graph()
    eps = 0.000001
    #Initialize data loader
    imageloader = DataLoader('/home/z003xr2y/data/tfrecords_val/',
                                5,
                                img_height, 
                                img_width,
                                'valid')
    # Load training data
    data_dict = imageloader.inputs(1,1)
    #import pdb;pdb.set_trace()
    #Concatenate color and depth for model input
    if inputs == "all":
        input_ts = tf.concat([data_dict['IR'],data_dict['depth'],data_dict['image']],axis=3) #data_dict['depth'],
    elif inputs == "IR_depth":
        input_ts = tf.concat([data_dict['IR'],data_dict['depth']],axis=3)
    elif inputs == "depth_color":
        input_ts = tf.concat([data_dict['depth'],data_dict['image']],axis=3)
    elif inputs =="IR_color":
        input_ts = tf.concat([data_dict['IR'],data_dict['image']],axis=3)
    elif inputs =="IR":
        input_ts = data_dict['IR']
    elif inputs =="color":
        input_ts = data_dict['image']
    elif inputs =="depth":
        input_ts = data_dict['depth']
        
    if model=="lastdecode":
        pred, pred_landmark, _ = disp_net(tf.cast(input_ts,tf.float32), is_training = False)
    elif model=="single":
        pred, pred_landmark, _ = disp_net_single(tf.cast(input_ts,tf.float32), is_training = False)
    elif model=="pose":
        pred, pred_landmark, pose, _ = disp_net_single_pose(tf.cast(input_ts,tf.float32))
    elif model=="multiscale":
        pred, pred_landmarks, _ = disp_net_single_multiscale(tf.cast(input_ts,tf.float32))
        pred_landmark = pred_landmarks[0]
    
    saver = tf.train.Saver([var for var in tf.model_variables()])
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    print(checkpoint)
    
    
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
    
        image_landmark = cv2.resize(image_landmark,(640,480),interpolation = cv2.INTER_AREA)
        cv2.imwrite(outname,image_landmark)
    
    
    with tf.Session() as sess:
    
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        
        pa=0;ma=0;mi=0;fwi=0;
        
        saver.restore(sess, checkpoint)
        count=0
        
        TP = np.zeros(28,dtype=np.float32)  # In the view and get detected points
        eu_dist = np.zeros(28,dtype=np.float32)    #clean points distance
        
        FP = np.zeros(28,dtype=np.float32)  # Out of view and get detected points
        #eu_dist = np.zeros(28,dtype=np.float32)    #clean points distance
        
        TN = np.zeros(28,dtype=np.float32)  # Out of view and not get detected points
        
        FN = np.zeros(28,dtype=np.float32)  # In the view and not get detected points
        
        Occ_TP = np.zeros(28,dtype=np.float32) #Occluded and get detected points
        eu_dist_occ = np.zeros(28,dtype=np.float32)    #occlude points distance
        
        Occ_FN = np.zeros(28,dtype=np.float32) #Occluded and not get detected points
        
        eu_dist_overall = np.zeros(28,dtype=np.float32) #Distance for all detected points
        pointscount = np.zeros(28,dtype=np.float32)  # In the view and get detected points
        
        
        try:
            while True:
                fetches = {
                    "pred":pred,
                    "pred_landmark": pred_landmark,
                    "gt_seg": data_dict["label"],
                    "gt_landmark": data_dict["points2D"],
                    "image": data_dict["image"],
                    "visibility": data_dict["visibility"]
                }
    
                if model=="pose":
                    fetches["pose"] = pose
                    fetches["quaternion"] = data_dict["quaternion"]
                    fetches["translation"] = data_dict["translation"]
                results = sess.run(fetches)
    
                #import pdb;pdb.set_trace()
    
                #redimage = np.zeros_like(image,image.dtype)
                #redimage[:,:]=(0,255,0)
                
                # #z = np.repeat(z,3,axis=2)
    
                # #z = cv2.resize(z,(image.shape[1],image.shape[0]),interpolation = cv2.INTER_AREA)
                # #image[z>0.5] = redimage[z>0.5]
    
                if with_seg:
                    #Quantitative evaluation
                    z = results["pred"][0][0,:,:,0]
                    z[z>0.5]=1.0
                    z[z<=0.5]=0.0
        
                    mask = results["gt_seg"][0,:,:,0]
                    pa += pixel_accuracy(z,mask)
                    ma += mean_accuracy(z,mask)
                    mi += mean_IU(z,mask)
                    fwi += frequency_weighted_IU(z,mask)            
    

                #Result dir
                # directory = os.path.join(datadir,'segment')
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                
                # image = cv2.resize(image,(640,480),interpolation = cv2.INTER_AREA)
                # cv2.imwrite(os.path.join(directory,name+'segm_ir_d_rgb_lmhm.png'),image)
    
                points2D = np.zeros([2,28],dtype=np.float32)
    
                
    
                for tt in range(28):
                    #import pdb;pdb.set_trace()
                    ind = get_lanmark_loc_from_hm(results["pred_landmark"][0,:,:,tt],thresh)
                    points2D[0,tt]=ind[1]
                    points2D[1,tt]=ind[0]
                    
                    ind_gt = get_lanmark_loc_from_hm(results["gt_landmark"][0,:,:,tt],thresh)
                    
                    #import pdb;pdb.set_trace()
                    #True positive of non-occlude case
                    if results["visibility"][0,tt]==1 and ind[1]!=-1:
                        TP[tt] = TP[tt]+1
                        eu_dist[tt] =  eu_dist[tt]+np.sqrt(np.square(ind[1]-ind_gt[1])+np.square(ind[0]-ind_gt[0]))
                    
                    #False positive case  
                    elif results["visibility"][0,tt]==0 and ind_gt[1]==-1 and ind[0]!=-1:
                        FP[tt] = FP[tt]+1
                        
                    #True negative
                    elif results["visibility"][0,tt]==0 and ind_gt[1]==-1 and ind[0]==-1:
                        TN[tt] = TN[tt]+1
                        
                    #False negative of non-occlude case
                    elif results["visibility"][0,tt]==1 and ind[0]==-1:
                        FN[tt] = FN[tt]+1
                        
                    #True positive of occlude case
                    elif results["visibility"][0,tt]==0 and ind_gt[1]!=-1 and ind[0]!=-1:
                        Occ_TP[tt] = Occ_TP[tt]+1
                        eu_dist_occ[tt] = eu_dist_occ[tt]+np.sqrt(np.square(ind[1]-ind_gt[1])+np.square(ind[0]-ind_gt[0]))
                        
                    #False negative of occlude case
                    elif results["visibility"][0,tt]==0 and ind_gt[1]!=-1 and ind[0]==-1:
                        Occ_FN[tt] = Occ_FN[tt]+1
                    
                    if ind[0]!=-1:
                        eu_dist_overall[tt] =  eu_dist_overall[tt]+np.sqrt(np.square(ind[1]-ind_gt[1])+np.square(ind[0]-ind_gt[0]))
                        pointscount[tt] = pointscount[tt]+1
    
                #visibility=np.ones(points2D.shape[1],dtype=np.float64)
                #drawlandmark(results["image"][0,:,:,:]*255.0,points2D, os.path.join('./test','landmark'+str(count)+'.png'),visibility)
                count = count+1

                #print("The %s frame is processed"%(count))
    
        except tf.errors.OutOfRangeError:
            print('Done ')
    
    
        pa = pa/count
        ma = ma/count
        mi = mi/count
        fwi = fwi/count
        print ("Pixel accuracy: %f"%pa)
        print ("Mean accuracy: %f"%ma)
        print ("Mean IU: %f"%mi)
        print ("Frequency weighted IU: %f"%fwi)
    
    
        #Generate graph and statistics
        #Sensitivity of non-occlude points
        #import pdb;pdb.set_trace()
        recall = (TP+eps)/(TP+FN+eps)
        recall_overall = np.sum(TP)/(np.sum(TP)+np.sum(FN))
        
        #Sensitivity of occlude points
        recall_occ = (Occ_TP+eps)/(Occ_TP+Occ_FN+eps)
        recall_occ_overall = np.sum(Occ_TP)/(np.sum(Occ_TP)+np.sum(Occ_FN))
        
        #Specificity of non-occlude points
        speci = (TN+eps)/(TN+FP+eps)
        speci_overall = np.sum(TN)/(np.sum(TN)+np.sum(FP))
        
        #Euclidean distance of non-occlude points
        eu_dist_each = (eu_dist+eps)/(TP+eps)
        eu_dist_avg = np.sum(eu_dist)/np.sum(TP)
        
        #Euclidean distance of occlude points
        eu_dist_occ_each = (eu_dist_occ+eps)/(Occ_TP+eps)
        eu_dist_occ_avg = np.sum(eu_dist_occ)/np.sum(Occ_TP)
        
        
        #EU distance of all detectd points
        eu_dist_overall_each = (eu_dist_overall+eps)/(pointscount+eps)
        eu_dist_overall_avg = np.sum(eu_dist_overall)/np.sum(pointscount)
        
        #EU distance normalized
        eu_dist_overall_normal = np.sum(eu_dist_overall_each)/28.0 
        
        #Avg num points per image
        avg_numpoint = np.sum(pointscount)/count
        
        #print (eu_dist_overall_each)
        #print("Avg dist: %f"%(eu_dist_overall_normal))
        
        #Generate report
        col=0
        worksheet.write_string(row,col,method)
        worksheet.write_number(row,col+1,recall_overall)
        worksheet.write_number(row,col+2,recall_occ_overall)
        worksheet.write_number(row,col+3,speci_overall)
        worksheet.write_number(row,col+4,eu_dist_avg)
        worksheet.write_number(row,col+5,eu_dist_occ_avg)
        worksheet.write_number(row,col+6,eu_dist_overall_avg)
        worksheet.write_number(row,col+7,eu_dist_overall_normal)
        worksheet.write_number(row,col+8,avg_numpoint)
        



for thresh in range(2000,20001,2000):

    # Create a workbook and add a worksheet.
    #import pdb;pdb.set_trace()
    workbook = xlsxwriter.Workbook('Evaluation%d.xlsx'%(thresh))
    worksheet = workbook.add_worksheet()
    
    worksheet.set_column(0, 8, 25)
    
    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': 1})
    
    
    # Write some data headers.
    worksheet.write('A1', 'Method', bold)
    worksheet.write('B1', 'recall_overall', bold)
    worksheet.write('C1', 'recall_occ_overall', bold)
    worksheet.write('D1', 'speci_overall', bold)
    worksheet.write('E1', 'eu_dist_avg', bold)
    worksheet.write('F1', 'eu_dist_occ_avg', bold)
    worksheet.write('G1', 'eu_dist_overall_avg', bold)
    worksheet.write('H1', 'eu_dist_overall_normal', bold)
    worksheet.write('I1', 'points_per_frame', bold)
    
    row = 1
    evaluation("IR","single","/home/z003xr2y/data/Multi-task_CNN/checkpoints_IR_landmark_single_domain_trans/",False,worksheet,row,"IR",thresh)
    # evaluation("IR_color","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_IR_color_landmark_multiscale/",False,worksheet,row+1,"IR+color",thresh)
    # evaluation("IR_depth","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_IR_depth_landmark_multiscale/",False,worksheet,row+2,"IR+depth",thresh)
    # evaluation("depth_color","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_depth_color_landmark_multiscale/",False,worksheet,row+3,"depth+color",thresh)
    # evaluation("depth","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_depth_landmark_multiscale/",False,worksheet,row+4,"depth",thresh)
    # evaluation("color","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_color_landmark_multiscale/",False,worksheet,row+5,"color",thresh)
    # evaluation("IR","multiscale","/home/z003xr2y/data/Multi-task_CNN/checkpoints_IR_landmark_multiscale/",False,worksheet,row+6,"IR",thresh)

    workbook.close()
