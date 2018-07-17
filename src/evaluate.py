import tensorflow as tf
import numpy as np
import time
import math
import os
import cv2
import tensorflow.contrib.slim as slim
from estimator_rui import *
import xlsxwriter


#---------------------------------------------
#Function to draw landmark points on image
#---------------------------------------------
def drawlandmark(image,points2D,outname,visibility):

    image_landmark = np.copy(image)#np.zeros(image.shape, np.uint8)
    for i in range(points2D.shape[1]):
        if visibility[i]==1:
            cv2.circle(image_landmark,(int(np.round(points2D[0,i])),int(np.round(points2D[1,i]))), 2, (0,255,0), -1)
        # else:
        #     cv2.circle(image_landmark,(int(np.round(points2D[0,i])),int(np.round(points2D[1,i]))), 2, (0,0,255), -1)

    #image_landmark = cv2.resize(image_landmark,(640,480),interpolation = cv2.INTER_AREA)
    cv2.imwrite(outname,image_landmark)


def get_lanmark_loc_from_hm(mask,thresh):

    ind = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
    if mask[ind]<thresh:
        ind = [-1,-1]

    return ind

def evaluate(opt,
             filename,
             m_trainer,
             losses,
             data_dict,
             output,
             global_step,
             coord_pair
             ):

    eps = 0.000001
    #Summaries
    m_trainer.construct_summary(data_dict,output,losses)
    thresh = 10000
    workbook = xlsxwriter.Workbook(filename+'%d.xlsx'%(thresh))
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
    worksheet.write('J1', 'Register_err', bold)
    worksheet.write('K1', 'visibility_err', bold)

    # global_step = tf.Variable(0,
    #                             name = 'global_step',
    #                             trainable = False)
    # incr_global_step = tf.assign(global_step,global_step+1)


    with tf.Session() as sess:
    
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        test_writer = tf.summary.FileWriter(opt.checkpoint_dir + '/logs/test',
                                                sess.graph)
        merged = tf.summary.merge_all()

        pa=0;ma=0;mi=0;fwi=0;
        # model_vars = collect_vars(m_trainer.scope_name)
        # model_vars['global_step'] = global_step
        #saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=m_trainer.scope_name))#tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pose_net"))
        #import pdb;pdb.set_trace()
        #checkpoint1 = "/home/z003xr2y/data/Multi-task_CNN/src/checkpoints/IR_single/lr1_0.0001_lr2_0.001_numEncode5_numFeatures32/model-41002"#tf.train.latest_checkpoint(opt.checkpoint_dir)
        checkpoint2 =  "/home/z003xr2y/data/Multi-task_CNN/src/checkpoints/IR_single_pose_geo_prepose/lr1_1e-05_lr2_0.001_numEncode5_numFeatures32_small_notlearnsoft/model-68001"
        #print("Resume training from previous checkpoint: %s" % checkpoint)
        #saver1.restore(sess, checkpoint1)
        saver2.restore(sess, checkpoint2)
        count=0

        avg_trans_error = 0.0  # point to point registered error
        avg_vis_error = 0.0
        
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
                    "output": output,
                    "gt_seg": data_dict["label"],
                    "gt_landmark": data_dict["points2D"],
                    "image": data_dict["image"],
                    "visibility": data_dict["visibility"],
                    "global_step": global_step,
                    "trans_loss": losses[4],
                    "images": coord_pair,
                    #"proj_img": coord_pair[1]
                }
                fetches["summary"] = merged
    
                if opt.model=="pose":
                    fetches["pose"] = pose
                    fetches["quaternion"] = data_dict["quaternion"]
                    fetches["translation"] = data_dict["translation"]
                    fetches["depth"] = data_dict["depth"]
                    fetches["matK"] = data_dict["matK"]

                if opt.with_vis:
                    fetches["vis_loss"] = losses[3]
                
                results = sess.run(fetches)
                gs = results["global_step"]
                test_writer.add_summary(results["summary"],gs)

                #import pdb;pdb.set_trace()
                if opt.proj_img:
                    cv2.imwrite(os.path.join('./test','proj'+str(count)+'_p1.png'),(results["images"][0][0,:,:,:]+0.5)*255)
                    cv2.imwrite(os.path.join('./test','proj'+str(count)+'_t1.png'),(results["images"][1]+0.5)*255)
                    cv2.imwrite(os.path.join('./test','proj'+str(count)+'_p2.png'),(results["images"][2][0,:,:,:]+0.5)*255)
                    cv2.imwrite(os.path.join('./test','proj'+str(count)+'_t2.png'),(results["images"][3]+0.5)*255)
                #     cv2.imwrite(os.path.join('./test','proj_tgt'+str(count)+'.png'),(results["image"][1]+0.5)*255)
    
                if opt.with_seg:
                    #Quantitative evaluation
                    z = results["output"][1][0,:,:,0]
                    z[z>0.5]=1.0
                    z[z<=0.5]=0.0
        
                    mask = results["gt_seg"][0,:,:,0]

                    pa += np.sum(np.logical_and(z, mask))/np.sum(mask)
                    # pa += pixel_accuracy(z,mask)
                    # ma += mean_accuracy(z,mask)
                    # mi += mean_IU(z,mask)
                    # fwi += frequency_weighted_IU(z,mask)            
    

                #Result dir
                points2D = np.zeros([3,28],dtype=np.float32)
                #thresh = 4.0#np.max(results["gt_landmark"][0,:,:,:])/2.0

                avg_trans_error = avg_trans_error+results["trans_loss"]
                # print(results["points1"])
                # print(results["points2"])
                print(results["trans_loss"])

                if opt.with_vis:
                    avg_vis_error = avg_vis_error+results["vis_loss"]

                #import pdb;pdb.set_trace()
                thresh = np.max(results["output"][0])/2.0
                print(thresh)
                for tt in range(28):
                    
                    ind = get_lanmark_loc_from_hm(results["output"][0][0,:,:,tt],thresh)
                    
                    points2D[0,tt]=ind[1]
                    points2D[1,tt]=ind[0]
                    
                    # if ind[0]!=-1:
                        
                    #     points2D[2,tt]=results["depth"][0,ind[0],ind[1],0]*100
                    #     points3D = util.cam2world(np.expand_dims(points2D[:,tt],axis=1),results["matK"][0,0,2],results["matK"][0,1,2],results["matK"][0,0,0],results["matK"][0,1,1])
                    #     points3D_pred = np.dot(quaternion.as_rotation_matrix(quaternion.as_quat_array(results["pose"][0,0:4])),points3D)
                    #     points3D_gt = np.dot(quaternion.as_rotation_matrix(quaternion.as_quat_array(results["quaternion"][0])),points3D)
                    #     points3D_pred = points3D_pred[:,0]+results["pose"][0,4:-1]*results["pose"][0,-1]
                    #     points3D_gt = points3D_gt[:,0]+results["translation"][0,0:3]*results["translation"][0,-1]
                        #import pdb;pdb.set_trace()
                        
                    ind_gt = get_lanmark_loc_from_hm(results["gt_landmark"][0,:,:,tt],thresh)
                    
                    
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

                if opt.with_vis:
                    visibility=results["output"][2][0,:] #np.ones(points2D.shape[1],dtype=np.float64)
                    visibility[visibility>0.5] = 1.0
                    visibility[visibility<=0.5] = 0
                #import pdb;pdb.set_trace()
                #drawlandmark((results["image"][0,:,:,:]+0.5)*255.0,points2D, os.path.join('./test','landmark'+str(count)+'.png'),results["visibility"][0,:])
                count = count+1

                print("The %s frame is processed"%(count))
    
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
        recall_occ_overall = (np.sum(Occ_TP)+eps)/(np.sum(Occ_TP)+np.sum(Occ_FN)+eps)
        
        #Specificity of non-occlude points
        speci = (TN+eps)/(TN+FP+eps)
        speci_overall = (np.sum(TN)+eps)/(np.sum(TN)+np.sum(FP)+eps)
        
        #Euclidean distance of non-occlude points
        eu_dist_each = (eu_dist+eps)/(TP+eps)
        eu_dist_avg = (np.sum(eu_dist)+eps)/(np.sum(TP)+eps)
        
        #Euclidean distance of occlude points
        eu_dist_occ_each = (eu_dist_occ+eps)/(Occ_TP+eps)
        eu_dist_occ_avg = (np.sum(eu_dist_occ)+eps)/(np.sum(Occ_TP)+eps)
        
        
        #EU distance of all detectd points
        eu_dist_overall_each = (eu_dist_overall+eps)/(pointscount+eps)
        eu_dist_overall_avg = (np.sum(eu_dist_overall)+eps)/(np.sum(pointscount)+eps)
        
        #EU distance normalized
        eu_dist_overall_normal = np.sum(eu_dist_overall_each)/28.0 
        
        #Avg num points per image
        avg_numpoint = np.sum(pointscount)/count

        # Avg register error
        avg_trans_error = avg_trans_error/count

        # Avg vis error
        avg_vis_error = avg_vis_error/count      
        
        #print (eu_dist_overall_each)
        #print("Avg dist: %f"%(eu_dist_overall_normal))
        
        #Generate report
        col=0
        row=1
        worksheet.write_string(row,col,"single")
        worksheet.write_number(row,col+1,recall_overall)
        worksheet.write_number(row,col+2,recall_occ_overall)
        worksheet.write_number(row,col+3,speci_overall)
        worksheet.write_number(row,col+4,eu_dist_avg)
        worksheet.write_number(row,col+5,eu_dist_occ_avg)
        worksheet.write_number(row,col+6,eu_dist_overall_avg)
        worksheet.write_number(row,col+7,eu_dist_overall_normal)
        worksheet.write_number(row,col+8,avg_numpoint)
        worksheet.write_number(row,col+9,avg_trans_error)
        worksheet.write_number(row,col+10,avg_vis_error)

