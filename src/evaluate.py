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

def evaluate(opt,
             m_trainer,
             losses,
             data_dict,
             output):

    eps = 0.000001
    #Summaries
    m_trainer.construct_summary(data_dict,output,losses)
    thresh = 50
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

    global_step = tf.Variable(0,
                                name = 'global_step',
                                trainable = False)
    incr_global_step = tf.assign(global_step,global_step+1)


    with tf.Session() as sess:
    
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        test_writer = tf.summary.FileWriter(opt.checkpoint_dir + '/logs/test',
                                                sess.graph)
        merged = tf.summary.merge_all()

        pa=0;ma=0;mi=0;fwi=0;
        model_vars = collect_vars(m_trainer.scope_name)
        model_vars['global_step'] = global_step
        saver = tf.train.Saver(model_vars)

        import pdb;pdb.set_trace()
        checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
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
                    "output": output,
                    "gt_seg": data_dict["label"],
                    "gt_landmark": data_dict["points2D"],
                    "image": data_dict["image"],
                    "visibility": data_dict["visibility"],
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
        
                    mask = results["gt_seg"][0,:,:,0]
                    pa += pixel_accuracy(z,mask)
                    ma += mean_accuracy(z,mask)
                    mi += mean_IU(z,mask)
                    fwi += frequency_weighted_IU(z,mask)            
    

                #Result dir
                points2D = np.zeros([3,28],dtype=np.float32)
                thresh = 4.0#np.max(results["gt_landmark"][0,:,:,:])/2.0
                #import pdb;pdb.set_trace()
                for tt in range(28):
                    import pdb;pdb.set_trace()
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
    
                visibility=np.ones(points2D.shape[1],dtype=np.float64)
                #drawlandmark(results["image"][0,:,:,:]*255.0,points2D, os.path.join('./test','landmark'+str(count)+'.png'),visibility)
                count = count+1
                #import pdb;pdb.set_trace()
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
        #mport pdb;pdb.set_trace()
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

