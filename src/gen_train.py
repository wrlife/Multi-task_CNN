import os,glob
#from XrayKinematics import *

directory = '/home/z003xr2y/data/tset/'

#Loop through all folders and get landmark location in image coord
dirlist = [x[0] for x in os.walk(directory)]

f = open('valid.txt','w')

#t_xk = XrayKinematics()

for dir in dirlist[1:]:
	
	imagelist = glob.glob(os.path.join(dir,'*color.png'))

	for i in range(len(imagelist)):

		prefix = imagelist[i][0:-9]
		#tmp_result = prefix+'result.txt'
		#tmp_tubepos = prefix+'tubepos.txt'

		#tubepos = t_xk.getTubePose(tmp_tubepos)
		#if tubepos is None:
		#	continue

		#Get detector to global transforamtion matrix
		#matDetector2Global = t_xk.getDetectorPose(tmp_result)
		#if matDetector2Global is None:
		#	continue
			
		f.write(prefix+'\n')



f.close()


