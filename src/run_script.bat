python main.py --dataset_dir=D:\\Exp_data\\data\\2017_0216_DetectorDetection\\tfrecords_hr_homo ^
               --num_encoders=3 ^
               --num_features=16 ^
               --batch_size=2 ^
               --summary_freq=1 ^
               --learning_rate=0.0002 ^
               --inputs=IR --model=single ^
               --pretrain_pose=True --with_DH=True ^
               --proj_img=True --continue_train=True
