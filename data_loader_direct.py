from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
import os, glob


class DataLoader(object):
    def __init__(self,
                 dataset_dir,
                 batch_size,
                 image_height,
                 image_width,
                 split):
        self.dataset_dir=dataset_dir
        self.batch_size=batch_size
        self.image_height=image_height
        self.image_width=image_width
        self.split=split



    #==================================
    # Load training data from tf records
    #==================================

    def inputs(self,batch_size, num_epochs,with_aug=False):
        """Reads input data num_epochs times.
        Args:
            train: Selects between the training (True) and validation (False) data.
            batch_size: Number of examples per returned batch.
            num_epochs: Number of times to read the input data, or 0/None to
            train forever.
        Returns:
            A tuple (images, labels), where:
            * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
            in the range [-0.5, 0.5].
            * labels is an int32 tensor with shape [batch_size] with the true label,
            a number in the range [0, mnist.NUM_CLASSES).
            This function creates a one_shot_iterator, meaning that it will only iterate
            over the dataset once. On the other hand there is no special initialization
            required.
        """
        def decode(serialized_example):
            """Parses an image and label from the given `serialized_example`."""
            features = tf.parse_single_example(
                serialized_example,
                # Defaults are not specified since both keys are required.
                features={
                    'color': tf.FixedLenFeature([], tf.string),
                    'IR': tf.FixedLenFeature([], tf.string),
                    'depth': tf.FixedLenFeature([], tf.string),
                    'mask': tf.FixedLenFeature([], tf.string),
				    'quaternion': tf.FixedLenFeature([], tf.string),
				    'translation': tf.FixedLenFeature([], tf.string),
                    'landmark_heatmap': tf.FixedLenFeature([], tf.string),
                    'visibility': tf.FixedLenFeature([], tf.string),
                    'matK': tf.FixedLenFeature([], tf.string),
                })

            # Convert from a scalar string tensor (whose single string has
            # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
            # [mnist.IMAGE_PIXELS].
            image = tf.decode_raw(features['color'], tf.float64)
            IR = tf.decode_raw(features['IR'], tf.float32)/255.0
            depth = tf.decode_raw(features['depth'], tf.float32)/100.0
            label = tf.decode_raw(features['mask'], tf.uint8)
            quaternion = tf.decode_raw(features['quaternion'], tf.float64)
            translation = tf.decode_raw(features['translation'], tf.float64)
            points2D = tf.decode_raw(features['landmark_heatmap'], tf.float32)
            visibility = tf.decode_raw(features['visibility'], tf.float32)
            matK = tf.decode_raw(features['matK'], tf.float64)

            image =  tf.cast(tf.reshape(image,[self.image_height, self.image_width, 3]),tf.float32)/255.0
            #image = tf.image.rgb_to_grayscale(image)/255.0

            IR = tf.cast(tf.reshape(IR,[self.image_height, self.image_width, 3]),tf.float32)
            depth = tf.cast(tf.reshape(depth,[self.image_height, self.image_width, 1]),tf.float32)
            label = tf.reshape(label,[self.image_height, self.image_width, 1])
            quaternion = tf.cast(tf.reshape(quaternion,[4]),tf.float32)

            translation = tf.cast(tf.reshape(translation,[3]),tf.float32)
            #import pdb;pdb.set_trace()
            norm = tf.sqrt(tf.reduce_sum(tf.square(translation),0, keep_dims=True))
            translation = translation / norm
            translation = tf.concat([translation,norm],axis=0)
            
            points2D = tf.reshape(points2D,[self.image_height, self.image_width,28])*5000.0

            #points2D = tf.reverse(points2D,[2])

            # points2D = tf.transpose(points2D,perm=[1,0])
            # sp1_gt, sp2_gt = tf.split(points2D, 2, 1)
            # sp1_gt = sp1_gt/self.image_width
            # sp2_gt = sp2_gt/self.image_height
            # points2D = tf.concat([sp1_gt,sp2_gt],axis=1)

            visibility.set_shape([28])
            visibility = tf.cast(visibility,tf.float32)
            matK = tf.cast(tf.reshape(matK,[3,3]),tf.float32)

            # Convert label from a scalar uint8 tensor to an int32 scalar.
            label = tf.cast(label, tf.float32)/255.0


            #Data augmentationmamatK
            data_dict = {}
            data_dict['image'] = image
            data_dict['IR'] = IR
            data_dict['depth'] = depth
            data_dict['label'] = label
            data_dict['quaternion'] = quaternion
            data_dict['translation'] = translation
            data_dict['points2D'] = points2D
            data_dict['visibility'] = visibility
            data_dict['matK'] = matK

            return data_dict

        def augment(data_dict):
        
            ir_batch, image_batch, depth_batch, label_batch,landmark_batch = self.data_augmentation(data_dict['IR'], data_dict['image'], data_dict['depth'],data_dict['label'], data_dict['points2D'],224,224)
            data_dict['image'] = image_batch
            data_dict['depth'] = depth_batch
            data_dict['label'] = label_batch
            data_dict['points2D'] = landmark_batch
            data_dict['IR'] = ir_batch
            
            return data_dict
            

        if not num_epochs:
            num_epochs = None
        filenames = glob.glob(os.path.join(self.dataset_dir,'*.tfrecords'))

        with tf.name_scope('input'):
            # TFRecordDataset opens a binary file and reads one record at a time.
            # `filename` could also be a list of filenames, which will be read in order.
            dataset = tf.data.TFRecordDataset(filenames)

            # The map transformation takes a function and applies it to every element
            # of the dataset.
            dataset = dataset.map(decode)
            # dataset = dataset.map(augment)
            # dataset = dataset.map(normalize)

            # The shuffle transformation uses a finite-sized buffer to shuffle elements
            # in memory. The parameter is the number of elements in the buffer. For
            # completely uniform shuffling, set the parameter to be the same as the
            # number of elements in the dataset.
            #dataset = dataset.shuffle(1000)#1000 + 3 * batch_size)
            dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(batch_size)
            if with_aug:
                dataset = dataset.map(augment)

            iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


    #==================================
    # Load training data from tf records
    #==================================

    def inputs_test(self,batch_size, num_epochs,with_aug=False):
        """Reads input data num_epochs times.
        Args:
            train: Selects between the training (True) and validation (False) data.
            batch_size: Number of examples per returned batch.
            num_epochs: Number of times to read the input data, or 0/None to
            train forever.
        Returns:
            A tuple (images, labels), where:
            * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
            in the range [-0.5, 0.5].
            * labels is an int32 tensor with shape [batch_size] with the true label,
            a number in the range [0, mnist.NUM_CLASSES).
            This function creates a one_shot_iterator, meaning that it will only iterate
            over the dataset once. On the other hand there is no special initialization
            required.
        """
        def decode(serialized_example):
            """Parses an image and label from the given `serialized_example`."""
            features = tf.parse_single_example(
                serialized_example,
                # Defaults are not specified since both keys are required.
                features={
                    'color': tf.FixedLenFeature([], tf.string),
                    'IR': tf.FixedLenFeature([], tf.string),
                    'depth': tf.FixedLenFeature([], tf.string),
                    'matK': tf.FixedLenFeature([], tf.string),
                })

            # Convert from a scalar string tensor (whose single string has
            # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
            # [mnist.IMAGE_PIXELS].
            image = tf.decode_raw(features['color'], tf.float64)/255.0
            IR = tf.decode_raw(features['IR'], tf.float64)/255.0
            depth = tf.decode_raw(features['depth'], tf.float64)/100.0
            matK = tf.decode_raw(features['matK'], tf.float64)

            image =  tf.cast(tf.reshape(image,[self.image_height, self.image_width, 3]),tf.float32)
            IR = tf.cast(tf.reshape(IR,[self.image_height, self.image_width, 3]),tf.float32)
            depth = tf.cast(tf.reshape(depth,[self.image_height, self.image_width, 1]),tf.float32)
            matK = tf.cast(tf.reshape(matK,[3,3]),tf.float32)

            #Data augmentationmamatK
            data_dict = {}
            data_dict['image'] = image
            data_dict['IR'] = IR
            data_dict['depth'] = depth
            data_dict['matK'] = matK

            return data_dict


        if not num_epochs:
            num_epochs = None
        filenames = glob.glob(os.path.join(self.dataset_dir,'*.tfrecords'))

        with tf.name_scope('input_test'):
            # TFRecordDataset opens a binary file and reads one record at a time.
            # `filename` could also be a list of filenames, which will be read in order.
            dataset = tf.data.TFRecordDataset(filenames)

            # The map transformation takes a function and applies it to every element
            # of the dataset.
            dataset = dataset.map(decode)
            # dataset = dataset.map(augment)
            # dataset = dataset.map(normalize)

            # The shuffle transformation uses a finite-sized buffer to shuffle elements
            # in memory. The parameter is the number of elements in the buffer. For
            # completely uniform shuffling, set the parameter to be the same as the
            # number of elements in the dataset.
            dataset = dataset.shuffle(1000)#1000 + 3 * batch_size)
            dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()


    #================================
    # Load rgb, depth, and mask through txt
    #================================
    def load_data_batch(self,split):

        # Reads pfathes of images together with their labels
        image_list,depth_list,label_list,ir_list,landmark_list = self.read_labeled_image_list(split)
        steps_per_epoch = int(len(image_list)//self.batch_size)

        #import pdb;pdb.set_trace()
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        depths = tf.convert_to_tensor(depth_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.string)
        irs = tf.convert_to_tensor(ir_list, dtype=tf.string)
        landmarks = tf.convert_to_tensor(landmark_list, dtype=tf.string)

        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images,depths,labels,irs,landmarks], 
                                                    num_epochs = 900,
                                                    shuffle=True)

        image,depth,label = self.read_images_from_disk(input_queue)

        # Optional Image and Label Batching
        image.set_shape((self.image_height, self.image_width, 3))
        depth.set_shape([self.image_height, self.image_width, 1])
        label.set_shape([self.image_height, self.image_width, 1])
        image_batch, depth_batch, label_batch = tf.train.batch([image,depth,label],
                                    num_threads = 8, batch_size=self.batch_size)

        # Data augmentation
        if split=='train':
            image_batch, depth_batch, label_batch = self.data_augmentation(image_batch, depth_batch, label_batch,224,224)
        
        return image_batch, depth_batch, label_batch, steps_per_epoch





    #===================================
    # Get file name list
    #===================================

    def read_labeled_image_list(self,split):
        """Reads a .txt file containing pathes and labeles
        Args:
           image_list_file: a .txt file with one /path/to/image per line
           label: optionally, if set label will be pasted after each line
        Returns:
           List with all filenames in file image_list_file
        """
        
        f = open(self.dataset_dir+'/'+split+'.txt', 'r')
        colorimages = []
        depthimages = []
        labelnames = []
        irnames = []
        landmarknames = []

        

        for line in f:

            basepath = line[:-8]
            name = line[-8:-1]

            colorimage = basepath+'\\color\\'+name+'color.png.color.png'
            colorimages.append(colorimage)

            depthimage = line[:-1]+'depth1.png'
            depthimages.append(depthimage)

            # irimage = line[:-1]+'ir.png'
            # irnames.append(irimage)
            
            if split=='train' or split=='valid':
                labelname = basepath+'\\mask\\'+name+'color.png.landmark_filtered.png'
                labelnames.append(labelname)
                #landmarkname = basepath+'\\landmark\\'+name+'landmark.txt'
                #landmarknames.append(landmarkname)
                
            else:
                labelname = line[:-1]+'depth0.png'
                labelnames.append(labelname)               

        return colorimages,depthimages,labelnames#,irnames,landmarknames


    def read_images_from_disk(self,input_queue):
        """Consumes a single filename and label as a ' '-delimited string.
        Args:
          filename_and_label_tensor: A scalar string tensor.
        Returns:
          Two tensors: the decoded image, and the string label.
        """

        #import pdb;pdb.set_trace()
        image_file = tf.read_file(input_queue[0])
        depth_file = tf.read_file(input_queue[1])
        label_file = tf.read_file(input_queue[2])
        ir_file = tf.read_file(input_queue[3])
        landmark_file = tf.read_file(input_queue[4])


        image = tf.to_float(tf.image.resize_images(tf.image.decode_png(image_file),[224,224]))

        depth = tf.to_float(tf.image.resize_images(tf.image.decode_png(depth_file,dtype=tf.uint16),[224,224]))

        label = tf.to_float(tf.image.resize_images(tf.image.decode_png(label_file),[224,224]))

        ir = tf.to_float(tf.image.resize_images(tf.image.decode_png(ir_file),[224,224]))

        #record_defaults = [[0.0]]*
        #proj_landmark_vis = 

        image = image/255.0

        depth = depth/1600.0

        label = tf.expand_dims(label[:,:,0],2)

        label = label/255.0

        ir = ir/255.0

        return image, depth, label


    def data_augmentation(self, ir, image, depth, label, landmark, out_h, out_w):

        def _random_true_false():
            prob = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            predicate = tf.less(prob, 0.5)
            return predicate

        # Random scaling
        def random_scaling(ir, image, depth, label,landmark):
            batch_size, in_h, in_w, _ = image.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.25)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)

            image = tf.image.resize_area(image, [out_h, out_w])
            depth = tf.image.resize_nearest_neighbor(depth, [out_h, out_w])
            label = tf.image.resize_nearest_neighbor(label, [out_h, out_w])
            ir = tf.image.resize_area(ir, [out_h, out_w])
            landmark = tf.image.resize_nearest_neighbor(landmark, [out_h, out_w])
            return ir, image, depth, label,landmark

        # Random cropping
        def random_cropping(ir, image, depth, label,landmark, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(image))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]

            image = tf.image.crop_to_bounding_box(
                image, offset_y, offset_x, out_h, out_w)
            depth = tf.image.crop_to_bounding_box(
                depth, offset_y, offset_x, out_h, out_w)
            label = tf.image.crop_to_bounding_box(
                label, offset_y, offset_x, out_h, out_w)
            landmark = tf.image.crop_to_bounding_box(
                landmark, offset_y, offset_x, out_h, out_w)
            ir = tf.image.crop_to_bounding_box(
                ir, offset_y, offset_x, out_h, out_w)
                
            return ir, image, depth, label, landmark

        # Random flip
        def random_flip(ir, image, depth, label,landmark):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            flip1 = _random_true_false()
            image = tf.cond(flip1, lambda:tf.image.flip_left_right(image),lambda:image)
            depth = tf.cond(flip1, lambda:tf.image.flip_left_right(depth),lambda:depth)
            label = tf.cond(flip1, lambda:tf.image.flip_left_right(label),lambda:label)
            landmark = tf.cond(flip1, lambda:tf.image.flip_left_right(landmark),lambda:landmark)
            ir = tf.cond(flip1, lambda:tf.image.flip_left_right(ir),lambda:ir)

            flip2 = _random_true_false()
            image = tf.cond(flip2, lambda:tf.image.flip_up_down(image),lambda:image)
            depth = tf.cond(flip2, lambda:tf.image.flip_up_down(depth),lambda:depth)
            label = tf.cond(flip2, lambda:tf.image.flip_up_down(label),lambda:label)
            landmark = tf.cond(flip2, lambda:tf.image.flip_up_down(landmark),lambda:landmark)
            ir = tf.cond(flip2, lambda:tf.image.flip_up_down(ir),lambda:ir)

            return ir,image, depth, label,landmark

        def random_color(image):

            color_ordering = np.random.randint(4, size = 1)
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)   

            return image     

        ir, image, depth, label,landmark = random_scaling(ir, image, depth, label,landmark)
        ir, image, depth, label,landmark = random_cropping(ir, image, depth, label,landmark, out_h, out_w)
        ir ,image, depth, label,landmark = random_flip(ir, image, depth, label,landmark)
        #image = random_color(image)

        return ir, image, depth, label,landmark

        

