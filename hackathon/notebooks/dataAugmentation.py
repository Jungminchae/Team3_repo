import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from functools import partial
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate, RandomCrop, ShiftScaleRotate, VerticalFlip
)
AUTOTUNE = tf.data.experimental.AUTOTUNE

transforms = Compose([
            Rotate(limit=60),
            ShiftScaleRotate(),
            RandomBrightness(limit=0.2),
            RandomContrast(limit=0.2, p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip()
        ])

class DataPreprocessing():
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def __init__(self, tfr_filepath, image_size, batch_size, buffer_size):
        self.image_size = image_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.image_shape = (image_size, image_size, 3)
        
        # tfr파일 경로를 받아온다. 
        self.tfr_filepath = tfr_filepath
        
        # tfr파일을 불러온다.
        self.raw_image_dataset = tf.data.TFRecordDataset(self.tfr_filepath)
        
        # Create a dictionary describing the features.
        self.image_feature_description = {
                'image' : tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            }
    
    #아래 두 함수는 한 묶음으로 tfr파일의 정보를 feature dict에 맞게 변환, 할당한다.
    def _parse_image_function(self,example_proto):
            return tf.io.parse_single_example(example_proto, self.image_feature_description)
        
    def _parsed_image_dataset(self):
        paresd_img_data = self.raw_image_dataset.map(self._parse_image_function)
        return paresd_img_data
    
    def data_alb(self):
        temp = self._parsed_image_dataset()
        temp = temp.shuffle(self.buffer_size)
        ds_alb = temp.map(partial(self.process_data, image_size=self.image_size), num_parallel_calls=self.AUTOTUNE)
        ds_alb = ds_alb.map(partial(self.set_shapes, img_shape=self.image_shape), num_parallel_calls=self.AUTOTUNE)
        ds_alb = ds_alb.repeat()
        ds_alb = ds_alb.batch(self.batch_size)
        ds_alb = ds_alb.prefetch(self.AUTOTUNE)
        return ds_alb
    
    # Augmentation을 적용시키는 함수이다.
    def aug_fn(self, image, img_size):
        data = {"image":image}
        aug_data = transforms(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img/255.0, tf.float32)
        aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
        return aug_img

    # tfr에 담겨있는 img, label을 받아와서 img만 aug_fn함수에 전달한 후
    # aug_img, label을 다시 반환하여 데이터쌍을 유지한다.
    def process_data(self, data, image_size):
        image = data["image"]
        label = data["label"]
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        aug_img = tf.numpy_function(func=self.aug_fn, inp=[image, image_size], Tout=tf.float32)
        return aug_img, label
    
    # 최종적으로 데이터의 shape을 정의해준다.
    def set_shapes(self, img, label, img_shape):
        img.set_shape(img_shape)
        label = tf.one_hot(label, 30)
        return img, label
    
    def view_image(self, ds):
        image, label = next(iter(ds)) # extract 1 batch from the dataset
        image = image.numpy()
        label = label.numpy()

        fig = plt.figure(figsize=(22, 22))
        for i in range(20):
            ax = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])
            ax.imshow(image[i])
            ax.set_title(f"Label: {tf.argmax(label[i])}")
            
    def __call__(self):
        return self.data_alb()

#tfr_filepath = os.path.join(os.getenv("HOME"),"data/food/food_data.tfr")
#dp_new = DataPreprocessing(tfr_filepath,image_size=224, batch_size=32, buffer_size=300)

