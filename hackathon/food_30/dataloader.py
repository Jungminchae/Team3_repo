import tensorflow as tf 
from make_tfr import FoodTFrecord


class FoodDataLoader(FoodTFrecord):

    def __init__(self, batch_size, image_size, label_num ,train_valid_rate ,**kwargs):
        super(FoodDataLoader, self).__init__(**kwargs)
        self.image_size = image_size
        self.batch_size = batch_size
        self.label_num = label_num
        self.train_valid_rate = train_valid_rate

    def decode_img(self, data):
        image = data['image']
        label = data['label']

        image = tf.image.decode_image(image, channels=3, expand_animations = False)
        image = tf.image.resize(image, (self.image_size,self.image_size))

        label = tf.one_hot(label, self.label_num)
        return image, label

    def food_tf_dataset(self):
        size = len(self.image_data_path)

        train_size = int(self.train_valid_rate[0] * size)
        val_size = int(self.train_valid_rate[1] * size)

        dataset = self.read_tfr()
        dataset = dataset.shuffle(size)

        # train
        train_ds = dataset.take(train_size)
        train_ds = train_ds.map(self.decode_img)
        train_ds = train_ds.batch(self.batch_size)
        train_ds = train_ds.repeat()
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        # valid 
        valid_ds = dataset.skip(train_size)
        valid_ds = dataset.take(val_size)
        valid_ds = valid_ds.map(self.decode_img)
        valid_ds = valid_ds.batch(self.batch_size)

        return train_ds, valid_ds
        

class FoodDataLoader_with_only_TFRecord(FoodDataLoader):

    def __init__(self, tfr_path, image_size, label_num):
        self.raw_image_dataset = tf.data.TFRecordDataset(tfr_path)
        image_feature_description = {
            'image' : tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        def _parse_image_function(example_proto):
            return tf.io.parse_single_example(example_proto, image_feature_description)
        self.parsed_image_dataset = self.raw_image_dataset.map(_parse_image_function)
        self.image_size = image_size
        self.label_num = label_num

    def food_tf_dataset(self, train_valid_rate, size, batch_size):

        print('1231333',train_valid_rate[0])
        train_size = int(train_valid_rate[0] * size)
        val_size = int(train_valid_rate[1] * size)

        dataset = self.parsed_image_dataset
        dataset = dataset.shuffle(size)

        # train
        train_ds = dataset.take(train_size)
        train_ds = train_ds.map(self.decode_img)
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.repeat()
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        # valid 
        valid_ds = dataset.skip(train_size)
        valid_ds = dataset.take(val_size)
        valid_ds = valid_ds.map(self.decode_img)
        valid_ds = valid_ds.batch(batch_size)

        return train_ds, valid_ds

    def decode_img(self, data):
        image = data['image']
        label = data['label']

        image = tf.image.decode_image(image, channels=3, expand_animations = False)
        image = tf.image.resize(image, (self.image_size,self.image_size))

        label = tf.one_hot(label, self.label_num)
        return image, label
