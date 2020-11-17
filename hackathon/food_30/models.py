import tensorflow as tf

class Modelselect:

    '''
    EfficientNetB0 => eb0
    EfficientNetB1 => eb1
    ...

    MobileNetv1 => mv1
    MobileNetv2 => mv2
    Xception => x
    NASNetmobile => nasm
    NASNetlarge => nasl
    Densenet => dn
    '''


    def __init__(self, model_name, image_size, class_num):
        self.model_name = model_name
        self.image_size = image_size 
        self.class_num = class_num


    def model(self):
        model_parameters = {
            'input_shape': (self.image_size, self.image_size, 3),
            'include_top': False,
            'weights': 'imagenet',
        }
        if self.model_name =='eb0':
            base_model = tf.keras.applications.EfficientNetB0(**model_parameters)
        elif self.model_name =='eb1':
            base_model = tf.keras.applications.EfficientNetB1(**model_parameters)
        elif self.model_name =='eb2':
            base_model = tf.keras.applications.EfficientNetB2(**model_parameters)
        elif self.model_name =='eb3':
            base_model = tf.keras.applications.EfficientNetB3(**model_parameters)
        elif self.model_name =='eb4':
            base_model = tf.keras.applications.EfficientNetB4(**model_parameters)
        elif self.model_name =='eb5':
            base_model = tf.keras.applications.EfficientNetB5(**model_parameters)
        elif self.model_name =='eb6':
            base_model = tf.keras.applications.EfficientNetB6(**model_parameters)
        elif self.model_name =='eb7':
            base_model = tf.keras.applications.EfficientNetB7(**model_parameters)
        elif self.model_name =='mv1':
            base_model = tf.keras.applications.MobileNet(**model_parameters)
        elif self.model_name =='mv2':
            base_model = tf.keras.applications.MobileNetV2(**model_parameters)
        elif self.model_name =='x':
            base_model = tf.keras.applications.Xception(**model_parameters)
        elif self.model_name =='nasm':
            base_model = tf.keras.applications.NASNetMobile(**model_parameters)
        elif self.model_name =='nasl':
            base_model = tf.keras.applications.NASNetLarge(**model_parameters)
        elif self.model_name =='d121':
            base_model = tf.keras.applications.DenseNet121(**model_parameters)

        flatten_layer = tf.keras.layers.GlobalAveragePooling2D()
        dense_layer = tf.keras.layers.Dense(512, activation='relu')
        prediction_layer = tf.keras.layers.Dense(self.class_num)

        model = tf.keras.Sequential([
            base_model,
            flatten_layer,
            dense_layer,
            prediction_layer
        ])

        print(model.summary())
        return model