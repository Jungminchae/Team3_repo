class FoodDataPaths:
    image_data_path = ''
    food_dir_path = ''
    tfr_path = ''
    model_save_path = ''
    models_dir = ''
    test_image_path = ''


    @classmethod
    def make_img_food_path(cls, img_data_path, food_dir_path):
        cls.image_data_path = img_data_path
        cls.food_dir_path = food_dir_path
    
    @classmethod
    def make_tfr_path(cls, tfr_path):
        cls.tfr_path = tfr_path

    @classmethod
    def make_model_save_path(cls, model_save_path):
        cls.model_save_path = model_save_path
    
    @classmethod
    def make_models_dir(cls, models_dir):
        cls.models_dir = models_dir

    @classmethod
    def make_test_image_path(cls, test_image_path):
        cls.test_image_path = test_image_path