import argparse
import tensorflow as tf
from models import Modelselect
from dataloader import FoodDataLoader, FoodDataLoader_with_only_TFRecord
from make_tfr import FoodTFrecord

def to_bool(x):
    if x.lower() in ['true','t']:
        return True
    elif x.lower() in ['false','f']:
        return False
    else:
        raise argparse.ArgumentTypeError('Bool 값을 넣으세요')


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['tfr', 'train', 'test'], help="TFRecord 만들기 or 모델 학습 or 모델 테스트")
    parser.add_argument("--food_dir_path", type=str, default='./', help="각 음식들의 폴더가 저장되어 있는 상위 폴더")
    parser.add_argument("--model_name", type=str, choices=["eb0","eb1","eb2","eb3","eb4","eb5","eb6","eb7","mv1","mv2","x","nasm","nasl"],default="eb0")
    parser.add_argument("--model_save_dir", type=str, default='./')
    parser.add_argument("--tfr_path", type=str, default='./')
    parser.add_argument("--only_tfr", type=to_bool, default='true')
    parser.add_argument("--tfr_size", type=int)
    parser.add_argument("--image_data_path", type=str, default='./')
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--label_num", type=int)
    parser.add_argument("--train_valid_rate" ,nargs='+')
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    args = parser.parse_args()

    # parameters 
    mc_dir_path = args.models_save_dir
    if mc_dir_path == './':
        raise NotADirectoryError("model 폴더에 저장하세요")
    tfr_path = args.tfr_path
    image_data_path = args.image_data_path
    food_dir_path = args.food_dir_path

    batch_size = args.batch_size
    image_size = args.image_size
    label_num = args.label_num
    epochs = args.epochs
    train_valid_rate = args.train_valid_rate

    # tfr 만들기
    if args.mode =='tfr':
        tfr_make = FoodTFrecord(tfr_path,image_data_path,food_dir_path)
        tfr_make.make_tfr()
    
    # model 학습
    if args.mode == "train":
        tfr_params = {
            'tft_path' : tfr_path, 'image_data_path': image_data_path, 'food_dir_path':food_dir_path
            }

        if args.only_tfr is False:
            dataloader = FoodDataLoader(batch_size,image_size,label_num,train_valid_rate, **tfr_params)
            train, valid = dataloader.food_tf_dataset()
            size = len(dataloader.image_data_path)

        elif args.only_tfr is True:
            dataloader = FoodDataLoader_with_only_TFRecord(tfr_path, image_size, label_num)
            size = args.tfr_size
            train, valid = dataloader.food_tf_dataset(train_valid_rate, size=size, batch_size=batch_size)

        model = Modelselect(model_name=args.model_name, image_size=image_size, class_num=label_num)
        model = model.model()

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience)
        mc = tf.keras.callbacks.ModelCheckpoint(
            filepath=mc_dir_path+'{epoch}-{val_loss:.2f}-{val_accuracy:.2f}.h5',
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        history = model.fit(
            train,
            epochs=args.epochs,
            validation_data=valid,
            steps_per_epoch=size / batch_size,
            callbacks=[es, mc],
            batch_size=batch_size
        )