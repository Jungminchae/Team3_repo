import argparse
import tensorflow as tf
from models import Modelselect
from dataloader import FoodDataLoader


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--food_dir_path", type=str)
    parser.add_argument("--tfr_path", type=str)
    parser.add_argument("--image_data_path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--label_num", type=int)
    parser.add_argument("--train_valid_rate")