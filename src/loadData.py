import numpy as np
import pandas as pd
import cv2
from sklearn import preprocessing


def load_images(data_frame, column_name):
    filelist = data_frame[column_name].to_list()
    return filelist


def load_labels(data_frame, column_name):
    label_list = data_frame[column_name].to_list()
    return label_list


def preprocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (100, 100))

    image_reshaped_items = [np.concatenate(item.reshape((1, len(item) * len(item[0]))), axis=0) for item in image]
    image_reshaped = np.concatenate(np.array(image_reshaped_items).reshape(1, len(image_reshaped_items) * len(image_reshaped_items[0])), axis=0)
    norm_image_reshaped = np.concatenate(preprocessing.normalize([image_reshaped]), axis=0)

    return norm_image_reshaped


def load_data(data_path):
    df = pd.read_csv(data_path)
    labels = load_labels(data_frame=df, column_name="label")
    images_paths = load_images(data_frame=df, column_name="filename")
    processed_images = [preprocess(image_path) for image_path in images_paths]

    return processed_images, labels
