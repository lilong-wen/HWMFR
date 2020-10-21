import os
import sys
import shutil
import cv2
import pickle
import numpy as np
import json

data_path = "../data/official_data/train_data/"
train_label_file = "../data/official_data/train_label.txt"
test_label_file = "../data/official_data/test_label.txt"

train_pkl_file = "../data/official_data/train.pkl"
test_pkl_file = "../data/official_data/test.pkl"

def load_and_gen(data_path, label_file, pkl_file):

    feature = {}
    with open(label_file, 'r') as f:
        data_dict = json.load(f)
        print("loading %d items" % len(data_dict))
        for item in data_dict.items():
            img = cv2.imread(str(data_path + item[0]), 0)
            img = np.expand_dims(img, axis = 0)
            feature[item[0]] = img

    with open(pkl_file, "wb") as f_pkl:
        pickle.dump(feature, f_pkl)
        print("generate pkl finished")


if __name__ == "__main__":

    # gen_pkl(train_data_path, train_label_file, train_pkl_file)
    # gen_pkl(test_data_path, test_label_file, test_pkl_file)
    load_and_gen(data_path, test_label_file, test_pkl_file)
    load_and_gen(data_path, train_label_file, train_pkl_file)
