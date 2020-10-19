import os
import sys
import shutil
import cv2
import pickle
import numpy as np

whole_data_path = "../data/off_image_test/"
whole_label_file = "../data/label.txt"

train_data_path = '../data/train/'
train_label_file = "../data/train_label.txt"
test_data_path = '../data/test/'
test_label_file = "../data/test_label.txt"

train_pkl_file = '../data/offline-train.pkl'
test_pkl_file = '../data/offline-test.pkl'

def split_and_write(data_path=whole_data_path,
                    label_file=whole_label_file,
                    train_data_path = train_data_path,
                    train_label_file = train_label_file,
                    test_data_path = test_data_path,
                    test_label_file = test_label_file):
    '''
    split a whole data file into train and valid
    args:
        data_path:
        label_path:
    return:
        null
    '''
    split_rate = 0.7
    if not  os.path.isfile(train_label_file) \
       and not os.path.isfile(test_label_file):
            with open(label_file, "r") as label_f:
                lines = label_f.readlines()
                raw_num = len(lines)

                train_number = int(raw_num * split_rate)
                train_lines = lines[0: train_number]
                valid_lines = lines[train_number: ]

                with open(train_label_file, "w") as train_label_f:
                    for line in train_lines:
                        train_label_f.write(line)

                with open(test_label_file, "w") as test_label_f:
                    for line in valid_lines:
                        test_label_f.write(line)
    else:
        print("label files exist")
    if not os.path.isdir (train_data_path):
        os.mkdir (train_data_path)
        with open(train_label_file, "r") as train_label_f:
            lines = train_label_f.readlines()
            for line in lines:
                file_key = line.strip ().split ("\t") [0]
                file_name = file_key + "_" + str (0) + ".bmp"
                shutil.copy2(data_path + file_name, train_data_path)
    else:
        print ("train image copied")

    if not os.path.isdir (test_data_path):
        os.mkdir (test_data_path)
        with open(test_label_file, "r") as test_label_f:
            lines = test_label_f.readlines()
            for line in lines:
                file_key = line.strip ().split ("\t") [0]
                file_name = file_key + "_" + str (0) + ".bmp"
                shutil.copy2(data_path + file_name, test_data_path)
    else:
        print ("test image copied")


def gen_pkl(image_path, label_file, pkl_file, channels=1):
    feature = {}
    with open (label_file, "r") as label_f:
        lines = label_f.readlines ()
        for line in lines:
            file_key = line.split ("\t") [0]
            file_name = image_path + file_key + "_" + str (0) + ".bmp"
            img = cv2.imread (file_name, 0)
            # img = img.transpose (2, 0, 1)
            img = np. expand_dims (img, axis=0)
            # print (img.shape)
            feature [file_key] = img
    print ("loading imags done")

    output_pkl = open (pkl_file, "wb")
    pickle.dump (feature, output_pkl)
    print ("generate pkl done")
    output_pkl.close ()


if __name__ == "__main__":
    # split_and_write()
    gen_pkl(train_data_path, train_label_file, train_pkl_file)
    gen_pkl(test_data_path, test_label_file, test_pkl_file)
