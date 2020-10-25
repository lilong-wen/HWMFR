from for_test_V20 import for_test
import cv2
import torch
import os
import json

val_data_root = "../data/official_data/valdata/"
val_json = "../data/official_data/val.json"

test_data_root = "../data/official_data/testdata/"
test_json = "../data/official_data/test_true.json"

def widder(im):

    row, col = im.shape[:2]
    bottom = im[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]
    bordersize = 5
    no = 0
    border = cv2.copyMakeBorder(
            im,
            top=bordersize,
            bottom=bordersize,
            left=no,
            right=no,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean]
    )

    return border

def trans1(data_root, json_file):
    data = []
    for item in os.listdir(data_root):
        val_dict = {}
        img_open = cv2.imread(data_root + item, 0)
        print(item)
        if img_open.shape[0] < 18:
            img_open = widder(img_open)
        #img_open = cv2.imread("./train_4.jpg", 0)
        img_open2 = torch.from_numpy(img_open).type(torch.FloatTensor)
        img_open2 = img_open2/255.0
        img_open2 = img_open2.unsqueeze(0)
        img_open2 = img_open2.unsqueeze(0)

        attention, prediction = for_test(img_open2)
        prediction_string = ''

        for i in range(attention.shape[0]):
            if prediction[i] == '<eol>':
                continue
            else:
                prediction_string = prediction_string + prediction[i] + " "
                # prediction_string = prediction_string + prediction[i]
        #print(prediction_string)
        val_dict['filename'] = str(item)
        val_dict['result'] = str(prediction_string)
        data.append(val_dict)
    with open(json_file, 'w') as f_val:
        json.dump(data, f_val)

if __name__ == "__main__":
    trans1(test_data_root, test_json)
