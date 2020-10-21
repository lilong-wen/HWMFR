import os
import sys
import json

data_path = "../data/official_data/train_data/"
label_file = "../data/official_data/train.txt"
sample_label_file = "../data/official_data/label_sample.txt"

split_ratio = 0.8
line_number = 113510

train_label_file = "../data/official_data/train_label.txt"
test_label_file = "../data/official_data/test_label.txt"
train_json = "../data/official_data/train.json"

# {"ImageFile": "train_0.jpg", "Label": "( 2 ) 2 N a O H + C u S O _ { 4 } = N a _ { 2 } S O _ { 4 } + C u ( O H ) _ { 2 } \\downarrow"}

def split():

    train_num = int(split_ratio * line_number)
    test_num = line_number - train_num
    #label_data = json.loads(json.dumps({"ImageFile": "train_0.jpg", "Label": "( 2 ) 2 N a O H + C u S O _ { 4 } = N a _ { 2 } S O _ { 4 } + C u ( O H ) _ { 2 } \\downarrow"}))
    #print(label_data['ImageFile'])

    label_data = {}
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            line_dict = json.loads(line)
            # print(line_dict['ImageFile'])
            label_data[line_dict['ImageFile']] = line_dict['Label']

        print("total %d items" % len(label_data))

    print("saving reformated whole label file")
    with open(train_json, 'w') as f_label:
        json.dump(label_data, f_label)

    print("split begin here")
    train_label_data = dict(list(label_data.items())[: train_num])
    test_label_data = dict(list(label_data.items())[train_num :])

    print("training item: %d " % len(train_label_data))
    print("testing item %d " % len(test_label_data))

    with  open(train_label_file, 'w') as f_train:
        json.dump(train_label_data, f_train)
    with open(test_label_file, 'w') as f_test:
        json.dump(test_label_data, f_test)

    print("split finished")

if __name__ == "__main__":
    split()
