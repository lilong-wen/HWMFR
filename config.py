'''
configs:
    train_datasets:
        offline-train.pkl: pickle file with name-image pairs
        train_caption: label file with name-label pairs
     dictionaries:
        lexicons in latex
     batch_Imagesize:
     valid_batch_Imagesize:
     ...
'''

train_datasets=['./data/official_data/train.pkl','./data/official_data/train_label.txt']
valid_datasets=['./data/official_data/test.pkl', './data/official_data/test_label.txt']
dictionaries=['./data/official_data/dictionary.txt']
batch_Imagesize=500000
valid_batch_Imagesize=500000
# batch_size for training and testing
batch_size=6
batch_size_test=6
# the max (label length/Image size) in training and testing
# you can change 'maxlen','maxImagesize' by the size of your GPU
maxlen=48
maxImagesize= 100000
# hidden_size in RNN
hidden_size = 256
# teacher_forcing_ratio
teacher_forcing_ratio = 1
# change the gpu id
gpu = [0,1]
# gpu = None
# learning rate
lr_rate = 0.0001
# flag to remember when to change the learning rate
flag = 0
# exprate
exprate = 0
# pre-trained encoder
pre_trained_pthfile = r'./models/densenet121-a639ec97.pth'
