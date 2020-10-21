import torch
import torch.utils.data as data
import numpy

class custom_dataset(data.Dataset):
    def __init__(self,train,train_label,batch_size):
        self.train = train
        self.train_label = train_label
        self.batch_size = batch_size

    def __getitem__(self, index):
        train_setting = torch.from_numpy(numpy.array(self.train[index]))
        label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(torch.LongTensor)

        size = train_setting.size()
        train_setting = train_setting.view(1,size[2],size[3])
        label_setting = label_setting.view(-1)

        #print(train_setting.shape)
        return train_setting,label_setting

    def __len__(self):
        return len(self.train)
