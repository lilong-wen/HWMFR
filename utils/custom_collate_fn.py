import torch

# Padding images to the max image size in a mini-batch and cat a mask.
def custom_collate_fn(batch):
    '''
    print("here print origin batch size")
    print(len(batch))
    print(len(batch[0]))
    print(batch[0][0].shape)
    '''
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    aa1 = 0
    bb1 = 0
    k = 0
    k1 = 0
    max_len = len(label[0])+1
    for j in range(len(img)):
        size = img[j].size()
        if size[1] > aa1:
            aa1 = size[1]
        if size[2] > bb1:
            bb1 = size[2]

    for ii in img:
        ii = ii.float()
        img_size_h = ii.size()[1]
        img_size_w = ii.size()[2]
        img_mask_sub_s = torch.ones(1,img_size_h,img_size_w).type(torch.FloatTensor)
        img_mask_sub_s = img_mask_sub_s*255.0
        img_mask_sub = torch.cat((ii,img_mask_sub_s),dim=0)
        padding_h = aa1-img_size_h
        padding_w = bb1-img_size_w
        m = torch.nn.ZeroPad2d((0,padding_w,0,padding_h))
        img_mask_sub_padding = m(img_mask_sub)
        img_mask_sub_padding = img_mask_sub_padding.unsqueeze(0)
        if k==0:
            img_padding_mask = img_mask_sub_padding
        else:
            img_padding_mask = torch.cat((img_padding_mask,img_mask_sub_padding),dim=0)
        k = k+1

    for ii1 in label:
        ii1 = ii1.long()
        ii1 = ii1.unsqueeze(0)
        ii1_len = ii1.size()[1]
        m = torch.nn.ZeroPad2d((0,max_len-ii1_len,0,0))
        ii1_padding = m(ii1)
        if k1 == 0:
            label_padding = ii1_padding
        else:
            label_padding = torch.cat((label_padding,ii1_padding),dim=0)
        k1 = k1+1

    img_padding_mask = img_padding_mask/255.0
    '''
    print("here print returned padded image shape")
    print(img_padding_mask.shape)
    print("here print returned padded label shape")
    print(label_padding.shape)
    '''
    return img_padding_mask, label_padding
