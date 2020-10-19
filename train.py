import torch

import numpy as np
np.set_printoptions(threshold=np.inf)

from utils.char_level_distance import edit_distance
from utils.load_dict import load_dict
from utils.data_loader import dataIterator
from utils.custom_dataset import custom_dataset
from utils.custom_collate_fn import custom_collate_fn
from utils.train_process import train_process

from models.densent_torchvision import densenet121
from models.attention_rnn import AttnDecoderRNN

from config import train_datasets, valid_datasets, dictionaries, \
    batch_Imagesize, valid_batch_Imagesize, batch_size, \
    batch_size_test, maxlen, maxImagesize, teacher_forcing_ratio,\
    gpu, lr_rate, flag, exprate, pre_trained_pthfile

cuda_available = True if torch.cuda.is_available() else False

word_dicts = load_dict(dictFile=dictionaries[0])

train,train_label = dataIterator(train_datasets[0],
                                 train_datasets[1],
                                 word_dicts,
                                 batch_size=1,
                                 batch_Imagesize=batch_Imagesize,
                                 maxlen=maxlen,
                                 maxImagesize=maxImagesize)

test,test_label = dataIterator(valid_datasets[0],
                               valid_datasets[1],
                               word_dicts,
                               batch_size=1,
                               batch_Imagesize=batch_Imagesize,
                               maxlen=maxlen,
                               maxImagesize=maxImagesize)

len_test = len(test)

image_train = custom_dataset(train,train_label,batch_size)
image_test = custom_dataset(test,test_label,batch_size)

train_loader = torch.utils.data.DataLoader(
    dataset = image_train,
    batch_size = batch_size,
    shuffle = True,
    collate_fn = custom_collate_fn,
    num_workers=2,
    )
test_loader = torch.utils.data.DataLoader(
    dataset = image_test,
    batch_size = batch_size_test,
    shuffle = True,
    collate_fn = custom_collate_fn,
    num_workers=2,
)

encoder = densenet121()
pretrained_dict = torch.load(pre_trained_pthfile)
encoder_dict = encoder.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
encoder_dict.update(pretrained_dict)
encoder.load_state_dict(encoder_dict)

attn_decoder1 = AttnDecoderRNN(hidden_size,112,dropout_p=0.5)
decoder_input_init = torch.LongTensor([111]*batch_size)
decoder_hidden_init = torch.randn(batch_size, 1, hidden_size)

if cuda_available:
    encoder=encoder.cuda()
    attn_decoder1 = attn_decoder1.cuda()
    encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
    attn_decoder1 = torch.nn.DataParallel(attn_decoder1, device_ids=gpu)
    decoder_input_init = decoder_input_init.cuda()
    decoder_hidden_init = decoder_hidden_init.cuda()



nn.init.xavier_uniform_(decoder_hidden_init)
criterion = nn.NLLLoss()

for epoch in range(200):
    encoder_optimizer1 = torch.optim.SGD(encoder.parameters(), lr=lr_rate,momentum=0.9)
    decoder_optimizer1 = torch.optim.SGD(attn_decoder1.parameters(), lr=lr_rate,momentum=0.9)

    running_loss=0
    whole_loss = 0

    encoder.train(mode=True)
    attn_decoder1.train(mode=True)

    for step,(x,y) in enumerate(train_loader):
        if x.size()[0]<batch_size:
            break
        h_mask = []
        w_mask = []
        for i in x:
            #h*w
            size_mask = i[1].size()
            s_w = str(i[1][0])
            s_h = str(i[1][:,1])
            w = s_w.count('1')
            h = s_h.count('1')
            h_comp = int(h/16)+1
            w_comp = int(w/16)+1
            h_mask.append(h_comp)
            w_mask.append(w_comp)

        if cuda_available:
            x = x.cuda()
            y = y.cuda()
        # out is CNN featuremaps
        output_highfeature = encoder(x)
        x_mean=[]
        for i in output_highfeature:
            x_mean.append(float(torch.mean(i)))
        # x_mean = torch.mean(output_highfeature)
        # x_mean = float(x_mean)
        for i in range(batch_size):
            decoder_hidden_init[i] = decoder_hidden_init[i]*x_mean[i]
            decoder_hidden_init[i] = torch.tanh(decoder_hidden_init[i])

        # dense_input is height and output_area is width which is bb
        output_area1 = output_highfeature.size()

        output_area = output_area1[3]
        dense_input = output_area1[2]
        target_length = y.size()[1]

        attention_sum_init = torch.zeros(batch_size,
                                         1,
                                         dense_input,
                                         output_area)
        decoder_attention_init = torch.zeros(batch_size,
                                             1,
                                             dense_input,
                                             output_area)
        if cuda_available:
            attention_sum_init = attention_sum_init.cuda()
            decoder_attention_init = decoder_attention_init.cuda()

        running_loss += train_process(target_length,
                                      attn_decoder1,
                                      output_highfeature,
                                      output_area,
                                      y,
                                      criterion,
                                      encoder_optimizer1,
                                      decoder_optimizer1,
                                      x_mean,
                                      dense_input,
                                      h_mask,
                                      w_mask,
                                      gpu,
                                      decoder_input_init,
                                      decoder_hidden_init,
                                      attention_sum_init,
                                      decoder_attention_init)


        if step % 20 == 19:
            pre = ((step+1)/len_train)*100*batch_size
            whole_loss += running_loss
            running_loss = running_loss/(batch_size*20)
            print('epoch is %d, lr rate is %.5f, te is %.3f, batch_size is %d, loading for %.3f%%, running_loss is %f' %(epoch,lr_rate,teacher_forcing_ratio, batch_size,pre,running_loss))
            # with open("training_data/running_loss_%.5f_pre_GN_te05_d02_all.txt" %(lr_rate),"a") as f:
            #     f.write("%s\n"%(str(running_loss)))
            running_loss = 0

    loss_all_out = whole_loss / len_train
    print("epoch is %d, the whole loss is %f" % (epoch, loss_all_out))
    # with open("training_data/whole_loss_%.5f_pre_GN_te05_d02_all.txt" % (lr_rate), "a") as f:
    #     f.write("%s\n" % (str(loss_all_out)))

    # this is the prediction and compute wer loss
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    whole_loss_t = 0

    encoder.eval()
    attn_decoder1.eval()
    print('Now, begin testing!!')

    for step_t, (x_t, y_t) in enumerate(test_loader):
        x_real_high = x_t.size()[2]
        x_real_width = x_t.size()[3]
        if x_t.size()[0]<batch_size_t:
            break
        print('testing for %.3f%%'%(step_t*100*batch_size_t/len_test),end='\r')
        h_mask_t = []
        w_mask_t = []
        for i in x_t:
            #h*w
            size_mask_t = i[1].size()
            s_w_t = str(i[1][0])
            s_h_t = str(i[1][:,1])
            w_t = s_w_t.count('1')
            h_t = s_h_t.count('1')
            h_comp_t = int(h_t/16)+1
            w_comp_t = int(w_t/16)+1
            h_mask_t.append(h_comp_t)
            w_mask_t.append(w_comp_t)

        x_t = x_t.cuda()
        y_t = y_t.cuda()
        output_highfeature_t = encoder(x_t)

        x_mean_t = torch.mean(output_highfeature_t)
        x_mean_t = float(x_mean_t)
        output_area_t1 = output_highfeature_t.size()
        output_area_t = output_area_t1[3]
        dense_input = output_area_t1[2]

        decoder_input_t = torch.LongTensor([111]*batch_size_t)
        decoder_input_t = decoder_input_t.cuda()
        decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).cuda()
        nn.init.xavier_uniform_(decoder_hidden_t)

        x_mean_t=[]
        for i in output_highfeature_t:
            x_mean_t.append(float(torch.mean(i)))
        # x_mean = torch.mean(output_highfeature)
        # x_mean = float(x_mean)
        for i in range(batch_size_t):
            decoder_hidden_t[i] = decoder_hidden_t[i]*x_mean_t[i]
            decoder_hidden_t[i] = torch.tanh(decoder_hidden_t[i])

        prediction = torch.zeros(batch_size_t,maxlen)
        #label = torch.zeros(batch_size_t,maxlen)
        prediction_sub = []
        label_sub = []
        decoder_attention_t = torch.zeros(batch_size_t,
                                          1,
                                          dense_input,
                                          output_area_t).cuda()
        attention_sum_t = torch.zeros(batch_size_t,
                                      1,
                                      dense_input,
                                      output_area_t).cuda()
        flag_z_t = [0]*batch_size_t
        loss_t = 0
        m = torch.nn.ZeroPad2d((0,maxlen-y_t.size()[1],0,0))
        y_t = m(y_t)
        for i in range(maxlen):
            decoder_output, \
                decoder_hidden_t, \
                decoder_attention_t, \
                attention_sum_t = attn_decoder1(decoder_input_t,
                                                decoder_hidden_t,
                                                output_highfeature_t,
                                                output_area_t,
                                                attention_sum_t,
                                                decoder_attention_t,
                                                dense_input,
                                                batch_size_t,
                                                h_mask_t,
                                                w_mask_t,
                                                gpu)

            ### you can see the attention when testing

            # print('this is',i)
            # for i in range(batch_size_t):
            #     x_real = numpy.array(x_t[i][0].data.cpu())

            #     show = numpy.array(decoder_attention_t[i][0].data.cpu())
            #     show = imresize(show,(x_real_width,x_real_high))
            #     k_max = show.max()
            #     show = show/k_max

            #     show_x = x_real+show
            #     plt.imshow(show_x, interpolation='nearest', cmap='gray_r')
            #     plt.show()

            topv,topi = torch.max(decoder_output,2)
            # if torch.sum(y_t[0,:,i])==0:
            #     y_t = y_t.squeeze(0)
            #     break
            if torch.sum(topi)==0:
                break
            decoder_input_t = topi
            decoder_input_t = decoder_input_t.view(batch_size_t)

            # prediction
            prediction[:,i] = decoder_input_t

        for i in range(batch_size_t):
            for j in range(maxlen):
                if int(prediction[i][j]) ==0:
                    break
                else:
                    prediction_sub.append(int(prediction[i][j]))
            if len(prediction_sub)<maxlen:
                prediction_sub.append(0)

            for k in range(y_t.size()[1]):
                if int(y_t[i][k]) ==0:
                    break
                else:
                    label_sub.append(int(y_t[i][k]))
            label_sub.append(0)

            dist, llen = cmp_result(label_sub, prediction_sub)
            total_dist += dist
            total_label += llen
            total_line += 1
            if dist == 0:
                total_line_rec = total_line_rec+ 1

            label_sub = []
            prediction_sub = []

    print('total_line_rec is',total_line_rec)
    wer = float(total_dist) / total_label
    sacc = float(total_line_rec) / total_line
    print('wer is %.5f' % (wer))
    print('sacc is %.5f ' % (sacc))
    # print('whole loss is %.5f'%(whole_loss_t/925))
    # with open("training_data/wer_%.5f_pre_GN_te05_d02_all.txt" % (lr_rate), "a") as f:
    #     f.write("%s\n" % (str(wer)))

    if (sacc > exprate):
        exprate = sacc
        print(exprate)
        print("saving the model....")
        print('encoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl' %(lr_rate))
        torch.save(encoder.state_dict(), 'model/encoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'%(lr_rate))
        torch.save(attn_decoder1.state_dict(), 'model/attn_decoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'%(lr_rate))
        print("done")
        flag = 0
    else:
        flag = flag+1
        print('the best is %f' % (exprate))
        print('the loss is bigger than before,so do not save the model')

    if flag == 10:
        lr_rate = lr_rate*0.1
        flag = 0
