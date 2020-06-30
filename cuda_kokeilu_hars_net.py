import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from sys import exit as kill
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
# from torchvision import utils
import time
import os

print('tarkista nuo saatanan backends')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()

koti = False


if koti:
    my_add = r'C:\Users\jpkorpel\PycharmProjects\uusi_sika\harjoitus_verkko'
    my_path = os.path.join(my_add, 'print_folder')
    print()
    X_data = np.load(r'C:\Users\jpkorpel\Desktop\hammas\andy\X_andy.npy')
    X_data = X_data[70:72]
    Y_data = np.load(r'C:\Users\jpkorpel\Desktop\hammas\andy\Y_andy.npy')
    Y_data = Y_data[70:72]
    card = 'cpu'
else:
    my_parent_dir = os.path.dirname(os.getcwd())
    my_data_folder = os.path.join(my_parent_dir, 'tooth_data')
    my_path = os.path.join(os.getcwd(), 'print_folder')
    X_data = np.load(os.path.join(my_data_folder, 'X_data_joint.npy'))
    Y_data = np.load(os.path.join(my_data_folder, 'Y_data_joint.npy'))
    card = 'cuda'


print('X_shape = ', X_data.shape)
print('X_dtype = ', X_data.dtype)

print('Y_shape = ', Y_data.shape)
print('Y_dtype = ', Y_data.dtype)
print('datat saatana ladattu.... tarkista vielä miksi Y ja X eri tyyppejä kun teet niitä lisää... oi miksi????')


norm_data = True
scale_data = False
if norm_data:
    print('mulkku norm')
    mean, std = np.mean(X_data), np.std(X_data)
    X_data = X_data - mean
    X_data = X_data / std

if scale_data:
    print('mulkku scale')
    maxim, minim = np.max(X_data), np.min(X_data)
    X_data = X_data + minim
    X_data = X_data / (maxim - minim)

# print(Y_data.shape)
X_data = X_data.astype(np.float32)
X_torch = torch.from_numpy(X_data)
Y_torch = torch.from_numpy(Y_data)
X_torch = X_torch.unsqueeze(1)

Y_torch = Y_torch.unsqueeze(1)

print('Y_torch_shape = ', Y_torch.shape)
print('X_torch_shape = ', X_torch.shape)

print('Y_torch_dtype = ', Y_torch.dtype)
print('X_torch_dtype = ', X_torch.dtype)
print('tyypit ja muodot oikein ennen dataloaderia!!!!!')


# Dataloader
batch_size = 1
train_dataset = TensorDataset(X_torch, Y_torch)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_block, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        skip = x  # save x to use in concat in the Decoder path
        out = self.maxPool(x)
        return out, skip


class Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size, padding, drop_out=False):
        super(Decoder_block, self).__init__()

        self.drop_out = drop_out
        self.dropout_layer = nn.Dropout2d(p=0.5)

        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding)
        self.upSample = nn.ConvTranspose2d(in_channels, out_channels, upsample_size, stride=2)

    def _crop_concat(self, upsampled, downsampled):
        """
         pad upsampled to the (h, w) of downsampled to concatenate
         for expansive path.
        Returns:
            The concatenated tensor
        """
        h = downsampled.size()[2] - upsampled.size()[2]
        w = downsampled.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, (0, w, 0, h))
        return torch.cat((downsampled, upsampled), 1)

    def forward(self, x, down_tensor):
        x = self.upSample(x)
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        if self.drop_out: x = self.dropout_layer(x)
        x = self.convr2(x)
        if self.drop_out: x = self.dropout_layer(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = Encoder_block(1, 64)
        self.down2 = Encoder_block(64, 128)
        self.down3 = Encoder_block(128, 256)
        self.down4 = Encoder_block(256, 512)

        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=1),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=1)
        )

        self.up1 = Decoder_block(in_channels=1024, out_channels=512, upsample_size=2, padding=1)
        self.up2 = Decoder_block(in_channels=512, out_channels=256, upsample_size=2, padding=1)
        self.up3 = Decoder_block(in_channels=256, out_channels=128, upsample_size=2, padding=1)
        self.up4 = Decoder_block(in_channels=128, out_channels=64, upsample_size=2, padding=1)

        # 1x1 convolution at the last layer
        self.outputNN = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)

        # self._initialize_weights()

    def forward(self, x):
        # x.cuda(device)
        x, skip1 = self.down1(x)
        # print(x.shape)
        x, skip2 = self.down2(x)
        # print(x.shape)
        x, skip3 = self.down3(x)
        # print(x.shape)
        x, skip4 = self.down4(x)
        # print(x.shape)
        x = self.center(x)
        # print(x.shape)
        x = self.up1(x, skip4)
        # print(x.shape)
        x = self.up2(x, skip3)
        # print(x.shape)
        x = self.up3(x, skip2)
        # print(x.shape)
        x = self.up4(x, skip1)
        # print(x.shape)
        x = self.outputNN(x)
        # print(x.shape)
        x = torch.sigmoid(x)
        return x

# print(network)
# summary(network, input_size=(1, 434, 434))

# loss_BCE = torch.nn.BCEWithLogitsLoss()  # reduction='none'


@torch.no_grad()
def l1_loss(a, b):
    a = a.squeeze(0).squeeze(0)    # chances to 434x434
    b = b.squeeze(0).squeeze(0)    # chances to 434x434
    my_help = abs(a - b)
    my_help = sum(sum(my_help)).item()    # calculates sums of absolute values from subtraction of matrix
    return my_help


d_batch_loss = {}
d_batch_l1 = {}
d_epoch_loss = {}
d_epoch_l1 = {}
d_epoch_max_loss = {}
d_epoch_max_l1 = {}
learning_rates = [0.0005, 0.001] #, 0.005, 0.01]

for lr in learning_rates:

    device = torch.device(card)
    network = UNet()
    network.to(device=device)

    loss_BCE = torch.nn.BCELoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)

    epoch_times = 10   # for future

    sika = True
    print('epoch_times', epoch_times)
    all_batch_loss = []
    all_sum_of_wrong = []
    epoch_loss = []
    epoch_l1 = []
    epoch_max_loss = []
    epoch_max_l1 = []
    for epoch in range(1, epoch_times + 1):
        batch_loss = []
        sum_of_wrong = []
        for batch_id, batch in enumerate(train_loader):  # get batch
            start_time = time.time()
            images, labels = batch
            images = images.to(device)
            #print(images.dtype)
            labels = labels.type(torch.FloatTensor)
            #print(labels.dtype)
            labels = labels.to(device)
            #print(labels.dtype)

            preds = network(images)  # Pass Batch   (shape = batch, 1, 434, 434)
            #print(preds.shape)
            #print(preds.requires_grad)

            # all_preds = torch.cat((all_preds, preds), dim=0)  # store all preds
            loss = loss_BCE(preds, labels)  # .. tensor(0.6402, device='cuda:0', grad_fn=<BinaryCrossEntropyBackward>)
            #print(loss)

            all_batch_loss.append(loss.item())
            batch_loss.append(loss.item())
            #print(batch_loss)

            my_help = l1_loss(preds, labels)
            print('sum of wrong pixels:', my_help)
            all_sum_of_wrong.append(my_help)
            sum_of_wrong.append(my_help)
            optimizer.zero_grad()
            loss.backward()  # Calculate Gradients
            optimizer.step()  # Update Weights

            print('loss:', loss.item())
            print('epoch:' + str(epoch) + ', bach_id: ' + str(batch_id + 1) + ' of ' + str(
                len(train_loader)) + ' takes time  ' + str(
                (time.time() - start_time)))
            del loss
            del preds
            torch.cuda.empty_cache()

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        epoch_l1.append(sum(sum_of_wrong)/len(sum_of_wrong))
        epoch_max_loss.append(max(batch_loss))
        epoch_max_l1.append(max(sum_of_wrong))
    d_batch_l1[lr] = all_sum_of_wrong
    d_batch_loss[lr] = all_batch_loss
    d_epoch_loss[lr] = epoch_loss
    d_epoch_l1[lr] = epoch_l1
    d_epoch_max_loss[lr] = epoch_max_loss
    d_epoch_max_l1[lr] = epoch_max_l1


string1 = 'Sum of Abolute errors of pixels in slices'
plt.figure(1)
plt.title(string1)
plt.xlabel('slice number')
plt.ylabel('L1-error')
for lr in learning_rates:
    plt.plot(range(1, len(d_batch_l1[lr]) + 1), d_batch_l1[lr], label='lr=' + str(lr))
plt.legend(loc="upper right")

my_file = 'Abolute_errors_with_different_lr.png'
plt.savefig(os.path.join(my_path, my_file))
plt.close()




string2 = 'BCE loss for all slices'
plt.figure(2)
plt.title(string2)
plt.xlabel('slice number')
plt.ylabel('torch.nn.BCELoss()')
for lr in learning_rates:
    plt.plot(range(1, len(d_batch_loss[lr]) + 1), d_batch_loss[lr], label='lr=' + str(lr))
plt.legend(loc="upper right")

my_file = 'BCEloss_errors_with_different_lr.png'
plt.savefig(os.path.join(my_path, my_file))
plt.close()

fig = plt.figure(3)
j = 0
for lr in learning_rates:
    j += 1
    string3 = 'lr=' + str(lr)
    plt.subplot(2, 2, j)
    plt.title(string3)
    #plt.xlabel('slice number')
    #plt.ylabel('torch.nn.BCELoss()')
    plt.plot(range(1, len(d_batch_loss[lr]) + 1), d_batch_loss[lr])
    plt.ylim(0, 1)
    plt.subplots_adjust(left=0.15,
                    bottom=0.15,
                    right=0.85,
                    top=0.85,
                    wspace=0.3,
                    hspace=0.35)


fig.suptitle('BCEloss_errors_with_different_lr wrt slice-number', fontsize=14)
my_file = 'BCEloss_errors_with_different_lr_sub_plots.png'
plt.savefig(os.path.join(my_path, my_file))
plt.close()


string4 = 'Average abolute error to slices in epoch'
plt.figure(4)
plt.title(string4)
plt.xlabel('Epoch number')
plt.ylabel('Average L1-error')
for lr in learning_rates:
    plt.plot(range(1, len(d_epoch_l1[lr]) + 1), d_epoch_l1[lr], label='lr=' + str(lr))
plt.legend(loc="upper right")

my_file = 'Average_Epoch_Abolute_errors_with_different_lr.png'
plt.savefig(os.path.join(my_path, my_file))
plt.close()

string5 = 'Average Bce_loss_results to slices in epoch'
plt.figure(5)
plt.title(string5)
plt.xlabel('Epoch number')
plt.ylabel('Average torch.nn.BCELoss()')
for lr in learning_rates:
    plt.plot(range(1, len(d_epoch_loss[lr]) + 1), d_epoch_loss[lr], label='lr=' + str(lr))
plt.legend(loc="upper right")

my_file = 'Average_epoch_BCEloss_errors_with_different_lr.png'
plt.savefig(os.path.join(my_path, my_file))
plt.close()


string6 = 'Max abolute error to slice in epoch'
plt.figure(6)
plt.title(string6)
plt.xlabel('Epoch number')
plt.ylabel('Max L1-error')
for lr in learning_rates:
    plt.plot(range(1, len(d_epoch_max_l1[lr]) + 1), d_epoch_max_l1[lr], label='lr=' + str(lr))
plt.legend(loc="upper right")

my_file = 'Max_Epoch_Abolute_errors_with_different_lr.png'
plt.savefig(os.path.join(my_path, my_file))
plt.close()

string7 = 'Max Bce_loss to slice in epoch'
plt.figure(7)
plt.title(string7)
plt.xlabel('Epoch number')
plt.ylabel('Max torch.nn.BCELoss()')
for lr in learning_rates:
    plt.plot(range(1, len(d_epoch_max_loss[lr]) + 1), d_epoch_max_loss[lr], label='lr=' + str(lr))
plt.legend(loc="upper right")

my_file = 'Max_epoch_BCEloss_errors_with_different_lr.png'
plt.savefig(os.path.join(my_path, my_file))
plt.close()



