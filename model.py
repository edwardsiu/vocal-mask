import torch
from torch import nn
import torch.nn.functional as F
from layers import *
from hparams import hparams as hp
from audio import *
from torch.utils.data import DataLoader, Dataset
from utils import num_params

from tqdm import tqdm
import numpy as np

class MobileNet(nn.Module):
    def __init__(self, indims, outdims):
        super().__init__()
        # 513x25
        self.init_conv = Conv3x3(3, 32, F.relu, stride=(2,1))
        # 257x25
        self.conv1 = DepthwiseConv(32, 64, stride=(2,1))
        # 129x25
        self.conv2 = DepthwiseConv(64, 128, stride=(2,2))
        # 65x13
        self.conv3 = DepthwiseConv(128, 128)
        self.conv4 = DepthwiseConv(128, 256, stride=(2,1))
        # 33x13
        self.conv5 = DepthwiseConv(256, 256)
        self.conv6 = DepthwiseConv(256, 512, stride=(2,2))
        # 17x7
        self.conv7 = nn.Sequential(
            DepthwiseConv(512, 512),
            DepthwiseConv(512, 512),
            DepthwiseConv(512, 512),
            DepthwiseConv(512, 512),
            DepthwiseConv(512, 512)
        )
        self.conv8 = DepthwiseConv(512, 1024, stride=(2,2))
        # 9x4
        self.conv9 = DepthwiseConv(1024, 1024)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, outdims)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MobileNetv2(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        # 513x25
        self.init_conv = Conv3x3(3, 32, F.relu, stride=(2,1))
        # 257x25
        self.conv1 = BottleneckBlockS1(32, 1, 16)
        self.conv2 = nn.Sequential(
                        BottleneckBlockS2(16, 6, 24),
                        BottleneckBlockS1(24, 6, 24))
        # 129x13
        self.conv3 = nn.Sequential(
                        BottleneckBlockS2(24, 6, 32, keepw=True),
                        BottleneckBlockS1(32, 6, 32),
                        BottleneckBlockS1(32, 6, 32))
        # 65x13
        self.conv4 = nn.Sequential(
                        BottleneckBlockS2(32, 6, 64),
                        BottleneckBlockS1(64, 6, 64),
                        BottleneckBlockS1(64, 6, 64),
                        BottleneckBlockS1(64, 6, 64))
        # 33x7
        self.conv5 = nn.Sequential(
                        BottleneckBlockS2(64, 6, 96, keepw=True),
                        BottleneckBlockS1(96, 6, 96),
                        BottleneckBlockS1(96, 6, 96))
        # 17x7
        self.conv6 = nn.Sequential(
                        BottleneckBlockS2(96, 6, 160, keepw=True),
                        BottleneckBlockS1(160, 6, 160),
                        BottleneckBlockS1(160, 6, 160))
        # 9x7
        self.conv7 = BottleneckBlockS1(160, 6, 320)
        self.conv8 = Conv1x1(320, 1280, F.relu6)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv9 = nn.Conv2d(1280, output_dims, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.avg_pool(x)
        x = self.conv9(x)
        x = x.view(x.size(0), -1)
        return x

class ConvNet(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        #self.activation = F.relu
        self.activation = F.leaky_relu
        self.conv1 = Conv3x3(1, 32, self.activation)
        self.conv2 = Conv3x3(32, 16, self.activation)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv3 = Conv3x3(16, 64, self.activation)
        self.conv4 = Conv3x3(64, 16, self.activation)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3)
        self.dropout2 = nn.Dropout(p=0.25)
        fcdims = 16*np.product([dim//9 for dim in input_dims])
        self.fc1 = FC(fcdims, 128, self.activation)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, output_dims)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

class Model(nn.Module):
    def __init__(self, input_dims, output_dims, model_type):
        super().__init__()
        if model_type=='convnet':
            self.cnn = ConvNet(input_dims, output_dims)
        elif model_type=='mobilenet':
            self.cnn = MobileNet(input_dims, output_dims)
        elif model_type=='mobilenetv2':
            self.cnn = MobileNetv2(input_dims, output_dims)
        self.sigmoid = nn.Sigmoid()
        num_params(self)
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.sigmoid(x)
        return x

    def generate_specs(self, device, wav):
        """Generate the vocal-accompaniment separated spectrograms"""

        mask = self.predict_mask(device, wav)
        Mvox, Macc = self.process_mask(mask)
        results = {
            "mask": { "vocals": Mvox, "accompaniment": Macc },
            "stft": stft(wav, preemp=False)
        }
        return results

    def generate_wav(self, device, wav):
        """Generate the vocal-accompaniment separated waveforms"""

        S = self.generate_specs(device, wav)
        Svox = self.apply_mask(S["mask"]["vocals"], S["stft"])
        Sacc = self.apply_mask(S["mask"]["accompaniment"], S["stft"])
        estimates = {
            "vocals": inv_spectrogram(Svox),
            "accompaniment": inv_spectrogram(Sacc)
        }
        return estimates

    def predict_mask(self, device, wav):
        """Perform a forward pass through the model to generate the
        vocal soft masks. 
        """

        self.eval()
        S = stft(wav, preemp=False)
        H, P, R = hpss_decompose(S)
        Hmel = scaled_mel_weight(H, hp.power["mix"], hp.per_channel_norm["mix"])
        Pmel = scaled_mel_weight(P, hp.power["mix"], hp.per_channel_norm["mix"])
        Rmel = scaled_mel_weight(R, hp.power["mix"], hp.per_channel_norm["mix"])
        padding = hp.stft_frames//2
        mel_spec = np.stack([Hmel, Pmel, Rmel])
        mel_spec = np.pad(mel_spec, ((0,0),(0,0),(padding,padding)), 'constant', constant_values=0)
        window = hp.stft_frames
        size = mel_spec.shape[-1]
        mask = []
        end = size - window
        for i in tqdm(range(0, end+1, hp.test_batch_size)):
            x = [mel_spec[:,:,j:j+window] for j in range(i, i+hp.test_batch_size) if j <= end]
            x = np.stack(x)
            _x = torch.FloatTensor(x).to(device)
            _y = self.forward(_x)
            y = _y.to(torch.device('cpu')).detach().numpy()
            mask += [y[j] for j in range(y.shape[0])]
        mask = np.vstack(mask).T
        return mask

    def apply_mask(self, mask, S):
        return S*mask

    def process_mask(self, mask):
        if hp.mask_at_eval:
            Mvox, Macc = self.get_binary_mask(mask)
        else:
            Mvox, Macc = self.get_soft_mask(mask)
        return Mvox, Macc

    def get_binary_mask(self, mask):
        Mvox = mask >= hp.eval_mask_threshold
        Macc = mask < hp.eval_mask_threshold
        return Mvox, Macc

    def get_soft_mask(self, mask):
        Mvox = mask * (mask > hp.eval_mask_threshold)
        inv_mask = 1 - mask
        Macc = inv_mask * (inv_mask > hp.eval_mask_threshold)
        return Mvox, Macc
        


def build_model():
    fft_bins = hp.fft_size//2+1
    model = Model((fft_bins, hp.stft_frames), fft_bins, hp.model_type)
    return model
