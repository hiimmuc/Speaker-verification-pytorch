import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

from models.ECAPA_TDNN import *
from models.ECAPA_utils import Conv1d as _Conv1d
from models.ECAPA_utils import BatchNorm1d as _BatchNorm1d
from models.conformer.conformer.model import Conformer

from utils import PreEmphasis
from nnAudio import features



class Conformer_(torch.nn.Module):
    """
    """

    def __init__(
        self,
        input_size=80,
        device="cpu",
        lin_neurons=192,
        activation=torch.nn.GELU,
        attention_channels=128,
        global_context=True,
        **kwargs
    ):

        super().__init__()

        self.aug = kwargs['augment']
        self.aug_chain = kwargs['augment_chain']
        sample_rate = int(kwargs['sample_rate'])
        hoplength = int(10e-3 * sample_rate)
        winlength = int(25e-3 * sample_rate)
        n_frames = sample_rate//hoplength + 2
        n_mels = kwargs['n_mels']
        input_size = n_mels
        
        self.blocks = nn.ModuleList()
        
        # we have 2 version of mels here before 0215 is old
        # check version of model
        # model saved name: path/domain_date_time_desc.model
        if 'initial_model_infer' in kwargs:
            if (kwargs['initial_model_infer'] is not None):
                if 'best_state' not in kwargs['initial_model_infer']:
                    version = int(kwargs['initial_model_infer'].split('/')[-1].split('_')[1])
                    fb_type = 'nnAudio'
                elif 'best_state' in kwargs['initial_model_infer']:
                    fb_type = 'nnAudio'
            else:
                fb_type = 'torchaudio'
                version = -1
                
        if  fb_type == 'nnAudio' or version >= 215:
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                features.mel.MelSpectrogram(sr=sample_rate, 
                                            n_fft=512, 
                                            win_length=winlength, 
                                            n_mels=n_mels, 
                                            hop_length=hoplength, 
                                            window='hamming', 
                                            fmin=0.0, fmax=4000,  
                                            trainable_mel=True, 
                                            trainable_STFT=True,
                                            verbose=False)
            )
        else:
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, 
                                                     n_fft=512, 
                                                     win_length=winlength, 
                                                     hop_length=hoplength,
                                                     f_min=10, f_max=4000,
                                                     window_fn=torch.hamming_window, 
                                                     n_mels=n_mels)
            )


        self.specaug = SpecAugment() # Spec augmentation

        # Conformer
        self.conformer_block = Conformer(
            input_dim = n_mels,
            encoder_dim = 144,
            num_attention_heads = 8,
            feed_forward_expansion_factor = 4,
            conv_expansion_factor = 2,
            input_dropout_p = 0.1,
            feed_forward_dropout_p = 0.1,
            attention_dropout_p = 0.1,
            conv_dropout_p = 0.1,
            conv_kernel_size = 3,
            half_step_residual = True,
            num_classes= lin_neurons,   
            num_encoder_layers=  16,
        )
        
        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            lin_neurons,
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=(lin_neurons * 2))

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=lin_neurons * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x, lengths=None):
        """Returns the embedding vector.
        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if self.aug and 'spec_domain' in self.aug_chain:
                x = self.specaug(x)

        # conformer
        x = x.transpose(1, -1)
        x = self.conformer_block(x)        
        x = x.transpose(1, -1)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)
        x = x.squeeze()       
        return x
    
def MainModel(nOut=512, **kwargs):
    model = Conformer_(lin_neurons=nOut, **kwargs)
    return model