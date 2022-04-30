## baseline for se resnet
class ResNetSE(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_filters,
                 nOut,
                 encoder_type='ASP',
                 n_mels=80,
                 att_dim=128,
                 **kwargs):
        super(ResNetSE, self).__init__()

        print('Embedding size is %d, encoder %s.' % (nOut, encoder_type))
        self.aug = kwargs['augment']
        self.aug_chain = kwargs['augment_chain']
        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels = n_mels

        self.conv1 = nn.Conv2d(1,
                               num_filters[0],
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block,
                                       num_filters[1],
                                       layers[1],
                                       stride=(2, 2))
        self.layer3 = self._make_layer(block,
                                       num_filters[2],
                                       layers[2],
                                       stride=(2, 2))
        self.layer4 = self._make_layer(block,
                                       num_filters[3],
                                       layers[3],
                                       stride=(2, 2))
        sample_rate = int(kwargs['sample_rate'])
        hoplength = int(10e-3 * sample_rate)
        winlength = int(25e-3 * sample_rate)
        
        self.specaug = SpecAugment()
        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            features.mel.MelSpectrogram(sr=sample_rate, 
                                        n_fft=512, 
                                        win_length=winlength, 
                                        n_mels=n_mels, 
                                        hop_length=hoplength, 
                                        window='hamming', 
                                        fmin=20, fmax=4000,  
                                        trainable_mel=False, 
                                        trainable_STFT=False,
                                        verbose=False)
        )
        outmap_size = int(self.n_mels / 8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * block.expansion * outmap_size, att_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(att_dim),
            nn.Conv1d(att_dim, num_filters[3] * block.expansion * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * block.expansion * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * block.expansion * outmap_size * 2
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if self.aug and 'spec_domain' in self.aug_chain:
                x = self.specaug(x)
        x = self.instancenorm(x).unsqueeze(1)

        assert len(x.size()) == 4  # batch x channel x n_mels x n_frames
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.encoder_type == "SAP":
            x = torch.mean(x, dim=2, keepdim=True)
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(dim=1).permute(0, 2, 1)  # batch * L * D
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
            
        elif self.encoder_type == "ASP":
            x = x.reshape(x.size()[0], -1, x.size()[-1])
            w = self.attention(x)
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x