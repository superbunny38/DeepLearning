#ResNet18 from scratch

class Custom_RESNET(nn.Module):
    def __init__(self):
        super(Custom_RESNET, self).__init__()
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)


        #Block1-1
        self.conv2_1 = nn.Conv2d(64,64,3,padding=1)
        self.conv2_2 = nn.Conv2d(64,64,3,padding=1)


        self.conv2_3 = nn.Conv2d(64,64,3,padding=1)
        self.conv2_4 = nn.Conv2d(64,64,3,padding=1)
        self.bn2_1 = nn.BatchNorm2d(num_features=64)
        self.bn2_2 = nn.BatchNorm2d(num_features=64)
        self.bn2_3 = nn.BatchNorm2d(num_features=64)
        self.bn2_4 = nn.BatchNorm2d(num_features=64)

        self.conv3_1 = nn.Conv2d(64,128,3,padding=1, stride=2)
        self.conv3_2 = nn.Conv2d(128,128,3,padding=1)
        self.conv3_3 = nn.Conv2d(128,128,3,padding=1)
        self.conv3_4 = nn.Conv2d(128,128,3,padding=1)
        self.bn3_1 = nn.BatchNorm2d(num_features=128)
        self.bn3_2 = nn.BatchNorm2d(num_features=128)
        self.bn3_3 = nn.BatchNorm2d(num_features=128)
        self.bn3_4 = nn.BatchNorm2d(num_features=128)

        self.conv4_1 = nn.Conv2d(128,256,3,padding=1, stride=2)
        self.conv4_2 = nn.Conv2d(256,256,3,padding=1)
        self.conv4_3 = nn.Conv2d(256,256,3,padding=1)
        self.conv4_4 = nn.Conv2d(256,256,3,padding=1)
        self.bn4_1 = nn.BatchNorm2d(num_features=256)
        self.bn4_2 = nn.BatchNorm2d(num_features=256)
        self.bn4_3 = nn.BatchNorm2d(num_features=256)
        self.bn4_4 = nn.BatchNorm2d(num_features=256)

        self.conv5_1 = nn.Conv2d(256,512,3,padding=1, stride=2)
        self.conv5_2 = nn.Conv2d(512,512,3,padding=1)
        self.conv5_3 = nn.Conv2d(512,512,3,padding=1)
        self.conv5_4 = nn.Conv2d(512,512,3,padding=1)
        self.bn5_1 = nn.BatchNorm2d(num_features=512)
        self.bn5_2 = nn.BatchNorm2d(num_features=512)
        self.bn5_3 = nn.BatchNorm2d(num_features=512)
        self.bn5_4 = nn.BatchNorm2d(num_features=512)

        self.adaptiveavgpool2d = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.maxpool2d(self.relu(self.bn1(self.conv1(x))))

        #Block1 (downsample = None)
        identity = x
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.bn2_2(self.conv2_2(x))
        x += identity
        x = self.relu(x)

        identity = x
        x = self.relu(self.bn2_3(self.conv2_3(x)))
        x = self.bn2_4(self.conv2_4(x))
        x += identity
        x = self.relu(x)

        #Block2
        identity_downsample = self.bn3_1(self.conv3_1(x))
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.bn3_2(self.conv3_2(x))
        x += identity_downsample
        x = self.relu(x)

        identity = x
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        x = self.bn3_4(self.conv3_4(x))
        x += identity
        x = self.relu(x)

        #Block3
        identity_downsample = self.bn4_1(self.conv4_1(x))
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.bn4_2(self.conv4_2(x))
        x += identity_downsample
        x = self.relu(x)

        identity = x
        x = self.relu(self.bn4_3(self.conv4_3(x)))
        x = self.bn4_4(self.conv4_4(x))
        x += identity
        x = self.relu(x)

        #Block4
        identity_downsample = self.bn5_1(self.conv5_1(x))
        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.bn5_2(self.conv5_2(x))
        x += identity_downsample
        x = self.relu(x)

        identity = x
        x = self.relu(self.conv5_3(x))
        x = self.bn5_4(self.conv5_4(x))
        x += identity
        x5_4 = self.relu(x)

        x = self.adaptiveavgpool2d(x5_4)
        x = x.view(-1,512)
        x = self.fc(x)
        return x

model = Custom_RESNET().to(DEVICE)
summary(model, (3, 32, 32))
