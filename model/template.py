import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

"""DSC Usage: DepthwiseSeparableConv(in_channels, out_channels, 3, 1, 1)"""
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, kernels_per_layer=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
"""ResidualBlock"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)  # 残差连接
        out = self.relu(out)
        return out
"""SelfAttention Usage: """   
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h) #X*Wq
        k = self.proj_k(h) #X*Wk
        v = self.proj_v(h) #X*Wv

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))#QKt/sqrt(dk)[1,4096,4096]
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)#softmax(QKt/sqrt(dk))

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v) #softmax(QKt/sqrt(dk))*V=Attention(Q,K,V)=head
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        out =  self.gamma+h + x #resnet
        return out
"""Transformer Encoder Usage:self.transformer=TransformerBottleneck(512,dim_feedforward=1024)"""       
class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, num_heads=8, dim_feedforward=1024, num_layers=3):
        super(TransformerEncoder, self).__init__()
        
        # Patch embedding: Flatten spatial dimensions and map to reduced feature dimension
        self.patch_embed = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels // 2, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reshape back to original dimensions after Transformer processing
        self.unpatch_embed = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
    
    def forward(self, x):
        # x: (B, C, H, W)
        batch_size, C, H, W = x.shape
        
        # Patch Embedding: Flatten the spatial dimensions
        x = self.patch_embed(x).view(batch_size, C // 2, -1).permute(2, 0, 1)  # (N, B, C)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)  # (N, B, C)
        
        # Reshape back to (B, C, H, W)
        x = x.permute(1, 2, 0).view(batch_size, C // 2, H, W)
        x = self.unpatch_embed(x)
        
        return x
"""CBAM Usage:CBAM(planes=input_channelnum)"""
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))# self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out =self.shared_MLP(self.max_pool(x))# self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
"""SE Usage: SEAttention(channel=512, reduction=8)"""
class SEAttention(nn.Module):
    # 初始化SE模块，channel为通道数，reduction为降维比率
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，将特征图的空间维度压缩为1x1
        self.fc = nn.Sequential(  # 定义两个全连接层作为激励操作，通过降维和升维调整通道重要性
            nn.Linear(channel, channel // reduction, bias=False),  # 降维，减少参数数量和计算量
            nn.ReLU(inplace=True),  # ReLU激活函数，引入非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 升维，恢复到原始通道数
            nn.Sigmoid()  # Sigmoid激活函数，输出每个通道的重要性系数
        )

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入x的批量大小b和通道数c
        y = self.avg_pool(x).view(b, c)  # 通过自适应平均池化层后，调整形状以匹配全连接层的输入
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
        return x * y.expand_as(x)  # 将通道重要性系数应用到原始特征图上，进行特征重新校准
   
class DL_MUL_COVRES_DPW_FC96(nn.Module):
    def __init__(self, num_classes=4, dropout_prob=0.1, hidden=24, num_encoders=3):
        super(DL_MUL_COVRES_DPW_FC96, self).__init__()
        self.num_encoders = num_encoders
        self.hidden=int(hidden)
        self.encoders = nn.ModuleList([self._create_encoder(self.hidden) for _ in range(num_encoders)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(num_encoders * self.hidden * 8 * 8, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self._initialize_weights()

    def _create_encoder(self, hidden):
        layers = [
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            CBAM(planes=32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            CBAM(planes=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            CBAM(planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            CBAM(planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=hidden, kernel_size=1, stride=1, padding=0),
            CBAM(planes=hidden),
        ]
        return nn.Sequential(*layers)
      
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        encode_outputs = [encoder(input.clone()) for encoder in self.encoders]
        concatenated = torch.cat(encode_outputs, dim=1)
        flattened = concatenated.view(concatenated.size(0), -1)
        output = self.dropout(flattened)
        fc1_output = F.leaky_relu(self.fc1(output))
        return fc1_output

    def get_concatenated_features(self, input):
        encode_outputs = [encoder(input.clone()) for encoder in self.encoders]
        concatenated = torch.cat(encode_outputs, dim=1)
        return concatenated
    
class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None):
        super(MultiTaskLoss, self).__init__()
        # 使用给定的权重，如果未给定则默认每个任务的权重相等
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0, 1.0]
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, targets):
        # 对每个任务计算MSE损失
        loss_O2 = self.mse_loss(preds[:, 0], targets[:, 0])
        loss_CO2 = self.mse_loss(preds[:, 1], targets[:, 1])
        loss_CH4 = self.mse_loss(preds[:, 2], targets[:, 2])
        loss_N2 = self.mse_loss(preds[:, 3], targets[:, 3])
        
        # 计算加权总损失
        total_loss = (self.weights[0] * loss_O2 +
                      self.weights[1] * loss_CO2 +
                      self.weights[2] * loss_CH4 +
                      self.weights[3] * loss_N2)
        
        return total_loss
