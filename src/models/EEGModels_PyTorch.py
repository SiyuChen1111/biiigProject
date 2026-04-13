"""
ARL_EEGModels - PyTorch Implementation
Converted from Keras/TensorFlow version

A collection of Convolutional Neural Network models for EEG
Signal Processing and Classification, using PyTorch

Requirements:
    (1) pytorch >= 1.0
    (2) numpy

Original paper: http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

To use:
    
    from EEGModels_PyTorch import EEGNet
    
    model = EEGNet(nb_classes=..., Chans=..., Samples=...)
    
    # Then train with standard PyTorch training loop
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

Portions of this project are works of the United States Government and are not
subject to domestic copyright protection under 17 USC Sec. 105. Those 
portions are released world-wide under the terms of the Creative Commons Zero 
1.0 (CC0) license. 

Other portions of this project are subject to domestic copyright protection 
under 17 USC Sec. 105. Those portions are licensed under the Apache 2.0 
license. The complete text of the license governing this material is in the 
file labeled LICENSE.TXT that is a part of this project's official 
distribution. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxNormConstraint:
    """
    Max norm constraint for linear layers.
    Clips the weight norm to max_value after each forward pass.
    """
    def __init__(self, max_value=0.25):
        self.max_value = max_value
    
    def __call__(self, module):
        with torch.no_grad():
            if hasattr(module, 'weight'):
                norm = module.weight.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, 0, self.max_value)
                scale = desired / (norm + 1e-8)
                module.weight.data *= scale


class DepthwiseConv2d(nn.Module):
    """
    Depthwise Convolution with depth multiplier.
    Equivalent to Keras DepthwiseConv2D.
    """
    def __init__(self, in_channels, depth_multiplier=1, kernel_size=(1, 1), 
                 stride=(1, 1), padding=(0, 0), bias=False, max_norm_val=1.0):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels * depth_multiplier,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        self.max_norm_val = max_norm_val
    
    def forward(self, x):
        return self.depthwise(x)
    
    def apply_max_norm(self):
        with torch.no_grad():
            for i in range(self.depthwise.weight.shape[1]):
                norm = self.depthwise.weight[:, i, :, :].norm(2)
                if norm > self.max_norm_val:
                    self.depthwise.weight[:, i, :, :] *= self.max_norm_val / norm


class SeparableConv2d(nn.Module):
    """
    Separable Convolution: Depthwise followed by Pointwise.
    Equivalent to Keras SeparableConv2D.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1),
                 stride=(1, 1), padding=(0, 0), bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, 1),
            bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Square(nn.Module):
    """Square activation function for ShallowConvNet."""
    def forward(self, x):
        return torch.square(x)


class Log(nn.Module):
    """Log activation function for ShallowConvNet."""
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-7, max=10000))


class EEGNet(nn.Module):
    """PyTorch Implementation of EEGNet
    
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
    advised to do some model searching to get optimal performance on your
    particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either 'SpatialDropout2D' or 'Dropout', passed as a string.

    Notes for transfer learning:

      - forward(x) returns logits of shape (batch, nb_classes)
      - forward_features(x) returns flattened backbone features of shape
        (batch, feature_dim)
      - reset_classifier(nb_classes) replaces the task head while keeping the
        feature extractor weights
    """
    
    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()
        
        self.Chans = Chans
        self.Samples = Samples
        self.norm_rate = norm_rate
        
        if dropoutType == 'SpatialDropout2D':
            self.dropout_type = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropout_type = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
        
        # Block 1
        self.conv2d_1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), 
                                   padding=(0, kernLength//2), bias=False)
        self.bn_1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = DepthwiseConv2d(F1, depth_multiplier=D, 
                                               kernel_size=(Chans, 1),
                                               max_norm_val=1.0)
        self.bn_2 = nn.BatchNorm2d(F1 * D)
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout_1 = self.dropout_type(dropoutRate)
        
        # Block 2
        self.separable_conv = SeparableConv2d(F1 * D, F2, kernel_size=(1, 16),
                                               padding=(0, 8))
        self.bn_3 = nn.BatchNorm2d(F2)
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.dropout_2 = self.dropout_type(dropoutRate)
        
        # Classifier
        self.flatten = nn.Flatten()
        self.feature_dim = F2 * (Samples // 32)
        self.classifier = nn.Linear(self.feature_dim, nb_classes)
        self.dense = self.classifier

    def _prepare_input(self, x):
        """Convert supported input layouts to NCHW for Conv2d."""
        if x.dim() == 4 and x.shape[-1] == 1:
            return x.permute(0, 3, 1, 2)

        if x.dim() == 3:
            return x.unsqueeze(1)

        return x

    def forward_features(self, x):
        """Return flattened EEGNet backbone features for transfer learning."""
        x = self._prepare_input(x)

        # Block 1
        x = self.conv2d_1(x) #shape: (batch, F1, Chans, Samples)，时间卷积,提取每个通道的时间模式。
        x = self.bn_1(x)
        x = self.depthwise_conv(x) #shape: (batch, F1 * D, Chans, Samples)，空间卷积,学习通道间空间模式。
        x = self.bn_2(x)
        x = F.elu(x)
        x = self.avg_pool_1(x)
        x = self.dropout_1(x)

        # Block 2
        x = self.separable_conv(x)
        x = self.bn_3(x)
        x = F.elu(x)
        x = self.avg_pool_2(x)
        x = self.dropout_2(x)

        return self.flatten(x)

    def forward_classifier(self, features):
        """Apply the task-specific classification head to extracted features."""
        return self.classifier(features)

    def reset_classifier(self, nb_classes):
        """Replace the classifier head for a new task label space."""
        new_classifier = nn.Linear(self.feature_dim, nb_classes)
        new_classifier = new_classifier.to(
            device=self.classifier.weight.device,
            dtype=self.classifier.weight.dtype
        )
        self.classifier = new_classifier
        self.dense = self.classifier
        return self.classifier

    def predict_proba(self, x):
        """Return class probabilities for inference workflows."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
        
    def forward(self, x):
        features = self.forward_features(x)
        logits = self.forward_classifier(features)

        return logits
    
    def apply_max_norm_constraints(self):
        """Apply max norm constraints to weights."""
        self.depthwise_conv.apply_max_norm()
        with torch.no_grad():
            norm = self.classifier.weight.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, self.norm_rate)
            scale = desired / (norm + 1e-8)
            self.classifier.weight.data *= scale


class EEGNet_SSVEP(nn.Module):
    """SSVEP Variant of EEGNet, as used in [1]. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. 
      D               : number of spatial filters to learn within each temporal
                        convolution.
      dropoutType     : Either 'SpatialDropout2D' or 'Dropout', passed as a string.
      
      
    [1]. Waytowich, N. et. al. (2018). Compact Convolutional Neural Networks
    for Classification of Asynchronous Steady-State Visual Evoked Potentials.
    Journal of Neural Engineering vol. 15(6). 
    http://iopscience.iop.org/article/10.1088/1741-2552/aae5d8

    """
    
    def __init__(self, nb_classes=12, Chans=8, Samples=256, 
                 dropoutRate=0.5, kernLength=256, F1=96, 
                 D=1, F2=96, dropoutType='Dropout'):
        super(EEGNet_SSVEP, self).__init__()
        
        self.Chans = Chans
        self.Samples = Samples
        
        if dropoutType == 'SpatialDropout2D':
            self.dropout_type = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropout_type = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
        
        # Block 1
        self.conv2d_1 = nn.Conv2d(1, F1, kernel_size=(1, kernLength), 
                                   padding=(0, kernLength//2), bias=False)
        self.bn_1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = DepthwiseConv2d(F1, depth_multiplier=D, 
                                               kernel_size=(Chans, 1),
                                               max_norm_val=1.0)
        self.bn_2 = nn.BatchNorm2d(F1 * D)
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout_1 = self.dropout_type(dropoutRate)
        
        # Block 2
        self.separable_conv = SeparableConv2d(F1 * D, F2, kernel_size=(1, 16),
                                               padding=(0, 8))
        self.bn_3 = nn.BatchNorm2d(F2)
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.dropout_2 = self.dropout_type(dropoutRate)
        
        # Classifier
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (Samples // 32), nb_classes)
        
    def forward(self, x):
        # Input shape: (batch, Chans, Samples, 1) -> (batch, 1, Chans, Samples)
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        
        # Block 1
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.depthwise_conv(x)
        x = self.bn_2(x)
        x = F.elu(x)
        x = self.avg_pool_1(x)
        x = self.dropout_1(x)
        
        # Block 2
        x = self.separable_conv(x)
        x = self.bn_3(x)
        x = F.elu(x)
        x = self.avg_pool_2(x)
        x = self.dropout_2(x)
        
        # Classifier
        x = self.flatten(x)
        x = self.dense(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    def apply_max_norm_constraints(self):
        """Apply max norm constraints to weights."""
        self.depthwise_conv.apply_max_norm()


class EEGNet_old(nn.Module):
    """PyTorch Implementation of EEGNet_v1 (https://arxiv.org/abs/1611.08024v2)

    This model is the original EEGNet model proposed on arxiv
            https://arxiv.org/abs/1611.08024v2
    
    with a few modifications: we use striding instead of max-pooling as this 
    helped slightly in classification performance while also providing a 
    computational speed-up. 
    
    Note that we no longer recommend the use of this architecture, as the new
    version of EEGNet performs much better overall and has nicer properties.
    
    Inputs:
        
        nb_classes     : total number of final categories
        Chans, Samples : number of EEG channels and samples, respectively
        regRate        : regularization rate for L1 and L2 regularizations
        dropoutRate    : dropout fraction
        kernels        : the 2nd and 3rd layer kernel dimensions (default is 
                         the [2, 32] x [8, 4] configuration)
        strides        : the stride size (note that this replaces the max-pool
                         used in the original paper)
    
    """
    
    def __init__(self, nb_classes, Chans=64, Samples=128, regRate=0.0001,
                 dropoutRate=0.25, kernels=[(2, 32), (8, 4)], strides=(2, 4)):
        super(EEGNet_old, self).__init__()
        
        self.Chans = Chans
        self.Samples = Samples
        self.kernels = kernels
        self.strides = strides
        
        # Layer 1
        self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=(Chans, 1))
        self.bn_1 = nn.BatchNorm2d(16)
        self.dropout_1 = nn.Dropout(dropoutRate)
        
        # Layer 2
        self.conv2d_2 = nn.Conv2d(16, 4, kernel_size=kernels[0], 
                                   padding=(kernels[0][0]//2, kernels[0][1]//2),
                                   stride=strides)
        self.bn_2 = nn.BatchNorm2d(4)
        self.dropout_2 = nn.Dropout(dropoutRate)
        
        # Layer 3
        self.conv2d_3 = nn.Conv2d(4, 4, kernel_size=kernels[1],
                                   padding=(kernels[1][0]//2, kernels[1][1]//2),
                                   stride=strides)
        self.bn_3 = nn.BatchNorm2d(4)
        self.dropout_3 = nn.Dropout(dropoutRate)
        
        # Classifier
        self.flatten = nn.Flatten()
        
        # Calculate flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            dummy = self._forward_features(dummy)
            self.flatten_size = dummy.view(1, -1).shape[1]
        
        self.dense = nn.Linear(self.flatten_size, nb_classes)
        
    def _forward_features(self, x):
        # Layer 1
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = F.elu(x)
        x = self.dropout_1(x)
        
        # Permute: (batch, C, H, W) -> (batch, H, C, W) equivalent to Keras Permute(2,1,3)
        x = x.permute(0, 2, 1, 3)
        
        # Layer 2
        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = F.elu(x)
        x = self.dropout_2(x)
        
        # Layer 3
        x = self.conv2d_3(x)
        x = self.bn_3(x)
        x = F.elu(x)
        x = self.dropout_3(x)
        
        return x
        
    def forward(self, x):
        # Input shape: (batch, Chans, Samples) -> (batch, 1, Chans, Samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = F.softmax(x, dim=1)
        
        return x


class DeepConvNet(nn.Module):
    """PyTorch implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """
    
    def __init__(self, nb_classes, Chans=64, Samples=256, dropoutRate=0.5):
        super(DeepConvNet, self).__init__()
        
        self.Chans = Chans
        self.Samples = Samples
        
        # Block 1
        self.conv2d_1a = nn.Conv2d(1, 25, kernel_size=(1, 5))
        self.conv2d_1b = nn.Conv2d(25, 25, kernel_size=(Chans, 1))
        self.bn_1 = nn.BatchNorm2d(25, eps=1e-5, momentum=0.1)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout_1 = nn.Dropout(dropoutRate)
        
        # Block 2
        self.conv2d_2 = nn.Conv2d(25, 50, kernel_size=(1, 5))
        self.bn_2 = nn.BatchNorm2d(50, eps=1e-5, momentum=0.1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout_2 = nn.Dropout(dropoutRate)
        
        # Block 3
        self.conv2d_3 = nn.Conv2d(50, 100, kernel_size=(1, 5))
        self.bn_3 = nn.BatchNorm2d(100, eps=1e-5, momentum=0.1)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout_3 = nn.Dropout(dropoutRate)
        
        # Block 4
        self.conv2d_4 = nn.Conv2d(100, 200, kernel_size=(1, 5))
        self.bn_4 = nn.BatchNorm2d(200, eps=1e-5, momentum=0.1)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout_4 = nn.Dropout(dropoutRate)
        
        # Classifier
        self.flatten = nn.Flatten()
        
        # Calculate flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            dummy = self._forward_features(dummy)
            self.flatten_size = dummy.view(1, -1).shape[1]
        
        self.dense = nn.Linear(self.flatten_size, nb_classes)
        
    def _forward_features(self, x):
        # Block 1
        x = self.conv2d_1a(x)
        x = self.conv2d_1b(x)
        x = self.bn_1(x)
        x = F.elu(x)
        x = self.maxpool_1(x)
        x = self.dropout_1(x)
        
        # Block 2
        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = F.elu(x)
        x = self.maxpool_2(x)
        x = self.dropout_2(x)
        
        # Block 3
        x = self.conv2d_3(x)
        x = self.bn_3(x)
        x = F.elu(x)
        x = self.maxpool_3(x)
        x = self.dropout_3(x)
        
        # Block 4
        x = self.conv2d_4(x)
        x = self.bn_4(x)
        x = F.elu(x)
        x = self.maxpool_4(x)
        x = self.dropout_4(x)
        
        return x
        
    def forward(self, x):
        # Input shape: (batch, Chans, Samples, 1) -> (batch, 1, Chans, Samples)
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    def apply_max_norm_constraints(self):
        """Apply max norm constraints to weights."""
        max_norm_val = 2.0
        for conv in [self.conv2d_1a, self.conv2d_1b, self.conv2d_2, 
                     self.conv2d_3, self.conv2d_4]:
            with torch.no_grad():
                norm = conv.weight.norm(2)
                if norm > max_norm_val:
                    conv.weight.data *= max_norm_val / norm
        
        with torch.no_grad():
            norm = self.dense.weight.norm(2)
            if norm > 0.5:
                self.dense.weight.data *= 0.5 / norm


class ShallowConvNet(nn.Module):
    """PyTorch implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25    
    
    Note that this implementation has not been verified by the original 
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations. 
    """
    
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
        super(ShallowConvNet, self).__init__()
        
        self.Chans = Chans
        self.Samples = Samples
        
        # Block 1
        self.conv2d_1a = nn.Conv2d(1, 40, kernel_size=(1, 13))
        self.conv2d_1b = nn.Conv2d(40, 40, kernel_size=(Chans, 1), bias=False)
        self.bn_1 = nn.BatchNorm2d(40, eps=1e-5, momentum=0.1)
        self.square = Square()
        self.avgpool_1 = nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7))
        self.log = Log()
        self.dropout_1 = nn.Dropout(dropoutRate)
        
        # Classifier
        self.flatten = nn.Flatten()
        
        # Calculate flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            dummy = self._forward_features(dummy)
            self.flatten_size = dummy.view(1, -1).shape[1]
        
        self.dense = nn.Linear(self.flatten_size, nb_classes)
        
    def _forward_features(self, x):
        # Block 1
        x = self.conv2d_1a(x)
        x = self.conv2d_1b(x)
        x = self.bn_1(x)
        x = self.square(x)
        x = self.avgpool_1(x)
        x = self.log(x)
        x = self.dropout_1(x)
        
        return x
        
    def forward(self, x):
        # Input shape: (batch, Chans, Samples, 1) -> (batch, 1, Chans, Samples)
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)
        
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    def apply_max_norm_constraints(self):
        """Apply max norm constraints to weights."""
        max_norm_val = 2.0
        for conv in [self.conv2d_1a, self.conv2d_1b]:
            with torch.no_grad():
                norm = conv.weight.norm(2)
                if norm > max_norm_val:
                    conv.weight.data *= max_norm_val / norm
        
        with torch.no_grad():
            norm = self.dense.weight.norm(2)
            if norm > 0.5:
                self.dense.weight.data *= 0.5 / norm


if __name__ == '__main__':
    # Test models
    batch_size = 16
    nb_classes = 4
    Chans = 64
    Samples = 128
    
    # Test EEGNet
    print("Testing EEGNet...")
    model = EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    x = torch.randn(batch_size, Chans, Samples, 1)
    y = model(x)
    features = model.forward_features(x)
    probs = model.predict_proba(x)
    print(f"EEGNet input shape: {x.shape}, logits shape: {y.shape}, feature shape: {features.shape}, probability shape: {probs.shape}")
    
    # Test EEGNet_SSVEP
    print("\nTesting EEGNet_SSVEP...")
    model_ssvep = EEGNet_SSVEP(nb_classes=12, Chans=8, Samples=256)
    x_ssvep = torch.randn(batch_size, 8, 256, 1)
    y_ssvep = model_ssvep(x_ssvep)
    print(f"EEGNet_SSVEP input shape: {x_ssvep.shape}, output shape: {y_ssvep.shape}")
    
    # Test DeepConvNet
    print("\nTesting DeepConvNet...")
    model_deep = DeepConvNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    y_deep = model_deep(x)
    print(f"DeepConvNet input shape: {x.shape}, output shape: {y_deep.shape}")
    
    # Test ShallowConvNet
    print("\nTesting ShallowConvNet...")
    model_shallow = ShallowConvNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    y_shallow = model_shallow(x)
    print(f"ShallowConvNet input shape: {x.shape}, output shape: {y_shallow.shape}")
    
    print("\nAll models tested successfully!")
