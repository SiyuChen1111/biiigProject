"""
Sample script using EEGNet (PyTorch) to classify Event-Related Potential (ERP) EEG data
from a four-class classification task, using the sample dataset provided in
the MNE [1, 2] package:
    https://martinos.org/mne/stable/manual/sample_dataset.html#ch-sample-data
  
The four classes used from this dataset are:
    LA: Left-ear auditory stimulation
    RA: Right-ear auditory stimulation
    LV: Left visual field stimulation
    RV: Right visual field stimulation

The code to process, filter and epoch the data are originally from Alexandre
Barachant's PyRiemann [3] package, released under the BSD 3-clause. A copy of 
the BSD 3-clause license has been provided together with this software to 
comply with software licensing requirements. 

When you first run this script, MNE will download the dataset and prompt you
to confirm the download location (defaults to ~/mne_data). Follow the prompts
to continue. The dataset size is approx. 1.5GB download. 

For comparative purposes you can also compare EEGNet performance to using 
Riemannian geometric approaches with xDAWN spatial filtering [4-8] using 
PyRiemann (code provided below).

[1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck,
    L. Parkkonen, M. Hämäläinen, MNE software for processing MEG and EEG data, 
    NeuroImage, Volume 86, 1 February 2014, Pages 446-460, ISSN 1053-8119.

[2] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, 
    R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen, MEG and EEG data 
    analysis with MNE-Python, Frontiers in Neuroscience, Volume 7, 2013.

[3] https://github.com/alexandrebarachant/pyRiemann. 

[4] A. Barachant, M. Congedo ,"A Plug&Play P300 BCI Using Information Geometry"
    arXiv:1409.0107. link

[5] M. Congedo, A. Barachant, A. Andreev ,"A New generation of Brain-Computer 
    Interface Based on Riemannian Geometry", arXiv: 1310.8115.

[6] A. Barachant and S. Bonnet, "Channel selection procedure using riemannian 
    distance for BCI applications," in 2011 5th International IEEE/EMBS 
    Conference on Neural Engineering (NER), 2011, 348-351.

[7] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass 
    Brain-Computer Interface Classification by Riemannian Geometry," in IEEE 
    Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

[8] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of 
    covariance matrices using a Riemannian-based kernel for BCI applications", 
    in NeuroComputing, vol. 112, p. 172-178, 2013.


Portions of this project are works of the United States Government and are not
subject to domestic copyright protection under 17 USC Sec. 105.  Those 
portions are released world-wide under the terms of the Creative Commons Zero 
1.0 (CC0) license.  

Other portions of this project are subject to domestic copyright protection 
under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0 
license.  The complete text of the license governing this material is in the 
file labeled LICENSE.TXT that is a part of this project's official 
distribution. 
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset

# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports (PyTorch version)
from src.models.EEGModels_PyTorch import EEGNet

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

##################### Process, filter and epoch the data ######################
data_path = sample.data_path()
data_root = Path(data_path)

# Set parameters and read data
raw_fname = data_root / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
event_fname = data_root / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)
labels = epochs.events[:, -1]

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data()*1000  # format is in (trials, channels, samples)
y = labels

kernels, chans, samples = 1, 60, 151

# take 50/25/25 percent of the data to train/validate/test
X_train = X[0:144,]
Y_train = y[0:144]
X_validate = X[144:216,]
Y_validate = y[144:216]
X_test = X[216:,]
Y_test = y[216:]

############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

Y_train_oh = to_categorical(Y_train-1, 4)
Y_validate_oh = to_categorical(Y_validate-1, 4)
Y_test_oh = to_categorical(Y_test-1, 4)

# convert data to NHWC (trials, channels, samples, kernels) format. Data 
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)
   
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes=4, Chans=chans, Samples=samples, 
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16, 
               dropoutType='Dropout')
model = model.to(device)

# count number of parameters in the model
numParams = sum(p.numel() for p in model.parameters())
print(f'Number of parameters: {numParams}')

# convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
Y_train_tensor = torch.LongTensor(Y_train-1).to(device)  # CrossEntropyLoss expects class indices
X_validate_tensor = torch.FloatTensor(X_validate).to(device)
Y_validate_tensor = torch.LongTensor(Y_validate-1).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
Y_test_tensor = torch.LongTensor(Y_test-1).to(device)

# create DataLoader
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = TensorDataset(X_validate_tensor, Y_validate_tensor)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

# class weights (all set to 1 as data is balanced)
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)
criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)

# training loop
num_epochs = 300
best_val_acc = 0.0
best_model_path = '/tmp/checkpoint.pt'

print('Starting training...')
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion_weighted(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        # Apply max norm constraints
        model.apply_max_norm_constraints()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
    
    train_acc = train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
    
    val_acc = val_correct / val_total
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.4f} [BEST]')
    else:
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, '
                  f'Val Acc: {val_acc:.4f}')

print('Training finished!')

# load optimal weights
model.load_state_dict(torch.load(best_model_path))
print(f'Loaded best model with validation accuracy: {best_val_acc:.4f}')

###############################################################################
# make prediction on test set.
###############################################################################
model.eval()
with torch.no_grad():
    probs = model(X_test_tensor)
    preds = torch.argmax(probs, dim=1).cpu().numpy()
    
acc = np.mean(preds == Y_test-1)
print("Classification accuracy: %f " % (acc))


############################# PyRiemann Portion ##############################

# code is taken from PyRiemann's ERP sample script, which is decoding in 
# the tangent space with a logistic regression

n_components = 2  # pick some components

# set up sklearn pipeline
clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds_rg = np.zeros(len(Y_test))

# reshape back to (trials, channels, samples)
X_train_rg = X_train.reshape(X_train.shape[0], chans, samples)
X_test_rg = X_test.reshape(X_test.shape[0], chans, samples)

# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format
clf.fit(X_train_rg, Y_train-1)
preds_rg = clf.predict(X_test_rg)

# Printing the results
acc2 = np.mean(preds_rg == Y_test-1)
print("Classification accuracy (PyRiemann): %f " % (acc2))

# plot the confusion matrices for both classifiers
names = ['audio left', 'audio right', 'vis left', 'vis right']
plt.figure(0)
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(Y_test - 1, preds),
    display_labels=names,
).plot()
plt.title('EEGNet-8,2 (PyTorch)')

plt.figure(1)
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(Y_test - 1, preds_rg),
    display_labels=names,
).plot()
plt.title('xDAWN + RG')

plt.show()
