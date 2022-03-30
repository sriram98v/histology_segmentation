import torch
import torch.nn as nn
import logging
from unet import UNet
from Fpn import FPN
from pspnet import PSPNet
from deeplab import deeplabv3
from losses import *
from histology_dataset import histologyDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from augs import *
import segmentation_models_pytorch as sm
from misc import entropy
import matplotlib.pyplot as plt




VAL_PERCENT = 0.4
EPOCHS = 1
BATCH_SIZE = 32
LR = 5e-5

# cp_dir = "./checkpoints/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = histologyDataset("./histology_dataset/30/train/images/", "./histology_dataset/30/train/GT/", color=True, transform=transforms.Compose([Brightness(100), Rotate(), ToTensor(), Resize(size=(256, 256))]))
n_val = int(len(dataset) * VAL_PERCENT)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

# model = UNet(1, dataset.num_classes)
# model = FPN(in_channels=3, classes=dataset.num_classes, activation="sigmoid")
model = sm.PSPNet(encoder_name='resnet34', encoder_weights = None, encoder_depth = 3, psp_out_channels = 512, classes=dataset.num_classes, activation="sigmoid")
# model = deeplabv3(pretrained=False, progress=True, num_classes=9)
# model = sm.Unet(BACKBONE, encoder_weights='imagenet')
# model = PSPNet(in_channels=3, num_classes=9)
model.to(device=device)
# criterion = ComboLoss(alpha=0.4)
criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, smooth=1, gamma=1)
optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
val_entropy = []

# print(model)
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
        for batch in train_loader:
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            pred_mask = model(imgs)
            # print(imgs.shape, true_masks.shape, pred_mask.shape)
            # print(pred_mask.shape)
            loss = criterion(pred_mask, true_masks)
            epoch_loss += loss.item()

            pbar.set_postfix(**{'loss (batch)': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

            pbar.update(imgs.shape[0])

    
    model.eval()
    val_loss = 0
    entropy_log = []
    with tqdm(total=n_val, desc=f'Validation', unit='img') as pbar2:
        for batch in val_loader:
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            pred_mask = model(imgs)
            loss = criterion(pred_mask, true_masks)
            val_loss += loss.item()
            # entropy_log.append(entropy(pred_mask.detach().cpu().numpy()))
            entropy_log.append(np.mean(entropy(pred_mask.detach().cpu().numpy())))

            pbar2.set_postfix(**{'Avg Val_loss': loss.item()})

            pbar2.update(imgs.shape[0])
    
    val_entropy.append(np.mean(entropy_log))
    
    scheduler.step(val_loss/n_val)
    # torch.save(model.state_dict(), cp_dir + f'model_ep{str(epoch)}_{str(val_loss/len(val_loader))}.pth')
    # print(f"\nSaved Checkpoint. Val_loss_{val_loss/len(val_loader)}")

# torch.save(model.state_dict(), cp_dir + f'Final_model.pth')
# print
plt.plot(list(range(EPOCHS)), entropy_log)
# plt.title('Unemployment Rate Vs Year')
# plt.xlabel('Year')
# plt.ylabel('Unemployment Rate')
# plt.show()
plt.savefig("entropy.png", dpi='figure', format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)


