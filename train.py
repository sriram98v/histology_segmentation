import torch
import torch.nn as nn
import logging
from unet import UNet
from Fpn import FPN
from losses import *
from histology_dataset import histologyDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm

VAL_PERCENT = 0.1
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.1

cp_dir = "./checkpoints/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = histologyDataset("./histology_dataset/train/images/", "./histology_dataset/train/GT/")
n_val = int(len(dataset) * VAL_PERCENT)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

# model = UNet(1, dataset.num_classes)
model = FPN(in_channels=1, classes=dataset.num_classes)
model.to(device=device)
criterion = FocalLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')

# logging.info(f'''Starting training:
#         Epochs:          {EPOCHS}
#         Batch size:      {BATCH_SIZE}
#         Learning rate:   {LR}
#         Training size:   {n_train}
#         Validation size: {n_val}
#         Checkpoints:     {save_cp}
#         Device:          {device.type}
#     ''')

best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
        for batch in train_loader:
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            pred_mask = model(imgs)
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
    with tqdm(total=n_val, desc=f'Validation', unit='img') as pbar2:
        for batch in val_loader:
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            pred_mask = model(imgs)
            loss = criterion(pred_mask, true_masks)
            val_loss += loss.item()

            pbar2.set_postfix(**{'Val_loss': val_loss/n_val})

            pbar2.update(imgs.shape[0])
        
    if val_loss<=best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), cp_dir + f'best_model_{str(val_loss)}.pth')
        print("Loss improved. Saved Checkpoint")

torch.save(model.state_dict(), cp_dir + f'Final_model.pth')
print("Done")


