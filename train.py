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

VAL_PERCENT = 0.3
EPOCHS = 100
BATCH_SIZE = 32
LR = 5e-5

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
criterion = TverskyLoss(alpha=0.3, beta=0.7, smooth=1)
optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# log_interval = 10

# writer = SummaryWriter()

# global_step = 0

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
            
            # if global_step%log_interval==0:
            #     writer.add_scalar('Loss/train', np.random.random(), n_iter)
    
    model.eval()
    val_loss = 0
    with tqdm(total=n_val, desc=f'Validation', unit='img') as pbar2:
        for batch in val_loader:
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            pred_mask = model(imgs)
            loss = criterion(pred_mask, true_masks)
            val_loss += loss.item()

            pbar2.set_postfix(**{'Avg Val_loss': val_loss/n_val})

            pbar2.update(imgs.shape[0])
    
    scheduler.step(val_loss/n_val)
    torch.save(model.state_dict(), cp_dir + f'model_ep{str(epoch)}_{str(val_loss)}.pth')
    print(f"Saved Checkpoint. Val_loss_{val_loss}")

torch.save(model.state_dict(), cp_dir + f'Final_model.pth')
print("Done")


