import torch
import torch.nn as nn
import logging
from bayesian_seg import *
from losses import *
from histology_dataset import histologyDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
from torchvision import transforms
from augs import *
import segmentation_models_pytorch as sm
import metrics
from torchsummary import summary


VAL_PERCENT = 0.4
EPOCHS = 100
BATCH_SIZE = 8
LR = 5e-4

cp_dir = "./checkpoints/"
writer=SummaryWriter('content/logsdir')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = histologyDataset("./histology_dataset/30/train/images/", "./histology_dataset/30/train/GT/", color=True, transform=transforms.Compose([Brightness(100), Rotate(), ToTensor(), Resize(size=(256, 256))]))
n_val = int(len(dataset) * VAL_PERCENT)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

model = Bayesian_UNet(3, dataset.num_classes, classes=dataset.classes)
model.to(device=device)
model.switch_custom_dropout(activate=False)
model.mode("train")
criterion = ELBO_FocalTverskyLoss(alpha=0.5, beta=0.5, smooth=1, gamma=1)
optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

data = next(iter(train_loader))
writer.add_graph(model,data['image'].to(device=device, dtype=torch.float32))
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    kl = 0
    
    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
        for i, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            pred_mask  = model(imgs)
            # kl_ = model.get_kl()
            loss = criterion(pred_mask, true_masks, model.get_kl(), beta=1e-7)
            epoch_loss += loss.item()
            kl += model.get_kl()

            pbar.set_postfix(**{'loss (batch)': loss.item()})

            loss.backward()
            optimizer.step()

            pbar.update(imgs.shape[0])
        
    writer.add_scalar('training loss',
                        epoch_loss/len(train_loader),
                        epoch)
    
    writer.add_scalar('KL Divergence',
                        kl/len(train_loader),
                        epoch)

    
    model.eval()
    val_loss = 0
    with tqdm(total=n_val, desc=f'Validation', unit='img') as pbar2:
        for i, batch in enumerate(val_loader, 1):
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            pred_mask = model(imgs)
            kl_ = model.get_kl()
            loss = criterion(pred_mask, true_masks, model.get_kl(), beta=1e-7)
            val_loss += loss.item()

            pbar2.set_postfix(**{'Avg Val_loss': loss.item()})

            pbar2.update(imgs.shape[0])
    writer.add_scalar('val loss',
                            val_loss/len(val_loader),
                            epoch)
            
    
    scheduler.step(val_loss/n_val)
    torch.save(model.state_dict(), cp_dir + f'model_ep{str(epoch)}_{str(val_loss/len(val_loader))}.pth')
    print(f"\nSaved Checkpoint. Val_loss: {val_loss/len(val_loader)}. Train_loss: {epoch_loss/len(train_loader)}")

torch.save(model.state_dict(), cp_dir + f'Final_model.pth')
print("Done")


