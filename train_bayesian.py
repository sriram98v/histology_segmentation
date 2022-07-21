import torch
from BayesianSeg.models.bayes_unet import *
from BayesianSeg.loss.losses import *
from BayesianSeg.datasets.histology_dataset import histologyDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
from torchvision import transforms
from BayesianSeg.datasets.augs import *
from BayesianSeg.metrics.metrics import get_TI

VAL_PERCENT = 0.4
EPOCHS = 500
BATCH_SIZE = 8
LR = 0.001

cp_dir = "./checkpoints/kvasir/"
writer=SummaryWriter('content/logsdir')

device = torch.device('cuda:0')

dataset = histologyDataset("./histology_dataset/30/train/images/", "./histology_dataset/30/train/GT/", color=True, 
                            transform=transforms.Compose([Rotate(), ToTensor(), Resize(size=(256, 256))]),
                            classes=['AS', 'Cartilage', 'RS', 'SM'])

n_val = int(len(dataset) * VAL_PERCENT)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

model = Bayesian_UNet(3, dataset.num_classes, classes=dataset.classes)
model.to(device=device)
criterion_m = DiceBCELoss()
criterion_kl = BKLLoss(last_layer_only=False)
kl_weight = 0.1
optimizer = optim.Adam(model.parameters(), lr=LR)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

data = next(iter(train_loader))
writer.add_graph(model,data['image'].to(device=device, dtype=torch.float32))
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    kl = 0
    total_TI = 0
    
    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
        for i, batch in enumerate(train_loader, 1):
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            pred_mask  = model(imgs)
            loss_m = criterion_m(pred_mask, true_masks)
            loss_kl = criterion_kl(model)
            loss = loss_m + kl*loss_kl
            total_TI += (1 - get_TI(pred=pred_mask, true=true_masks, alpha=1, beta=1, smooth=0, gamma=1).item())
            epoch_loss += loss.item()

            pbar.set_postfix(**{'TI (batch)': loss_m.item(), 'KL Div (batch)': loss_kl.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(imgs.shape[0])
            kl += loss_kl.item()
        
    writer.add_scalar('training loss',
                        epoch_loss/len(train_loader),
                        epoch)
    
    writer.add_scalar('KL Divergence',
                        kl/len(train_loader),
                        epoch)
    
    writer.add_scalar('Training mIOU',
                        total_TI/len(train_loader),
                        epoch)

    
    model.eval()
    val_loss = 0
    total_TI *= 0
    with tqdm(total=n_val, desc=f'Validation', unit='img') as pbar2:
        for i, batch in enumerate(val_loader, 1):
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            pred_mask = model(imgs)
            loss_m = criterion_m(pred_mask, true_masks)
            loss_kl = criterion_kl(model)
            loss = loss_m + kl*loss_kl
            val_loss += loss.item()
            total_TI += (1 - get_TI(pred=pred_mask, true=true_masks, alpha=1, beta=1, smooth=0, gamma=1).item())

            pbar2.set_postfix(**{'Avg Val_loss': loss.item()})

            pbar2.update(imgs.shape[0])
    writer.add_scalar('val loss',
                            val_loss/len(val_loader),
                            epoch)

    writer.add_scalar('Validation mIOU',
                        total_TI/len(train_loader),
                        epoch)            
    
    writer.add_image("input image", imgs, dataformats='NCHW', global_step=epoch)
    writer.add_image("predicted mask", torch.unsqueeze(pred_mask[:, 1, :, :], 1), dataformats='NCHW', global_step=epoch)
    writer.add_image("True mask", torch.unsqueeze(true_masks[:, 1, :, :], 1), dataformats='NCHW', global_step=epoch)

    # scheduler.step(val_loss/n_val)
    torch.save(model.state_dict(), cp_dir + f'model_ep{str(epoch)}_{str(val_loss/len(val_loader))}.pth')
    print(f"\nSaved Checkpoint. Val_loss: {val_loss/len(val_loader)}. Train_loss: {epoch_loss/len(train_loader)}")

torch.save(model.state_dict(), cp_dir + f'Final_model.pth')
print("Done")