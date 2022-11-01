import torch
from BayesianSeg.loss.losses import *
from BayesianSeg.datasets.kvasir_dataset import kvasirDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from BayesianSeg.models.bayes_unet import *
from torch import optim
from tqdm import tqdm
from torchvision import transforms
from BayesianSeg.datasets.augs import *
import os
import math
import random
import matplotlib.pyplot as plt
from BayesianSeg.metrics.metrics import IoU, get_entropies


LABEL_PERCENT = 0.05
VAL_PERCENT = 0.2
EPOCHS = 300
BATCH_SIZE = 16
LR = 0.001
DP = 0.5

IMAGES = os.listdir(os.path.join("./Kvasir-SEG/", "images"))[:500]
CLASSES = ['polyp']

random.shuffle(IMAGES)

IMAGES = IMAGES[:int(DP*len(IMAGES))]

n_train = int(len(IMAGES) * (1-VAL_PERCENT))
n_label = int(n_train * LABEL_PERCENT)
n_unlabel = len(IMAGES) - n_label

val_set = IMAGES[n_train:]
train_set = IMAGES[:n_train]
label_set = train_set[:n_label]
unlabel_set = train_set[n_label:]

val_dataset = kvasirDataset("./Kvasir-SEG/images/", "./Kvasir-SEG/masks/", color=True, 
                            transform=transforms.Compose([Rotate(), ToTensor(), Resize(size=(256, 256))]),
                            classes=['polyp'], im_names=val_set)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

cp_dir = "./checkpoints/"
writer=SummaryWriter('content/logsdir')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

active_batch_size = int(len(IMAGES) * LABEL_PERCENT)

def train_model(label_set, active_epoch):
    model = Bayesian_UNet(3, label_set.num_classes, classes=label_set.classes)
    model.to(device=device)
    criterion_m = DiceBCELoss()
    criterion_kl = BKLLoss(last_layer_only=False)
    kl_weight = 0.1
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(label_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model.train()
    TIs = []
    for epoch in range(EPOCHS):
        epoch_loss = 0
        kl = 0
        total_TI = 0

        
        with tqdm(total=n_label, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
            for i, batch in enumerate(train_loader, 1):
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)
                pred_mask  = model(imgs)
                loss_m = criterion_m(pred_mask, true_masks)
                loss_kl = criterion_kl(model)
                loss = loss_m + kl_weight*loss_kl
                total_TI += IoU(pred=pred_mask, true=true_masks).item()
                epoch_loss += loss.item()
                kl += loss_kl

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
        TIs.append(total_TI/len(train_loader))
    
    fig, axs = plt.subplots(1, 1)
    axs.plot(list(range(EPOCHS)), TIs)
    writer.add_figure("Active epoch", fig, global_step=active_epoch)

    return model

def validate_model(model, active_epoch=None):
    model.eval()
    total_TI = 0
    with tqdm(total=len(val_loader), desc=f'Validation', unit='img') as pbar2:
        for i, batch in enumerate(val_loader, 1):
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            pred_mask = model(imgs)
            total_TI += IoU(pred=pred_mask, true=true_masks).item()

            pbar2.set_postfix(**{'Avg Tversky index': total_TI/i})
    
    writer.add_image("input image", imgs, dataformats='NCHW', global_step=active_epoch)
    writer.add_image("predicted mask", pred_mask, dataformats='NCHW', global_step=active_epoch)
    writer.add_image("True mask", torch.unsqueeze(true_masks, 1), dataformats='NCHW', global_step=active_epoch)

    return total_TI/len(val_dataset)

def sample_images(model, unlabel_set, k=10, num_iter=30):

    new_ims = []
    model.eval()

    for i in tqdm(range(len(unlabel_dataset)), desc=f'Sampling'):
        im_name = unlabel_set.im_names[i]
        preds = []
        for _ in range(num_iter):
            im = unlabel_set[i]['image']
            out = torch.squeeze(model(torch.unsqueeze(im, dim=0).to(device=device, dtype=torch.float32)))
            preds.append(torch.nn.functional.softmax(out, dim=0).detach().cpu())
        E = get_entropies(preds, mode="smart")
        new_ims.append((im_name, E.item()))
    
    new_ims.sort(key = lambda x: x[1], reverse=True)

    return new_ims[:k]    

n = 0
while(len(unlabel_set)>0):
    label_dataset = kvasirDataset("./Kvasir-SEG/images/", "./Kvasir-SEG/masks/", color=True, 
                            transform=transforms.Compose([Rotate(), ToTensor(), Resize(size=(256, 256))]),
                            classes=['polyp'], im_names=label_set)
    model = train_model(label_dataset, n)
    performance = validate_model(model, n)
    torch.save(model.state_dict(), cp_dir + str(len(label_set)/len(IMAGES)*100) + f'.pth')
    unlabel_dataset = kvasirDataset("./Kvasir-SEG/images/", "./Kvasir-SEG/masks/", color=True, 
                            transform=transforms.Compose([Rotate(), ToTensor(), Resize(size=(256, 256))]),
                                                        classes=['polyp'], im_names=unlabel_set)
    torch.cuda.empty_cache()
    new_ims = sample_images(model, unlabel_dataset, k=math.floor(len(IMAGES) * LABEL_PERCENT))
    for i in new_ims:
        label_set.append(i[0])
        unlabel_set.remove(i[0])

    writer.add_scalar('Validation mIOU',
                    performance,
                    n)
    n+=1


torch.save(model.state_dict(), cp_dir + f'Final_model.pth')
print("Done")


