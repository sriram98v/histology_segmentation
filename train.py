import torch
from BayesianSeg.models.bayes_unet import *
from BayesianSeg.loss import Loss
from BayesianSeg.metrics import Metric
from BayesianSeg.datasets.histology_dataset import histologyDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import torchvision
from BayesianSeg.datasets.augs import *
import os
from utils import get_args, parse_config
from evaluate import evaluate

def train(config):    
    EPOCHS = config["epochs"]
    TRAIN_BATCH_SIZE = config["dataset"]["train_batch_size"]
    TEST_BATCH_SIZE = config["dataset"]["test_batch_size"]
    LR = config["lr"]
    KL_WEIGHT = config["kl_weight"]
    PRIOR_MU = config["model"]["prior_mu"]
    PRIOR_SIGMA = config["model"]["prior_sigma"]
    BILINEAR = config["model"]["bilinear"]
    
    LOSS = config["loss"]["name"]
    LOSS_KWARGS = config["loss"]["kwargs"]
    LOSS_KL = config["div"]["name"]
    LOSS_KL_KWARGS = config["div"]["kwargs"]
    
    EVAL_METRIC = config["eval_metric"]["name"]
    EVAL_KWARGS = config["eval_metric"]["kwargs"]

    TRAIN_PATH = os.path.join(config["dataset"]["dataset_dir"], "train")
    TEST_PATH = os.path.join(config["dataset"]["dataset_dir"], "test")
    CLASSES = config["dataset"]["classes"] if len(config["dataset"]["classes"])>0 else os.listdir(os.path.join(TRAIN_PATH, "GT"))
    NUM_CLASSES = len(CLASSES)

    if not os.path.exists(config["cp_dir"]):
        os.makedirs(config["cp_dir"])
        print("Created logs directory at "+config["cp_dir"])
    cp_dir = config["cp_dir"]

    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
        print("Created logs directory at "+config["log_dir"])

    writer=SummaryWriter(config["log_dir"])

    try:
        DEVICE = torch.device(config["device"])
    except:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kwargs = {'num_workers': config["dataset"]["num_workers"], 'pin_memory': True} if 'cuda' in DEVICE.type  else {'num_workers': config["dataset"]["num_workers"], 'pin_memory': False}

    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])
        print("Created logs directory at "+config["log_dir"])
    
    train_set = histologyDataset(os.path.join(TRAIN_PATH, "images"), os.path.join(TRAIN_PATH, "GT"),
                            color=True, transform=BSCompose([Rotate(), ToTensor(), Resize(size=(256, 256))]))
    test_set = histologyDataset(os.path.join(TEST_PATH, "images"), os.path.join(TEST_PATH, "GT"),
                            color=True, transform=BSCompose([ToTensor(), Resize(size=(256, 256))]))

    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

    model = Bayesian_UNet(PRIOR_MU, PRIOR_SIGMA, 3, NUM_CLASSES, classes=CLASSES, bilinear=BILINEAR)
    model.to(device=DEVICE)
    criterion_m = Loss(LOSS, DEVICE, **LOSS_KWARGS)
    criterion_kl = Loss(LOSS_KL, DEVICE, **LOSS_KL_KWARGS)
    criterion_acc = Metric(EVAL_METRIC, **EVAL_KWARGS)
    kl_weight = KL_WEIGHT
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        epoch_loss = 0
        kl = 0
        mIoU = 0
        
        with tqdm(total=len(train_loader), desc=f'Training Epoch {epoch + 1}/{EPOCHS}') as pbar:
            model.train()
            for i, batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                imgs = batch[0].to(device=DEVICE, dtype=torch.float32)
                true_masks = batch[1].to(device=DEVICE, dtype=torch.float32)
                pred_mask  = model(imgs)
                if kl_weight==0:
                    loss_kl = torch.Tensor([0]).to(device=DEVICE, dtype=torch.float32)
                else:
                    loss_kl = criterion_kl(model)
                loss_m = criterion_m(pred_mask, true_masks)
                loss = loss_m + kl_weight*loss_kl
                mIoU += criterion_acc(pred=pred_mask, true=true_masks).item()
                epoch_loss += loss.item()
                kl += loss_kl.item()

                loss.backward()
                optimizer.step()

                pbar.update()
                pbar.set_postfix(**{f"{criterion_m}": loss_m.item(), f"{criterion_kl}": loss_kl.item(), f"Total_loss": loss.item()})

        test_acc = evaluate(model, dataloader=test_loader, device=DEVICE, metric=criterion_acc)

        writer.add_scalar(f"Training Loss kl_weight={kl_weight}",
                            epoch_loss/len(train_loader),
                            epoch)
        
        writer.add_scalar(f'KL Divergence kl_weight={kl_weight}',
                            kl/len(train_loader),
                            epoch)
        
        writer.add_scalar(f'Training mIOU kl_weight={kl_weight}',
                            mIoU/len(train_loader),
                            epoch)
        
        writer.add_scalar(f"Test Accuracy m{criterion_acc}",
                            test_acc,
                            epoch)
        
        writer.add_image("input image", imgs[0, :, :, :], dataformats='CHW', global_step=epoch)
        writer.add_image("predicted mask", pred_mask[0, 1, :, :], dataformats='HW', global_step=epoch)
        writer.add_image("True mask", true_masks[0, 1, :, :], dataformats='HW', global_step=epoch)

        torch.save(model.state_dict(), cp_dir + f'model_ep{str(epoch)}.pth')
        print(f"\nSaved Checkpoint. Train_loss: {epoch_loss/len(train_loader):.3f}")

    torch.save(model.state_dict(), cp_dir + f'Final_model.pth')
    print("Done")

if __name__=="__main__":
    args = get_args()
    config = parse_config(args.config)
    train(config)