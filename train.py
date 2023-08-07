import torch
from BayesianSeg.models import *
from BayesianSeg.loss import Loss
from BayesianSeg.metrics import Metric
from BayesianSeg.datasets import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from BayesianSeg.datasets.augs import *
from torchvision.utils import draw_segmentation_masks
import os
from utils import get_args, parse_config
from evaluate import evaluate
import random

def train(config):    
    EPOCHS = config["epochs"]
    TRAIN_BATCH_SIZE = config["dataset"]["train_batch_size"]
    TEST_BATCH_SIZE = config["dataset"]["test_batch_size"]
    LR = config["lr"]
    KL_WEIGHT = config["kl_weight"]
    PRIOR_MU = config["model"]["prior_mu"]
    PRIOR_SIGMA = config["model"]["prior_sigma"]
    BILINEAR = config["model"]["bilinear"]
    OUT_LAYER = config["model"]["out_layer"]
    
    LOSS = config["loss"]["name"]
    LOSS_KWARGS = config["loss"]["kwargs"]
    LOSS_KL = config["div"]["name"]
    LOSS_KL_KWARGS = config["div"]["kwargs"]
    
    EVAL_METRIC = config["eval_metric"]["name"]
    EVAL_KWARGS = config["eval_metric"]["kwargs"]

    DS_NAME = config["dataset"]["name"]
    ROOT_DIR = config["dataset"]["root_dir"]
    DISPLAY_IDXS = config["dataset"]["display_idxs"]

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

    # train_set = DS(DS_NAME, ROOT_DIR, split='train', transforms=[Resize(size=(256, 256)), norm_im(), Rotate(180, fill=1)])

    # test_set = DS(DS_NAME, ROOT_DIR, split='val', transforms=[Resize(size=(256, 256)), norm_im()])

    train_set = DS("cityscapes", "cityscapes/", split='train', transforms=[PIL_to_tensor(), Resize(size=(256, 256)), class_to_channel(35), norm_im()])#, Rotate(180, fill=1)])

    test_set = DS("cityscapes", "cityscapes/", split='val', transforms=[PIL_to_tensor(), Resize(size=(256, 256)), class_to_channel(35)])

    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)

    model = Bayesian_UNet(PRIOR_MU, PRIOR_SIGMA, 3, len(train_set.classes), classes=train_set.classes, bilinear=BILINEAR, out_layer=OUT_LAYER)
    model.to(device=DEVICE)
    criterion_m = Loss(LOSS, DEVICE, **LOSS_KWARGS)
    criterion_kl = Loss(LOSS_KL, DEVICE, **LOSS_KL_KWARGS)
    criterion_acc = Metric(EVAL_METRIC, **EVAL_KWARGS)
    kl_weight = KL_WEIGHT
    optimizer = optim.Adam(model.backbone.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        epoch_loss = 0
        kl = 0
        mIoU = 0
        
        with tqdm(total=len(train_loader), desc=f'Training Epoch {epoch}/{EPOCHS}') as pbar:
            model.train_backbone()
            for i, batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                imgs = batch[0].to(device=DEVICE, dtype=torch.float32)
                true_masks = batch[1].to(device=DEVICE, dtype=torch.float32)
                pred_masks  = model(imgs)
                if kl_weight==0:
                    loss_kl = torch.Tensor([0]).to(device=DEVICE, dtype=torch.float32)
                else:
                    loss_kl = criterion_kl(model)
                loss_m = criterion_m(pred_masks, true_masks)
                loss = loss_m + kl_weight*loss_kl
                mIoU += criterion_acc(pred=pred_masks, true=true_masks).item()
                epoch_loss += loss.item()
                kl += loss_kl.item()

                loss.backward()
                optimizer.step()

                pbar.update()
                pbar.set_postfix(**{f"{criterion_m}": loss_m.item(), f"{criterion_kl}": loss_kl.item(), f"Total_loss": loss.item()})

        test_acc = evaluate(model, dataloader=test_loader, device=DEVICE, metric=criterion_acc)
        
        idx = random.randint(0, len(test_set)-1)
        writer_im_test, writer_tar_test = test_set[idx]
        writer_preds_test = torch.squeeze(model(torch.unsqueeze(writer_im_test.to(device=DEVICE, dtype=torch.float32), dim=0)))

        idx = random.randint(0, len(train_set)-1)
        writer_im_train, writer_tar_train = train_set[idx]
        writer_preds_train = torch.squeeze(model(torch.unsqueeze(writer_im_train.to(device=DEVICE, dtype=torch.float32), dim=0)))


        writer.add_scalar(f"Training Loss backbone kl_weight={kl_weight}",
                            epoch_loss/len(train_loader),
                            epoch)
        
        writer.add_scalar(f'KL Divergence backbone kl_weight={kl_weight}',
                            kl/len(train_loader),
                            epoch)
        
        writer.add_scalars(f'mIOU backbone kl_weight={kl_weight}',
                            {"Train": mIoU/len(train_loader), 
                             "Test": test_acc},
                            epoch)
        
        if DISPLAY_IDXS==None:
            DISPLAY_IDXS = list(range(len(train_set.classes)))
        
        writer_true_image_test = torch.stack([draw_segmentation_masks((writer_im_test*255).to(device="cpu", dtype=torch.uint8), (writer_tar_test[i]>0.5).to(device="cpu")) for i in DISPLAY_IDXS])
        writer_pred_image_test = torch.stack([draw_segmentation_masks((writer_im_test*255).to(device="cpu", dtype=torch.uint8), (writer_preds_test[i]>0.5).to(device="cpu")) for i in DISPLAY_IDXS])
        writer_test = torchvision.utils.make_grid(torch.cat([writer_true_image_test, writer_pred_image_test], dim=-1), nrow=4, padding=10)

        if DISPLAY_IDXS==None:
            DISPLAY_IDXS = list(range(len(train_set.classes)))
        
        writer_true_image_train = torch.stack([draw_segmentation_masks((writer_im_train*255).to(device="cpu", dtype=torch.uint8), (writer_tar_train[i]>0.5).to(device="cpu")) for i in DISPLAY_IDXS])
        writer_pred_image_train = torch.stack([draw_segmentation_masks((writer_im_train*255).to(device="cpu", dtype=torch.uint8), (writer_preds_train[i]>0.5).to(device="cpu")) for i in DISPLAY_IDXS])
        writer_train = torchvision.utils.make_grid(torch.cat([writer_true_image_train, writer_pred_image_train], dim=-1), nrow=4, padding=10)
        

        writer.add_image("test", writer_test, global_step=epoch)
        writer.add_image("train", writer_train, global_step=epoch)
        writer.add_images("Raw Preds", torch.unsqueeze(writer_preds_test, dim=1), global_step=epoch)


    optimizer = optim.Adam(model.crf.parameters(), lr=LR)
    criterion_m = Loss(LOSS, DEVICE, **LOSS_KWARGS)

    for epoch in range(1, EPOCHS+1):
        epoch_loss = 0
        kl = 0
        mIoU = 0
        
        with tqdm(total=len(train_loader), desc=f'Training Epoch {epoch}/{EPOCHS}') as pbar:
            model.train_crf()
            for i, batch in enumerate(train_loader, 1):
                optimizer.zero_grad()
                imgs = batch[0].to(device=DEVICE, dtype=torch.float32)
                true_masks = batch[1].to(device=DEVICE, dtype=torch.float32)
                pred_masks  = model(imgs)
                loss = criterion_m(pred_masks, true_masks)
                mIoU += criterion_acc(pred=pred_masks, true=true_masks).item()
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                pbar.update()
                pbar.set_postfix(**{f"Total_loss": loss.item()})

        test_acc = evaluate(model, dataloader=test_loader, device=DEVICE, metric=criterion_acc)
        
        idx = random.randint(0, len(test_set)-1)
        writer_im_test, writer_tar_test = test_set[idx]
        writer_preds_test = torch.squeeze(model(torch.unsqueeze(writer_im_test.to(device=DEVICE, dtype=torch.float32), dim=0)))

        idx = random.randint(0, len(train_set)-1)
        writer_im_train, writer_tar_train = train_set[idx]
        writer_preds_train = torch.squeeze(model(torch.unsqueeze(writer_im_train.to(device=DEVICE, dtype=torch.float32), dim=0)))


        writer.add_scalar(f"Training Loss crf kl_weight={kl_weight}",
                            epoch_loss/len(train_loader),
                            epoch)
        
        writer.add_scalar(f'KL Divergence crf kl_weight={kl_weight}',
                            kl/len(train_loader),
                            epoch)
        
        writer.add_scalars(f'mIOU crf kl_weight={kl_weight}',
                            {"Train": mIoU/len(train_loader), 
                             "Test": test_acc},
                            epoch)
        
        if DISPLAY_IDXS==None:
            DISPLAY_IDXS = list(range(len(train_set.classes)))
        
        writer_true_image_test = torch.stack([draw_segmentation_masks((writer_im_test*255).to(device="cpu", dtype=torch.uint8), (writer_tar_test[i]>0.5).to(device="cpu")) for i in DISPLAY_IDXS])
        writer_pred_image_test = torch.stack([draw_segmentation_masks((writer_im_test*255).to(device="cpu", dtype=torch.uint8), (writer_preds_test[i]>0.5).to(device="cpu")) for i in DISPLAY_IDXS])
        writer_test = torchvision.utils.make_grid(torch.cat([writer_true_image_test, writer_pred_image_test], dim=-1), nrow=4, padding=10)

        if DISPLAY_IDXS==None:
            DISPLAY_IDXS = list(range(len(train_set.classes)))
        
        writer_true_image_train = torch.stack([draw_segmentation_masks((writer_im_train*255).to(device="cpu", dtype=torch.uint8), (writer_tar_train[i]>0.5).to(device="cpu")) for i in DISPLAY_IDXS])
        writer_pred_image_train = torch.stack([draw_segmentation_masks((writer_im_train*255).to(device="cpu", dtype=torch.uint8), (writer_preds_train[i]>0.5).to(device="cpu")) for i in DISPLAY_IDXS])
        writer_train = torchvision.utils.make_grid(torch.cat([writer_true_image_train, writer_pred_image_train], dim=-1), nrow=4, padding=10)
        

        writer.add_image("test", writer_test, global_step=epoch)
        writer.add_image("train", writer_train, global_step=epoch)
        writer.add_images("Raw Preds", torch.unsqueeze(writer_preds_test, dim=1), global_step=epoch)

        torch.save(model.state_dict(), cp_dir + f'model_ep{str(epoch)}.pth')
        print(f"\nSaved Checkpoint. Train_loss: {epoch_loss/len(train_loader):.3f}")

    torch.save(model.state_dict(), cp_dir + f'Final_model.pth')
    writer.close()


if __name__=="__main__":
    args = get_args()
    config = parse_config(args.config)
    torch.manual_seed(0)
    train(config)