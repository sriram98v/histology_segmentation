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
    LR = config["lr"]

    BILINEAR = config["model"]["bilinear"]
    OUT_LAYER = config["model"]["out_layer"]
    CRF = config["model"]["crf"]
    
    LOSS = config["loss"]["name"]
    LOSS_KWARGS = config["loss"]["kwargs"]
    
    EVAL_METRIC = config["eval_metric"]["name"]
    EVAL_KWARGS = config["eval_metric"]["kwargs"]

    DS_NAME = config["dataset"]["name"]
    ROOT_DIR = config["dataset"]["root_dir"]
    DISPLAY_IDXS = config["dataset"]["display_idxs"]
    TRAIN_BATCH_SIZE = config["dataset"]["train_batch_size"]
    TEST_BATCH_SIZE = config["dataset"]["test_batch_size"]


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

    train_set = DS(DS_NAME, ROOT_DIR, split='train', transforms=[norm_im(), Resize(size=(256, 256)), Rotate(180, fill=1)])

    test_set = DS(DS_NAME, ROOT_DIR, split='val', transforms=[norm_im(), Resize(size=(256, 256))])

    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)

    model = Frequentist_UNet(3, len(train_set.classes), classes=train_set.classes, bilinear=BILINEAR, out_layer=OUT_LAYER)
    model.to(device=DEVICE)
    criterion_m = Loss(LOSS, **LOSS_KWARGS)
    criterion_acc = Metric(EVAL_METRIC, **EVAL_KWARGS)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        epoch_loss = 0
        kl = 0
        mIoU = 0
        
        with tqdm(total=len(train_loader), desc=f'Training Epoch {epoch}/{EPOCHS}') as pbar:
            model.train()
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
                print(imgs)

        test_acc = evaluate(model, dataloader=test_loader, device=DEVICE, metric=criterion_acc)
        
        idx = random.randint(0, len(test_set)-1)
        # print("test ", idx, len(test_set)-1)
        writer_im_test, writer_tar_test = test_set[idx]
        writer_preds_test = torch.squeeze(model(torch.unsqueeze(writer_im_test.to(device=DEVICE, dtype=torch.float32), dim=0)))

        idx = random.randint(0, len(train_set)-1)
        # print("train ", idx, len(train_set))
        writer_im_train, writer_tar_train = train_set[idx]
        writer_preds_train = torch.squeeze(model(torch.unsqueeze(writer_im_train.to(device=DEVICE, dtype=torch.float32), dim=0)))


        writer.add_scalar(f"Training Loss backbone",
                            epoch_loss/len(train_loader),
                            epoch)
        
        writer.add_scalar(f'KL Divergence backbone',
                            kl/len(train_loader),
                            epoch)
        
        writer.add_scalars(f'mIOU backbone',
                            {"Train": mIoU/len(train_loader), 
                             "Test": test_acc},
                            epoch)
        
        writer.add_images("images", imgs, global_step=epoch)
        writer.add_images("gt", true_masks, global_step=epoch)
        writer.add_images("preds", pred_masks, global_step=epoch)


        torch.save(model.state_dict(), cp_dir + f'model_ep_backbone{str(epoch)}.pth')
        print(f"\nSaved Checkpoint. Train_loss_backbone: {epoch_loss/len(train_loader):.3f}")

    torch.save(model.state_dict(), cp_dir + f'Final_model_backbone.pth')

if __name__=="__main__":
    args = get_args()
    config = parse_config(args.config)
    train(config)