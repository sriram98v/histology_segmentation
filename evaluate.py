import torch
from tqdm import tqdm
import os
from BayesianSeg.models import *
from BayesianSeg.metrics import Metric
from BayesianSeg.datasets.histology_dataset import histologyDataset
from torch.utils.data import DataLoader
from BayesianSeg.datasets.augs import *
from BayesianSeg.misc import *
from utils import get_args, parse_config
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def evaluate(model, dataloader, device, metric, crf_out_layer=None, writer=None, step=0, log_str=""):
    num_batches = len(dataloader)
    total_score = 0
    total_score_crf = 0
    num_iter = 0

    # iterate over the validation set
    with tqdm(total=len(dataloader), desc=f'Testing') as pbar:
        model.eval()
        for n,batch in enumerate(dataloader):
            imgs = batch[0].to(device=device, dtype=torch.float32)
            true_masks = batch[1].to(device=device, dtype=torch.float32)
            pred_logits  = model(imgs)
            if isinstance(crf_out_layer, nn.Module):
                pred_logits_crf = crf_out_layer(pred_logits, imgs)
                score = metric(pred=normalize(pred_logits), true=true_masks).item()            
                total_score += score
                score_crf = metric(pred=normalize(pred_logits_crf), true=true_masks).item()
                total_score_crf += score_crf
                pbar.update()
                pbar.set_postfix(**{f"{metric}": total_score/(n+1),
                                    f"{metric}_crf": total_score_crf/(n+1)})
            else:
                score = metric(pred=normalize(pred_logits), true=true_masks).item()            
                total_score += score
                pbar.update()
                pbar.set_postfix(**{f"{metric}": total_score/(n+1)})

            num_iter += 1

    if isinstance(writer, SummaryWriter):
        writer.add_images(f"{log_str}_Validation Input", imgs, step)
        writer.add_images(f"{log_str}_Validation GT", true_masks, step)
        writer.add_images(f"{log_str}_Validation Prediction", normalize(pred_logits), step)
        if isinstance(crf_out_layer, nn.Module):
            writer.add_images(f"{log_str}_Validation Prediction CRF", normalize(pred_logits_crf), step)

    return total_score / max(num_batches, 1)

if __name__=="__main__":
    args = get_args()
    config = parse_config(args.config)

    EPOCHS = config["epochs"]
    TEST_BATCH_SIZE = config["dataset"]["test_batch_size"]
    BILINEAR = config["model"]["bilinear"]
    TRAIN_PATH = os.path.join(config["dataset"]["dataset_dir"], "train")
    TEST_PATH = os.path.join(config["dataset"]["dataset_dir"], "test")
    CLASSES = config["dataset"]["classes"] if len(config["dataset"]["classes"])>0 else os.listdir(os.path.join(TRAIN_PATH, "GT"))
    NUM_CLASSES = len(CLASSES)
    MODEL_PATH = config["model"]["path"]
    PRIOR_MU = config["model"]["prior_mu"]
    PRIOR_SIGMA = config["model"]["prior_sigma"]
    
    EVAL_METRIC = config["eval_metric"]["name"]
    EVAL_KWARGS = config["eval_metric"]["kwargs"]

    try:
        DEVICE = torch.device(config["device"])
    except:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kwargs = {'num_workers': config["dataset"]["num_workers"], 'pin_memory': True} if 'cuda' in DEVICE.type  else {'num_workers': config["dataset"]["num_workers"], 'pin_memory': False}
    
    test_set = histologyDataset(os.path.join(TEST_PATH, "images"), os.path.join(TEST_PATH, "GT"),
                            color=True, transform=BSCompose([Resize(size=(256, 256)), norm_im()]))

    test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)

    model = Bayesian_UNet(PRIOR_MU, PRIOR_SIGMA, 3, NUM_CLASSES, classes=CLASSES, bilinear=BILINEAR)
    model.to(device=DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    accuracy = evaluate(model=model, dataloader=test_loader, device=DEVICE, metric=Metric(EVAL_METRIC, **EVAL_KWARGS), log_str="")

    print(f"Accuracy: {accuracy}")