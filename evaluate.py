import torch
from tqdm import tqdm
import os
from BayesianSeg.models import *
from BayesianSeg.metrics import Metric
from BayesianSeg.datasets.histology_dataset import histologyDataset
from torch.utils.data import DataLoader
from BayesianSeg.datasets.augs import *
from utils import get_args, parse_config

def evaluate(model, dataloader, device, metric):
    num_batches = len(dataloader)
    total_score = 0

    # iterate over the validation set
    with tqdm(total=len(dataloader), desc=f'Testing') as pbar:
        model.eval()
        for batch in dataloader:
            imgs = batch[0].to(device=device, dtype=torch.float32)
            true_masks = batch[1].to(device=device, dtype=torch.float32)
            pred_mask  = model(imgs)

            score = metric(pred=pred_mask, true=true_masks).item()
            total_score += score

            pbar.update()
            pbar.set_postfix(**{f"{metric}": score})

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

    accuracy = evaluate(model=model, dataloader=test_loader, device=DEVICE, metric=Metric(EVAL_METRIC, **EVAL_KWARGS))

    print(f"Accuracy: {accuracy}")