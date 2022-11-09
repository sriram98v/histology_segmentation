# SmartHisto #
SmartHisto is a Deep Bayesian Neural Network Framework that can be used to
- Train a bayesian model on annotated histopathology images.
- Identify structures on histopathology slices.
- Interact with pathologists in order to reduce annotation efforts during dataset aggregation.

## Installation ##
SmartHisto can be installed via pip
```bash
pip install smarthisto
```

Alternatively, SmartHisto can be installed from source by cloning this repository and running

```bash
python setup.py install
```

## Training a new model ##
This section describes how to train a new model 

In order to make predicitons you first need to train a model on your local dataset. Your training dataset must be restrucutured as follows:
```
└── Dataset
    ├── GT
    │   ├── class 1
    │   │   ├── 0.png
    │   │   ├── 1000200.png
    │   │   ├── 1000500.png
    │   │   └── 1000800.png
    │   ├── class 2
    │   ├── class 3
    │   ├── class 4
    │   ├── class 5
    .   .
    .   .
    └── images
        ├── 0.png
        ├── 1000200.png
        ├── 1000500.png
        └── 1000800.png

```
Each of the images in the ```GT``` directory is a binary image with true marked on the corresponding pixels. 

With this directory structure, you can use the provided dataloaders. Custom datasets are described in the further below

## Make predictions ##
To make predictions, run predict.py with the arguments as directed in the help manual. You can see the help manual by running:
```bash
python predict,py --help
```

For example:
```bash
python predict.py -m ./Final_model.pth -i test_images/ -d cuda
```

Note: You will need to add your own pretrained model. Training script will be modified for the repository shortly.