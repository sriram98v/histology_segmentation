# SmartHisto #
![alt text](proj_im.png)
SmartHisto is a Deep Bayesian Neural Network Framework that can be used to
- Train a bayesian model on annotated histopathology images.
- Identify structures on histopathology slices.
- Interact with pathologists in order to reduce annotation efforts during dataset aggregation.

## Usage ##
You can clone this repository and run the scripts below.

### Prerequisite ###
You will need to install the torchbnn package from the git branch [https://github.com/sriram98v/bayesian-neural-network-pytorch](url). See the README for instructions.

## Training a new model ##
This section describes how to train a new model 

### Dataset ###
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

### Training Config ###
In order to train a model on your dataset, you need to set up a training configuration. A default configuration can be generated by running

```bash
python gen_config.py
```

You can change this configuration file as per your system specifications.

### Training a model ###
You can now train a model by running

```bash
python Activetrain.py -c config.json
```
This will generate two new files which are ```model_config.json``` and ```model.pth```. You will need both in order to make predicitons.

## Make predictions ##
To make predictions, run predict.py with the arguments as directed in the help manual. You can see the help manual by running:

```bash
python predict.py --help
```

For example:
```bash
python predict.py -m ./Final_model.pth -i test_images/ -d cuda -c model_config.json
```

## Thanks to
* @kumar-shridhar [github:PyTorch-BayesianCNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)
* @Harry-24k [github: bayesian-neural-network-pytorch](https://github.com/Harry24k/bayesian-neural-network-pytorch)
