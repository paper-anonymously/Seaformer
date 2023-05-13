# Seaformer

A PyTorch implementation of our Seaformer

## Dependencies
Install the dependencies via [Anaconda](https://www.anaconda.com/):
+ Python (>=3.8)
+ PyTorch (>=1.8.1)
+ NumPy (>=1.17.4)
+ Scipy (>=1.7.3)
+ torch-geometric(>=2.0.4)
+ tqdm(>=4.62.2)
+ sentence_transformers
+ towhee
+ json

## Dataset
Three datasets (i.e., [SMPD](https://smp-challenge.com/download.html), [ICIP](https://iplab.dmi.unict.it/popularitydataset/SIPD2020CHALLENGE/train/), [WeChat](https://algo.weixin.qq.com/2021/problem-description)) can be downloaded from official website address.


create virtual environment:
```
conda create --name Seaformer python=3.9
```

activate environment:
```
conda activate Seaformer
```

install pytorh from [pytorch](https://pytorch.org/get-started/previous-versions/):
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
```

To install all dependencies:
```
pip install -r requirements.txt
```

## Usage
Here we provide the implementation of Seaformer along with SMPD dataset.

+ To train and evaluate on SMPD:
```
python run.py -data_name=SMP
```
More running options are described in the codes, e.g., `-data_name= SMP`

## Folder Structure

Seaformer

```     
└── models: # The file includes each part of the modules in Seaformer.
    ├── HGAT.py # The core source code of BHT.
    ├── mmmodels.py # The core source code of Seaformer.
    ├── TransformerBlock.py # The core source code of multi-head attention.

└── utils: # The file includes each part of basic modules (e.g., metrics, earlystopping).
    ├── EarlyStopping.py  # The core code of the early stopping operation.
    ├── img_text_embedding.py # The core source code of visual (textual) feature extraction
    ├── parsers.py        # The core source code of parameter settings. 
└── dataLoader.py:     # Data loading.
└── run.py:            # Run the model.
```
