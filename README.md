# Lightweight and Fast Time-Series Anomaly Detection via Point-Level and Sequence-Level Reconstruction Discrepancy (TNNLS 2025)
This repository contains the adapted implementation for the LFTSAD ([paper](https://ieeexplore.ieee.org/document/11006753)) for using the ESA-ADB dataset and its associated metrics.

## Framework
<img src="https://github.com/infogroup502/LFTSAD/blob/main/img/workflow.png" width="850px">

## Requirements
Create a new COnda environment and run 
```bash 
conda create --name LFTSAD --file spec-file.txt
```

## Data 
Place the ESA-MissionX and preprocessed folders obtained from running the ESA-ADB preprocessing scripts into the data folder. 


## Code Description
There are six files/folders in the source
- data_factory: The preprocessing folder/file. All datasets preprocessing codes are here.
esa_data_loader.py loads data for the ESA experiments.
- esa_main.py: The main python file. You can adjustment all parameters in there.
- esa_metrics: Contains code to calculate the metrics for ESA-ADB
- model: LFTSAD model folder
- esa_solver_complete.py: Another python file. The training, validation, and testing processing are all in there
- environment.yml: Python packages needed to run this repo
## Usage
1. Install packages using conda env create -f environment.yml
2. Activate environment using conda activate LFTSAD
3. Download the datasets and place such that dataset x_months.test and x_months.train for Mission Y as shown in the image<img width="441" height="514" alt="image" src="https://github.com/user-attachments/assets/77060afc-ac64-41e9-9de2-6aa489f21952" />

4. To train and evaluate LFTSAD on a dataset, for example on 3 month data, run the following command:
```bash
python esa_main.py --dataset 3_months
```

## BibTex Citation
```bash
@ARTICLE{11006753,
  author={Chen, Lei and Tang, Jiajun and Zou, Ying and Liu, Xuxin and Xie, Xingquan and Deng, Guangyang},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Lightweight and Fast Time-Series Anomaly Detection via Point-Level and Sequence-Level Reconstruction Discrepancy}, 
  year={2025},
  volume={36},
  number={9},
  pages={17295-17309},
  keywords={Accuracy;Computer architecture;Computational modeling;Anomaly detection;Image edge detection;Training;Learning systems;Fourth Industrial Revolution;Time series analysis;Real-time systems;All-multilayer perceptron network (MLP)-based anomaly detection;lightweight network and design;point-level and sequence-level;reconstruction discrepancy;time-series anomaly detection (TSAD)},
  doi={10.1109/TNNLS.2025.3565807}}
```
