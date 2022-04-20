# STHSL
A pytorch implementation for the paper:<br />
Spatial-Temporal Hypergraph Self-Supervised Learning for Crime Prediction.<br />
In ICDE 2022.

# Introduction
Spatial-Temporal Hypergraph Self-Supervised Learning for Crime Prediction (STHSL) is a spatio-temporal prediction networks. By adding self-supervised learning methods as auxiliary tasks, STHSL can tackle the label scarcity issue in crime prediction.

# Structure
* Datasets: including NYC and CHI datasets used in our experiments, which are released by and available at [STSHN](https://github.com/akaxlh/ST-SHN)
* Save: model save path，for testing
* model: model of STHSL
* others: files required for model training 

# Environment Requirement
The code can be run in the following environments, other version of required packages may also work.
* python==3.9.7
* numpy==1.22.3
* pytorch==1.9.0

# Run the codes 
* NYC-Crimes dataset: Train and Test
```
python train.py --data NYC
```
```
python test.py --data NYC --checkpoint ./Save/NYC/your_file_names
```

* Chicago-Crimes dataset: Train and Test
```
python train.py --data CHI
```
```
python test.py --data CHI --checkpoint ./Save/CHI/your_file_names
```
