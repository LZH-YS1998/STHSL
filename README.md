# STHSL
A pytorch implementation for the paper:<br />
Spatial-Temporal Hypergraph Self-Supervised Learning for Crime Prediction.<br />
In **ICDE 2022**.

# Introduction
Spatial-Temporal Hypergraph Self-Supervised Learning for Crime Prediction (STHSL) is a spatio-temporal prediction networks. By adding self-supervised learning methods as auxiliary tasks, STHSL can tackle the label scarcity issue in crime prediction.

# Structure
* Datasets: Including NYC and CHI datasets used in our experiments, which are released by and available at STSHN.
* Save: Including the save files of NYC and CHI datasets, using for testing.
* DataHandler: 
* model: 

# Environment Requirement
The code can be run in the following environments, other version of required packages may also work.
* python==3.9.7
* numpy==1.22.3
* pytorch==1.9.0

# Run the codes 
* NYC-Crimes dataset
```
python train.py --data NYC
```

* Chicago-Crimes dataset
```
python train.py --data CHI
```
