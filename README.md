# STHSL
A pytorch implementation for the paper:
Spatial-Temporal Hypergraph Self-Supervised Learning for Crime Prediction

# Introduction
Spatial-Temporal Hypergraph Self-Supervised Learning for Crime Prediction (STHSL) is a spatio-temporal prediction networks. By adding self-supervised learning methods as auxiliary tasks, STHSL can tackle the label scarcity issue in crime prediction.

# Environment Requirement
The code can be run in the following environments, other version of required packages may also work.
* python==3.9.7
* numpy==1.22.3
* pytorch==1.9.0

# Run the codes 
* NYC-Crimes dataseet
```
python train.py --data NYC
\```Cancel changes
* Chicago-Crimes dataseet
```
python train.py --data CHI
\```
