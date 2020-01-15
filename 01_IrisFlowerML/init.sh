#!/bin/bash

# create folder if not exists
mkdir -p data

# download data set
wget -O data/iris.data https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data