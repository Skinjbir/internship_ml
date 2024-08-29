#!/bin/sh
ls -lR
# Run data.py
python3 data_stage/data.py

# Run train.py
python3 train_stage/train.py
