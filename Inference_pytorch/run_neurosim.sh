#!/bin/bash
python relay_inference.py --model=$1
python run_neurosim.py --model=$1
python summary_neurosim.py --model=$1