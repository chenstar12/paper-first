#!/bin/bash

python main.py train --model=MSCI0D --num_fea=2 --dataset=Kindle_Store_data --lambda1=0.5 --inference='trans-PD1'
