#!/bin/bash

python main.py train --model=MSCI0D1 --num_fea=2 --dataset=Kindle_Store_data --lambda1=0.4 --inference='trans-PD1'
python main.py train --model=MSCI0D1 --num_fea=2 --dataset=Kindle_Store_data --lambda1=0.5 --inference='trans-PD1'
python main.py train --model=MSCI0D1 --num_fea=2 --dataset=Kindle_Store_data --lambda1=0.6 --inference='trans-PD1'