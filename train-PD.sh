#!/bin/bash

echo 'start train .......'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.1 --inference='trans-PD' --lambda1C=0.05 --lambda2C=0.01
echo 'start train .......'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.4 --inference='trans-PD' --lambda1C=0.2 --lambda2C=0.01