#!/bin/bash

echo 'start train .......'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.6 --inference='trans-PD1' --lambda1C=0.3 --lambda2C=0.01
echo 'start train .......'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.8 --inference='trans-PD1' --lambda1C=0.4 --lambda2C=0.01
