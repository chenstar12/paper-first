#!/bin/bash

echo 'start train .......'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --inference='trans-PDA' --lambda2=0.001 --lambda1C=0.05 --lambda2C=0.001
echo 'start train .......'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --inference='trans-PDA' --lambda2=0.005 --lambda1C=0.05 --lambda2C=0.01
