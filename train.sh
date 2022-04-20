#!/bin/bash

echo 'sigmoid added'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.3 --inference='trans-PD1'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.4 --inference='trans-PD1'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.5 --inference='trans-PD1'

python main.py train --model=MSCI0D0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.4 --inference='trans-PD1'
python main.py train --model=MSCI0D0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.4 --inference='trans-PD'
