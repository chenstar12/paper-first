#!/bin/bash

python main.py train --model=MSCI0D0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.2 --inference='trans-PD'
python main.py train --model=MSCI0D0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.4 --inference='trans-PD'
python main.py train --model=MSCI0D0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.6 --inference='trans-PD'
python main.py train --model=MSCI0D0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.8 --inference='trans-PD'
