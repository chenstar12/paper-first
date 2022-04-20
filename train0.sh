#!/bin/bash

python main.py train --model=MSCI0D0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.1 --inference='trans-PD'
python main.py train --model=MSCI0D0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.4 --inference='trans-PD'
python main.py train --model=MSCI0D0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.6 --inference='trans-PD'
