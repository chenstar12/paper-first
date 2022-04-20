#!/bin/bash

python main.py train --model=MSCI0D1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.1 --inference='trans-PD1'
python main.py train --model=MSCI0D1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.3 --inference='trans-PD1'
python main.py train --model=MSCI0D1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.5 --inference='trans-PD1'
