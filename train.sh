#!/bin/bash

python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.1 --inference='trans-PD1'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.6 --inference='trans-PD1'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.8 --inference='trans-PD1'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.1 --inference='trans-PD1'
