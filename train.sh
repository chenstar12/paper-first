#!/bin/bash

python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.4 --inference='trans-PD1'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.7 --inference='trans-PD1'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1=2.1 --inference='trans-PD1'
