#!/bin/bash

python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.01 --inference='trans-PD1'
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.05 --inference='trans-PD1'
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.1 --inference='trans-PD1'
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.5 --inference='trans-PD1'
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.9 --inference='trans-PD1'
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.5 --inference='trans-PD1'

python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.01
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.05
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.1
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.5
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.9
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.5
