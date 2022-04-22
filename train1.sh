#!/bin/bash

python main.py train --model=MSCI0D1T0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.01 --inference='trans-PD'
python main.py train --model=MSCI0D1T0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.05 --inference='trans-PD'
python main.py train --model=MSCI0D1T0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.1 --inference='trans-PD'
python main.py train --model=MSCI0D1T0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.5 --inference='trans-PD'
python main.py train --model=MSCI0D1T0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.9 --inference='trans-PD'
python main.py train --model=MSCI0D1T0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.5 --inference='trans-PD'
python main.py train --model=MSCI0D1T0 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=2.1 --inference='trans-PD'

python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.01 --inference='trans-PD'
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.05 --inference='trans-PD'
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.1 --inference='trans-PD'
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.5 --inference='trans-PD'
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.9 --inference='trans-PD'
python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=1.5 --inference='trans-PD'
