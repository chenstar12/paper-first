#!/bin/bash

python main.py train --model=MSCI0D1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.1 --inference='trans-PD1'
python main.py train --model=MSCI0D1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.3 --inference='trans-PD1'
python main.py train --model=MSCI0D1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.5 --inference='trans-PD1'
python main.py train --model=MSCI0D1 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.7 --inference='trans-PD1'

python main.py train --model=MSCI0D --num_fea=2 --lambda1=0.2 --inference='trans-PD1'
python main.py train --model=MSCI0D0 --num_fea=2 --lambda1=0.2 --inference='trans-PD'

python main.py train --model=MSCI0D --num_fea=2 --lambda1=0.4 --inference='trans-PD1'
python main.py train --model=MSCI0D0 --num_fea=2 --lambda1=0.4 --inference='trans-PD'

python main.py train --model=MSCI0D --num_fea=2 --lambda1=0.6 --inference='trans-PD1'
python main.py train --model=MSCI0D0 --num_fea=2 --lambda1=0.6 --inference='trans-PD'
