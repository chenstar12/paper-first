#!/bin/bash

python main.py train --model=MSCI0D1T1 --num_fea=2  --dataset=Gourmet_Food_data --lambda1=0.1 --inference=trans-tanh --batch_size=1024
python main.py train --model=MSCI0D1T3 --num_fea=2  --dataset=Gourmet_Food_data --lambda1=0.1 --inference=trans-tanh --batch_size=1024
python main.py train --model=MSCI0D1T5 --num_fea=2  --dataset=Gourmet_Food_data --lambda1=0.1 --inference=trans-tanh --batch_size=1024
python main.py train --model=MSCI0D1T5B --num_fea=2  --dataset=Gourmet_Food_data --lambda1=0.1 --inference=trans-tanh --batch_size=1024

python main.py train --model=MSCI0D1T1 --num_fea=2 --dataset=Gourmet_Food_data  --inference='' --ei='' --batch_size=1024
python main.py train --model=MSCI0D1T3 --num_fea=2 --dataset=Gourmet_Food_data  --inference='' --ei='' --batch_size=1024
python main.py train --model=MSCI0D1T5 --num_fea=2 --dataset=Gourmet_Food_data  --inference='' --ei='' --batch_size=1024
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data  --inference='' --ei='' --batch_size=1024