#!/bin/bash

python main.py train --model=MSCI0D1T5 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.01
python main.py train --model=MSCI0D1T5A --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.01
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.01
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.01 --inference=tanh

python main.py train --model=MSCI0D1T5 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.05
python main.py train --model=MSCI0D1T5A --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.05
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.05
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.05 --inference=tanh

python main.py train --model=MSCI0D1T5 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.1
python main.py train --model=MSCI0D1T5A --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.1
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.1
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.1 --inference=tanh

python main.py train --model=MSCI0D1T5 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.3
python main.py train --model=MSCI0D1T5A --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.3
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.3
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.3 --inference=tanh

python main.py train --model=MSCI0D1T5 --num_fea=2 --dataset=Gourmet_Food_data  --inference='' --ei=''
python main.py train --model=MSCI0D1T5A --num_fea=2 --dataset=Gourmet_Food_data  --inference='' --ei=''
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data  --inference='' --ei=''


python main.py train --model=MSCI0D1T5 --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.5
python main.py train --model=MSCI0D1T5A --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.5
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.5
python main.py train --model=MSCI0D1T5B --num_fea=2 --dataset=Gourmet_Food_data --lambda1=0.5 --inference=tanh
