#!/bin/bash

for i in `seq 1 20`; do
  echo $i
  echo 'start train .......'
  # python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda2=0.001 --inference=PDA
done