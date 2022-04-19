#!/bin/bash

echo 'only inference in eval..............................................................'
echo 'start train .......'
python main.py train --model=MSCI0D --num_fea=2 --dataset=Gourmet_Food_data --lambda1C=0.3 --lambda2C=0.01