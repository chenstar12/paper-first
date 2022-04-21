#!/bin/bash

echo '消融实验/case study'

# 训练时re-weighting，推理时直接输出，即inference = ''
python main.py train --inference=''

