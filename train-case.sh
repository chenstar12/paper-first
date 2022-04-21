#!/bin/bash

echo '消融实验/case study'

# 训练时re-weighting，推理时直接输出，即inference = ''
echo '消融实验-01'
python main.py train --inference=''

# re-weighting只用polarity；注意：只做一次实验，参数和已有最优实验一致（--model=MSCI0D1，--inference=trans-PD1,）
echo '消融实验-02'

# re-weighting只用subjectiviy(没意义；仅有主观性，代表不了任何情感倾向)

