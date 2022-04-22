#!/bin/bash

echo '消融实验-00/case study ----- 不re-weighting,但保留inference（看看是否是个性化的sentiment带来的性能提升！！！）'

echo '消融实验-01 ---- 保留训练re-weighting，但推理时直接输出，即inference参数为空'
python main.py train --inference=''

# re-weighting只用polarity；注意：只做一次实验，参数和已有最优实验一致（--model=MSCI0D1，--inference=trans-PD1,）
echo '消融实验-02'
python main.py train --model=MSCI0D1 --inference=trans-PD --lambda=0.5



# re-weighting只用subjectiviy(没意义；仅有主观性，代表不了任何情感倾向)

