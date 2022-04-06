# 修改output：
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=mlp --dataset=Video_Games_data

python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=fm --dataset=Video_Games_data

python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=lfm --dataset=Video_Games_data

python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=nfm --dataset=Video_Games_data

# batch_size
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=mlp --dataset=Video_Games_data --batch_size=256

python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=mlp --dataset=Video_Games_data --batch_size=512

# ......