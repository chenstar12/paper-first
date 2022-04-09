# 修改output：
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=mlp --dataset=Video_Games_data
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=fm --dataset=Video_Games_data
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=lfm --dataset=Video_Games_data
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=nfm --dataset=Video_Games_data

# batch_size
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=mlp --dataset=Video_Games_data --batch_size=64
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=mlp --dataset=Video_Games_data --batch_size=256
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=mlp --dataset=Video_Games_data --batch_size=256 --lr=4e-3
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=mlp --dataset=Video_Games_data --batch_size=512 --lr=4e-3

# 待定
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=mlp --dataset=Video_Games_data --batch_size=64
# 排除 ---- 学习率太小：batch=512; lr=0.001; 目前范围：(0.001, 0.003)
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=mlp --dataset=Video_Games_data --batch_size=512
# 排除 ---- 学习率太大：换算batch=512； lr=0.00456
python main.py train --model=MSCI --num_fea=2 --ui_merge=dot --output=mlp --dataset=Video_Games_data --batch_size=256 --lr=4e-3

