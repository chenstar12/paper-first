import pandas as pd
import numpy as np

user = np.random.randint(1, 20, (50))
item = np.random.randint(1000, 2000, (50))

df = pd.DataFrame({'user': pd.Series(user),
                   'item': pd.Series(item)})
# print(df.groupby('user').groups)  # 查看分组

df_cnt = df.groupby('user').count()
print(df_cnt[df_cnt['item'] >= 2])  # 这些user有2个以上交互item
