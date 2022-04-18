import pandas as pd
import numpy as np

user = np.random.randint(1, 20, (50))
item = np.random.randint(1000, 2000, (50))

df = pd.DataFrame({'user': pd.Series(user),
                   'item': pd.Series(item)})
# print(df.groupby('user').groups)  # 查看分组

df_cnt = df.groupby('user').count()
print(df_cnt[df_cnt['item'] >= 2].index)  # 这些user有2个以上交互item

uid = df_cnt[df_cnt['item'] >= 2].index
print('user with interacted item >=2 index: ', uid)

# print(df[df['user'] in uid])
# df.drop(df['user'] in uid, inplace=True)
# print(df)
for u in uid:
    # print(df[df['user'] == id])
    # print(df)
    df.drop(df[df['user'] == u].index, inplace=True)
    print(df.shape)

print(df)
