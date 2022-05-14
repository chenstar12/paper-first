import pandas as pd
import numpy as np

user = np.random.randint(10, 12, (10))
item = np.random.randint(1000, 1003, (10))
print(user)
print(item)
d = dict()
for i in range(len(user)):
    if (user[i], item[i]) not in d.keys():
        print('none')
        d[(user[i], item[i])] = 1
    else:
        d[(user[i], item[i])] += 1
#
# for (k1,k2),v in d.items():
#     print((k1,k2))
#     print(v)

print(d)
print(len(d))
df = pd.DataFrame({'user': pd.Series(user),
                   'item': pd.Series(item)})
print(df)
# print(df)

# df_cnt = df.groupby('user').count()
# print(df_cnt[df_cnt['item'] >= 2].index)  # 这些user有2个以上交互item

# uid = df_cnt[df_cnt['item'] >= 2].index
# print('user with interacted item >=2 index: ', uid)


# for u in uid:
#     df.drop(df[df['user'] == u].index, inplace=True)
#     print(df.shape)

# print(df)
