uidList = ['bbb','aa','vccvd']
for (i, uid) in enumerate(uidList):
    print(i,uid)


user2id = dict((uid,i) for (i, uid) in enumerate(uidList))
user2id_l = list((uid,i) for (i, uid) in enumerate(uidList))
user2id_t = tuple((uid,i) for (i, uid) in enumerate(uidList))
print(user2id)
print(user2id_l)
print(user2id_t)
