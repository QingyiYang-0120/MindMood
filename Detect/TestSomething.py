import os
from flask import g
# name = g.user.username  # 拿到用户登陆时的用户名
name = 'yqy'
fold_num = sum([os.path.isdir("../EATD-Corpus/" + name + "/" + listx) for listx in
                os.listdir("../EATD-Corpus/" + name)])

print(fold_num)

import os
pre = "yqy" + ".png"
pre = os.path.join('detailedresult', pre)
print(pre)
pre = " url_for('static', filename='%s') " % pre
print(pre)
