#encoding: utf-8

# 所有的扩展文件，插件
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail

mail = Mail()
# 解决循环引用问题
db = SQLAlchemy()