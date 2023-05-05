#encoding: utf-8
# config.py 整个项目的配置文件


import os
SECRET_KEY = os.urandom(24)

SQLALCHEMY_TRACK_MODIFICATIONS = True

DEBUG = True

# 数据库的连接
HOSTNAME = '127.0.0.1'
PORT     = '3306'
DATABASE = 'QA' 
USERNAME = 'root'
PASSWORD = '' # 请修改为mysql密码
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
SQLALCHEMY_DATABASE_URI = DB_URI

# 邮箱配置: 可以由第三方自动登陆
MAIL_SERVER = "smtp.qq.com"
MAIL_USE_SSL = True
MAIL_PORT = 465
# 请写入您的服务器邮箱与对应密码
MAIL_USERNAME = "" 
MAIL_PASSWORD = "" 
MAIL_DEFAULT_SENDER = ""
