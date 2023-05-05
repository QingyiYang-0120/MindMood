#encoding: utf-8

from exts import db
from datetime import datetime

class UserModel(db.Model):
    __tablename__ = "user"
    # 用户表user：用户id，用户名username，密码password，邮箱email，注册时间join_time
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    join_time = db.Column(db.DateTime, default=datetime.now)

class EmailCaptchaModel(db.Model):
    __tablename__ = "email_captcha"
    # 邮箱验证码表email_captcha：表id，邮箱email，验证码captcha
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(100), nullable=False)   # 邮箱
    captcha = db.Column(db.String(100), nullable=False) # 验证码

class QuestionModel(db.Model):
    __tablename__ = "question"
    # 问题表question：问题id，标题title，内容content，发布时间create_time，发布作者的id，
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    create_time = db.Column(db.DateTime, default=datetime.now)

    # 外键
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    # 关系 通过question.author可以获取对应的UserModel模型对象 而不需要再通过author_id去取
    author = db.relationship(UserModel, backref="questions")

# 寻求更多帮助的表
class ContactModel(db.Model):
    __tablename__ = "contactInfo"
    # 记录寻求帮助的表contactInfo：contact id，真实姓名name，联系电话tele_num，内容content，发布时间create_time，发布作者的user_id
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(10), nullable=False)
    tele_num = db.Column(db.String(15), nullable=False)
    content = db.Column(db.Text, nullable=False)
    create_time = db.Column(db.DateTime, default=datetime.now)

    # 外键
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    # 关系 通过question.author可以获取对应的UserModel模型对象 而不需要再通过author_id去取
    author = db.relationship(UserModel, backref="contacts")


# 记录回答的表
class AnswerModel(db.Model):
    __tablename__ = "answer"
    # 评论表answer：评论id，评论内容content，评论时间create_time，评论的问题的id，发布评论作者id，
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    content = db.Column(db.Text, nullable=False)
    create_time = db.Column(db.DateTime, default=datetime.now)

    # 外键   传给 db.ForeignKey()的参数 'user.id' 表明这列的值是roles表中相应行的id值
    question_id = db.Column(db.Integer, db.ForeignKey("question.id"))
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"))

    # 针对每个外键，定义两个关系
    # 我有一个question，可以通过answers去拿到question下面所有的答案
    # db.relationship() 的第一个参数表明这个关系的另一端是哪个模型（如QuestionModel），
    # backref参数向本模型（AnswerModel）中添加一个名为参数值（answers）的属性，从而定义反向关系
    # 通过 AnswerModel 实例的这个属性可以获取对应的 QuestionModel 模型对象，而不用再通过 role_id 外键获取。
    question = db.relationship(QuestionModel, backref=db.backref("answers", order_by=create_time.desc()))
    author = db.relationship(UserModel, backref="answers")