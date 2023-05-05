import string
import random
from flask import Blueprint, request, render_template, jsonify, redirect, url_for, session
from exts import mail, db
from flask_mail import Message
from models import EmailCaptchaModel, UserModel
from .forms import RegisterForm, LoginForm

from werkzeug.security import generate_password_hash, check_password_hash  # 生成加密密码

# /auth 从此这个文件里的所有蓝图都要以/auth开头
bp = Blueprint("auth", __name__, url_prefix="/auth")

# 退出登陆
@bp.route("/logout")
def logout():
    session.clear()  # 将cookie里面的信息清空
    return redirect("/")

# bp.route：如果没有指定methods参数，默认就是GET请求
@bp.route("/captcha/email")
def get_email_captcha():
    # /captcha/email/<email> 路径传参
    # /captcha/email?email=xxx@qq.com  查询字符串传参
    email = request.args.get("email")
    # 4/6位：随机数组、字母、数组和字母的组合
    source = string.digits*4
    captcha = random.sample(source, 4)  # 从source中随机采样产生4位数据 captcha是列表类型
    captcha = "".join(captcha)  # 将字符串拼接成字符串
    # I/O：Input/Output 较为费时（后面可以考虑延迟优化）
    message = Message(subject="心灵侦探注册验证码", recipients=[email], body=f"您好！您正在注册心灵侦探——基于多模态数据挖掘的抑郁判别系统账号，您的验证码是:{captcha}。")
    mail.send(message)

    # memcached(纯内存，没有同步机制)/redis(有同步机制)
    # 用数据库表的方式存储(邮箱-验证码)的对应关系
    email_captcha = EmailCaptchaModel(email=email, captcha=captcha)
    db.session.add(email_captcha)
    db.session.commit()

    # 返回数据要满足RESTful API的格式
    # {code: 200正常/400客服端错误/500服务器错误, message: "", data: {}}
    return jsonify({"code": 200, "message": "", "data": None})

# 测试一下自动发送邮箱验证码是否成功
@bp.route("/mail/test")
def mail_test():
    message = Message(subject="邮箱测试", recipients=["1837125703@qq.com"], body="这是一条测试邮件")
    mail.send(message)
    return "邮件发送成功"

# 处理用户登陆
@bp.route("/registerNew", methods=['GET', 'POST'])
def register_login():
    if request.method == 'GET':
        return render_template("UserLoginRegister.html")
    else:
        form = LoginForm(request.form)
        if form.validate():
            email = form.email.data
            password = form.password.data
            user = UserModel.query.filter_by(email=email).first()

            if not user:
                print("邮箱在数据库中不存在！")
                return redirect(url_for("auth.register_login"))

            if check_password_hash(user.password, password):
                # cookie：中不适合存储太多的数据，只适合存储少量的数据，一般用来存放登录授权的东西
                # flask中的session，是经过加密后存储在cookie中的
                session['user_id'] = user.id
                return redirect("/")

            else:
                print("密码错误！")
                return redirect(url_for("auth.register_login"))

        else:
            print(form.errors)
            return redirect(url_for("auth.register_login"))


# 处理用户注册
@bp.route("/handle_register", methods=['POST'])
def handle_register():
    form = RegisterForm(request.form)
    if form.validate():
        email = form.registeremail.data
        username = form.username.data
        password = form.password.data
        user = UserModel(email=email, username=username, password=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()  # 将用户提交的注册数据插入到数据库里
        return redirect(url_for("auth.register_login"))  # 跳转函数
    else:
        print(form.errors)
        return redirect(url_for("auth.register"))
