import wtforms
from wtforms.validators import Email, Length, EqualTo, InputRequired, Regexp
from models import UserModel, EmailCaptchaModel
from exts import db

# Form：主要就是用来验证前端提交的数据是否符合要求
class RegisterForm(wtforms.Form):
    registeremail = wtforms.StringField(validators=[Email(message="邮箱格式错误！")])
    captcha = wtforms.StringField(validators=[Length(min=4, max=4, message="验证码格式错误！")])
    username = wtforms.StringField(validators=[Length(min=3, max=20, message="用户名格式错误！")])
    password = wtforms.StringField(validators=[Length(min=6, max=20, message="密码格式错误！")])
    password_confirm = wtforms.StringField(validators=[EqualTo("password", message="两次密码不一致！")])

    # 自定义验证： 1. 邮箱是否已经被注册
    def validate_email(self, field):
        registeremail = field.data
        user = UserModel.query.filter_by(email=registeremail).first()
        if user:
            raise wtforms.ValidationError(message="该邮箱已经被注册！")

    # 2. 验证码是否正确
    def validate_captcha(self, field):
        captcha = field.data
        registeremail = self.registeremail.data
        # 得到了一个对象存在captcha_model里
        captcha_model = EmailCaptchaModel.query.filter_by(email=registeremail, captcha=captcha).first()
        if not captcha_model:
            raise wtforms.ValidationError(message="邮箱或验证码错误！")
        # else:
        #     # todo：可以删掉captcha_model
        #     db.session.delete(captcha_model)
        #     db.session.commit()

# 登陆的表单
class LoginForm(wtforms.Form):
    email = wtforms.StringField(validators=[Email(message="邮箱格式错误！")])
    password = wtforms.StringField(validators=[Length(min=6, max=20, message="密码格式错误！")])

# 发布问题的表单
class QuestionForm(wtforms.Form):
    title = wtforms.StringField(validators=[Length(min=2, max=100, message="标题格式错误！")])
    content = wtforms.StringField(validators=[Length(min=3,message="内容格式错误！")])

# 回答/评论的表单
class AnswerForm(wtforms.Form):
    content = wtforms.StringField(validators=[Length(min=2, message="内容格式错误！")])
    question_id = wtforms.IntegerField(validators=[InputRequired(message="必须要传入问题id！")])

# ContactUS 的表单
class ContactForm(wtforms.Form):
    name = wtforms.StringField(validators=[Length(min=1, max=10, message="真实姓名格式错误！")])
    tele_num = wtforms.StringField(validators=[Regexp(r'1[34578]\d{9}', message='手机号格式错误')])
    content = wtforms.StringField(validators=[Length(min=3, message="内容格式错误！")])