from flask import Blueprint, request, render_template, g, redirect, url_for
from .forms import QuestionForm, AnswerForm, ContactForm
from models import QuestionModel, AnswerModel, ContactModel
from exts import db
from decorators import login_required

# /auth 从此这个文件里的所有蓝图都要以/qa开头
bp = Blueprint("qa", __name__, url_prefix="/")

# http://127.0.0.1:5000
# 应用收到客户端发来的请求时，要找到处理该请求的视图函数
# 应用实例需要知道对每个URL的请求要运行哪些代码，所以保存了一个URL到Python函数的映射关系。
# 处理URL和函数之间关系的程序称为路由，下方我们将index()函数注册为应用根地址的处理程序

# 心灵微语系统首页
@bp.route("/")
def index():
    return render_template("HomePage.html")

# 路由URL中放在尖括号里的内容就是动态部分，任何能匹配静态部分的URL都会映射到这个路由上
# 调用视图函数时，flask会将动态部分作为参数传入函数

# 具体点击某条问题跳转到问题与评论的详情页
@bp.route("/qa/detail/<qa_id>")
def qa_detail(qa_id):
    question = QuestionModel.query.get(qa_id)
    return render_template("QuestionDetail.html", question=question)

# 要提交表单一般都是使用POST，如果不写methos默认为'GET'
# @bp.route("/answer/public", methods=['POST'])
@bp.post("/answer/public")
@login_required  # 检查是否登陆了
def public_answer():
    form = AnswerForm(request.form)
    if form.validate():
        content = form.content.data
        question_id = form.question_id.data
        answer = AnswerModel(content=content, question_id=question_id, author_id=g.user.id)
        db.session.add(answer)
        db.session.commit()
        return redirect(url_for("qa.qa_detail", qa_id=question_id))
    else:
        print(form.errors)
        return redirect(url_for("qa.qa_detail", qa_id=request.form.get("question_id")))

# Questiondisplay里面的search框提交之后，返回筛选后的问题列表
@bp.route("/search")
def search():
    # /search?q=flask  或  /search/<q>
    q = request.args.get("q")
    questions = QuestionModel.query.filter(QuestionModel.title.contains(q)).all()
    return render_template("QuestionDisplay.html", questions=questions)

@bp.route("/extern")
def extern():
    return render_template("HomePage.html")

# 跳转到发布问题页面
@bp.route("/postQuestion", methods=['GET', 'POST'])
@login_required
def postQuestion():
    if request.method == 'GET':
        return render_template("QuestionPost.html")
    else:
        # 请求对象request封装了客户端发送的HTTP请求
        # 对包含表单数据的post请求来说，用户填写的信息通过request.form访问
        form = QuestionForm(request.form)
        if form.validate():
            title = form.title.data
            content = form.content.data
            # g：处理请求时用作临时存储的对象，每次请求都会重设这个变量
            question = QuestionModel(title=title, content=content, author=g.user)
            db.session.add(question)
            db.session.commit()
            # todo: 跳转到这篇问答的详情页
            #return redirect("/answerQuestion")
            return redirect(url_for("qa.displayQuestion"))
        else:
            print(form.errors)
            return redirect(url_for("qa.postQuestion"))

# 跳转到展示所有问题的问题页面
@bp.route("/displayQuestion")
def displayQuestion():
    # 把当前时刻数据库内所有的问答都拿出来，以便在页面上显示，在返回页面index.html的时候把questions参数传过去
    questions = QuestionModel.query.order_by(QuestionModel.create_time.desc()).all()
    # 渲染：使用真实值替换变量，再返回最终得到的响应字符串；Flask使用Jinja2这个强大模板引擎渲染模版
    # Flask 提供的 render_template() 函数把 Jinja2 模板引擎集成到了应用中。其第一个参数是模板的文件名，
    # 随后的参数都是键 – 值对，表示模板中变量对应的具体值。
    return render_template("QuestionDisplay.html", questions=questions)


# 跳转到发布ContactUs界面
@bp.route("/QuestionContact", methods=['GET', 'POST'])
@login_required
def QuestionContact():
    if request.method == 'GET':
        return render_template("QuestionContact.html")
    else:
        form = ContactForm(request.form)
        if form.validate():
            name = form.name.data
            tele_num = form.tele_num.data
            content = form.content.data
            contactInfo = ContactModel(name=name, tele_num=tele_num, content=content, author=g.user)
            db.session.add(contactInfo)
            db.session.commit()
        return render_template("QuestionConSuccessfully.html")