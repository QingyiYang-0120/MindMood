#encoding: utf-8

import config
from flask import Flask, session, g      # g是flask中专门用来存储全局对象的
from exts import db, mail
from models import UserModel

from blueprints.qa import bp as qa_bp     # 关于登陆注册有关的处理
from blueprints.auth import bp as auth_bp # 关于问答有关的处理
from blueprints.test import bp as test_bp # 关于抑郁测试有关的处理
from flask_migrate import Migrate         # 数据库迁移工具

# Web服务器使用名为Web服务器网关接口（WSGI，Web server gateway interface）的协议
# 把接收自客户端的所有请求都转交给这个Flask类的对象app处理
app = Flask(__name__)
# 绑定配置文件
app.config.from_object(config)
# 将db与app绑定
db.init_app(app)
mail.init_app(app)

migrate = Migrate(app, db)

# 注册蓝图
app.register_blueprint(auth_bp)
app.register_blueprint(qa_bp)
app.register_blueprint(test_bp)


# before_request/ before_first_request/ after_request
# hook
@app.before_request
# before_request注册一个函数，在每次请求之前运行
# 在请求钩子函数和视图函数之间共享数据一般使用上线全局变量g，从数据库中加载已登陆用户
# 并将其保存到g.user中，随后调用视图函数时，便可以通过g.user获取用户
def my_before_request():
    user_id = session.get("user_id")
    if user_id:
        user = UserModel.query.get(user_id)
        setattr(g, "user", user)
    else:
        setattr(g, "user", None)


# 把变量存放在上下文处理器中，其中返回的数据在所有的模块中都可以被使用
@app.context_processor
def my_context_processor():
    return {"user": g.user}

if __name__ == '__main__':
    app.run(debug=True)
