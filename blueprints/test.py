import base64
import json
import os
import subprocess
import warnings
from io import BytesIO

# 画图部分
import joblib
import matplotlib.pyplot as plt
from flask import Blueprint, g, redirect, render_template, request, url_for

warnings.filterwarnings("ignore")
import numpy as np

import jieba
import itertools
from wordcloud import WordCloud

import pandas as pd
import sys

user = sys.argv[1]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 文件上传要用到的内容
from werkzeug.utils import secure_filename
from decorators import login_required

# 后台执行要用到的包
import threading
import subprocess

def script_run(cmd):
    res_mark = '[res_json]'  # 进程返回标记
    subp = subprocess.Popen(cmd, encoding='utf-8', stdout=subprocess.PIPE)
    out, err = subp.communicate()
    res = None
    for line in out.splitlines():
        if line.startswith((res_mark,)):
            res = json.loads(line.replace(res_mark, '', 1))
            break
    return res

is_executing = False

def execute_script(arg1):
    # 执行Python脚本
    # name = g.user.username
    name = arg1
    print(name)
    global is_executing
    is_executing = True
    print(is_executing)
    # 执行脚本文件
    print("start runnning!")
    subprocess.call(['python', 'Detect/6_show.py', name])
    print("finishing!")
    is_executing = False
    print(is_executing)


def adb_shell(cmd):
    res = os.popen(cmd).read()
    return res

bp = Blueprint("test", __name__, url_prefix="/test")

res = None
text = None

@bp.route('/testAnalysis', methods=["GET"])
@login_required
def testAnalysis():
    global res
    global text
    name = g.user.username
    print("testAnalysis!")
    # 创建一个线程来执行脚本
    thread = threading.Thread(target=execute_script, args=(name,))
    # 启动线程
    thread.start()
    print("guole!")
    return render_template('TestResult.html', result=res, advice=text)



@bp.route('/testBegin', methods=["GET", "POST"])
@login_required
def testBegin():
    global res
    global text
    UPLOAD_FOLDER = './EATD-Corpus/'
    ALLOWED_EXTENSIONS = {'wav', 'txt'}  # 可上传的文件名后缀集合

    name = g.user.username  # 拿到用户登陆时的用户名
    # print(g.user.username) # 检查一下

    # 当用户上传文件的时候，将POST得到的文件保存在服务器的指定位置
    if request.method == 'POST':
        files = request.files.getlist("myfile")
        print(files)  # 在控制台打印一下POST请求中收到的文件
        print(type(files))  # 以及收到的文件类型

        check = ['positive.txt', 'neutral.txt', 'negative.txt', 'positive.wav', 'neutral.wav', 'negative.wav']
        # print(set(check))
        # print(set(files))
        file_set = []
        for file in files:
            filename = secure_filename(file.filename)
            file_set.append(filename)

        # print(set(file_set))  # 使用set()将list转化为集合，判断两个集合 check的元素是否都在files中 <则是真子集 <=是子集
        flag = set(check) <= set(file_set)
        # print(flag)

        if flag == False:  # 未上传正确的内容发出警告
            return render_template("TestBegin.html", alerm=True)

        # 保存在EATD-Corpus里面相应的当前登陆的用户名文件夹里面
        folder = UPLOAD_FOLDER + str(name)
        print("folder = %s" % folder)

        # 用户是否是第一次上传数据（还未存在用户文件夹）
        if not os.path.exists(folder):
            os.mkdir(folder)
            print("first time create the user folder!")

        # 然后要考虑用户是第几次上传
        fold_num = sum([os.path.isdir(folder + "/" + listx) for listx in os.listdir(folder)])
        print("right now， the num of the test: %d" % fold_num)

        # 创建新的保存数据的文件夹
        os.mkdir(os.path.join(folder, str(fold_num + 1)))
        folder = os.path.join(folder, str(fold_num + 1))

        for file in files:
            filename = secure_filename(file.filename)
            print(filename) # 当前正在保存哪个子目录
            if file and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS:
                print(file)
                file.save(os.path.join(folder, filename))

        # import sys
        # path = sys.argv[0]
        # abs_path = os.path.abspath(sys.argv[0])
        # dirname, filename = os.path.split(abs_path)
        # print(path)
        # print(abs_path)
        # print(dirname, filename)
        # 返回当前的工作目录

        print("os.getcwd: %s" % os.getcwd())  # /Users/yangqingyi/Desktop/JS-AI/心灵侦探

        for pyfile in ['1_one_for_audio.py', '2_one_for_text.py', '3_two.py']:
            pre = os.path.join(os.getcwd(), 'Detect')
            print("running directory:%s" % os.path.join(pre, pyfile))
            os.system("python %s %s" % (os.path.join(pre, pyfile), name))

        res = adb_shell('python %s %s' % (os.path.join(os.getcwd(), 'Detect/4_classification.py'), name))

        # print(res)
        # print(res[2:4] == "抑郁" or print(res[4:6] == "正常")

        if res[2:4] == "抑郁":
            text = '<span style="color:#FFB6C1;font-weight:bold;font-size:20px;">Suggestions:</span><br>' \
                   '1. 不管你现在多么的痛苦，都要坚持住。不要被眼前的困难所打败，如果很累了，那就降低要求，回家好好休息一段时间，给自己一段时间来疗愈心情或者寻求医生正规地治疗。<br/><br/>' \
                   '2. 科学治疗，对症治疗，对症用药治疗加上心理治疗会更好。带着症状去生活，不去过度关注自己的症状，忙碌充实起来，反而症状会消失掉。<br/><br/>' \
                   '3. 坚持运动，运动出汗，对提高情绪也是有帮助的，因为出汗大脑里会分泌一些多巴胺出来，这种神经递质能使人愉悦。<br/><br/>' \
                   '4. 多吸收正能量，负面能量太多是黑暗；而人若想要阳光起来，需要靠积极的正能量。<br/>' \

        elif res[4:6] == "正常":
            text = '<span style="color:#FFB6C1;font-weight:bold">Suggestions:</span><br>' \
                   "1. 工作或是生活当中，都要注意建议良好的人际关系，并且在有压力的时候积极倾诉和求助。<br/>" \
                   "2. 在平时生活当中，一定要寻找自己的生活乐趣，要不断的尝试创新，这样能给自己的精神上得到一定的满足，能够放松身心，起到保持心理健康的效果。<br/>" \
                   "3. 时常和家人保持联系，家是我们避风的港湾，而家庭环境所具有的安全感会造成非常重要的影响，有了家人的爱护和理解我们就会有了安全感。<br/>" \
                   "4. 客观的对自身进行评价，评价不宜过高过低。<br/>" \
                   "5. 加强与外界的接触，可以丰富自身精神生活，亦或可以及时调整自己适应环境。"

        # 执行二维位置可视化与文本分析
        pre = os.path.join(os.getcwd(), 'Detect')
        os.system("python %s %s" % (os.path.join(pre, '5_analysis.py'), name))
        # return render_template('TestResult.html', result=res, advice=text)
        return redirect(url_for("test.testAnalysis"))
        # return render_template('TestResult.html')
    # request.method == 'GET' 当用户单纯在请求该页面的时候
    else:
        return render_template("TestBegin.html", alerm=False)

# @bp.route('/run', methods=["GET"])
# def run_script():
#     print("testAnalysis!")
#     # 创建一个线程来执行脚本
#     thread = threading.Thread(target=execute_script)
#     thread.start()
#     print("guole!")
#     # 如果访问的是 /your_page 页面，则运行 Python 脚本
#     # subprocess.Popen(["python", "your_script.py"])



@bp.route('/easytestBegin', methods=["GET", "POST"])
@login_required
def easytestBegin():
    name = g.user.username  # 拿到用户登陆时的用户名
    res = None
    if request.method == 'POST':
        content = request.form.get('content')
        print(content)  # 在控制台打印一下POST请求中收到的文件
        print(type(content))  # 以及收到的文件类型

        print("os.getcwd: %s" % os.getcwd())  # /Users/yangqingyi/Desktop/JS-AI/心灵侦探
        res = adb_shell('python %s %s %s' % (os.path.join(os.getcwd(), 'Detect/0_easy.py'), name, content))
        print(res)
        if res[:2] == "正常":
            result = "正常"
        else:
            result = "抑郁"
        return render_template('TestBeginEasy.html', result = result, advice=res)
    else:
        return render_template("TestBeginEasy.html", advice=res)


@bp.route('/testDetail', methods=["GET"])
@login_required
def testDetail():
    name = g.user.username  # 拿到用户登陆时的用户名

    # data = pd.read_csv("EATD-Corpus/" + str(name) + "/" + "record.csv")
    # with open('Detect/stopwordlist.txt', 'r', encoding='utf-8') as f:  # 停用词的文本
    #     stop = f.read()
    # stop = stop.split()
    # stop = [' ', '\n', '这部'] + stop
    #
    # fig = plt.figure(figsize=(15, 12), dpi=700)
    # ax = fig.add_subplot(221)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    #
    # ax.plot(np.arange(0, data.shape[0]), data["SDS分数上限"], linestyle="--", label="SDS分数上限")
    # ax.plot(np.arange(0, data.shape[0]), data["SDS分数下限"], linestyle="--", label="SDS分数下限")
    # ax.plot(np.arange(0, data.shape[0]), data["预测SDS分数"], label="预测SDS分数")
    # ax.scatter(np.arange(0, data.shape[0]), data["SDS分数上限"])
    # ax.scatter(np.arange(0, data.shape[0]), data["SDS分数下限"])
    # ax.scatter(np.arange(0, data.shape[0]), data["预测SDS分数"], marker="*")
    # plt.grid()
    # plt.legend()
    # ax.set_xlabel("记录")
    # ax.set_ylabel("分数")
    # ax.set_title(name + "SDS分数变化图", fontsize=20)
    #
    # train_features = ["Detect/fuse_features.npy"]
    # train_labels = ["Detect/fuse_labels.npy"]
    #
    # # from sklearn import manifold
    # # tsne = manifold.TSNE()
    # tsne = joblib.load("Detect/Model/tsne.dat")
    #
    # X_train = np.load(train_features[0])
    # y = np.load(train_labels[0])
    #
    # x_test = data[["sds_x", "sds_y"]].values
    # X_tsne = tsne.fit_transform(X_train)  # 对训练X进行降维
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # X0 = X_norm[y == 0]
    # X1 = X_norm[y == 1]
    #
    # ax1 = fig.add_subplot(222)
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['top'].set_visible(False)
    # plt.title("被试者情绪倾向二维可视化图", fontsize=20)
    # plt.scatter(X0[:, 0], X0[:, 1], label="历史正常", marker="o", color="green")
    # plt.scatter(X1[:, 0], X1[:, 1], label="历史抑郁", marker="x", color="red")
    # plt.scatter(x_test[:, 0], x_test[:, 1], label=name, marker="d", color="blue")
    # for i in range(x_test.shape[0]):
    #     if (data["结果"][i] == 1):
    #         plt.text(x_test[i, 0], x_test[i, 1], str(i + 1) + "X")
    #     else:
    #         plt.text(x_test[i, 0], x_test[i, 1], str(i + 1))
    # plt.grid()
    # plt.legend(bbox_to_anchor=(0.5, -0.1), loc=8, ncol=10)
    # plt.xticks([])
    # plt.yticks([])
    #
    # ax2 = fig.add_subplot(223)
    # x0 = np.mean(data["正常概率"].values)
    # x1 = np.mean(data["抑郁概率"].values)
    # piex = np.array([x0, x1])
    # patches, l_text, p_text = plt.pie(piex,labels=['正常', '抑郁'],colors=["lightgreen", "mediumorchid"],
    #                                   explode=(0, 0.2), autopct='%.2f%%')
    # for t in p_text:
    #     t.set_size(20)
    # for t in l_text:
    #     t.set_size(12)
    # plt.title(user + "情绪概率可视化展示", fontsize=20)  # 设置标题
    # plt.grid()
    #
    # ax3 = fig.add_subplot(224)
    # sens = ""
    # for i in data["个人摘要"]:
    #     sens += i[1:-1]
    # lines = pd.Series(sens)
    # data_pos = lines.apply(jieba.lcut)
    # data_pos = data_pos.apply(lambda x: [i for i in x if i not in stop and len(i) > 1])
    # data_pos.index = [i for i in range(data_pos.shape[0])]
    # num = pd.Series(list(itertools.chain(*list(data_pos)))).value_counts()  # 统计词频
    # # 把scale=32 这个属性删掉了
    # wc = WordCloud(font_path='Detect/SimHei.ttf', background_color='White',height=350, colormap="RdPu")
    # wc2 = wc.fit_words(num)
    # plt.imshow(wc2)
    # plt.axis('off')
    # plt.title(name + "用户文本关键词记录", fontsize=20)
    #
    # # figure 保存为二进制文件
    # buffer = BytesIO()
    # plt.savefig(buffer)
    # plot_data = buffer.getvalue()
    #
    # #  将matplotlib图片转换为HTML
    # imb = base64.b64encode(plot_data)  # 对plot_data进行编码
    # ims = imb.decode()
    # imd = "data:image/png;base64," + ims

    # 将生成的png图片推送给前端

    #return render_template("TestDetail.html", img=imd)

    # pre = name + ".png"
    # pre = os.path.join('detailedresult', pre)
    # print(pre)
    # pre = " url_for('static', filename='%s') " % pre
    # print(pre)
    # global is_executing
    # print(is_executing)
    # while 1:
    #     if not is_executing:
    #         return render_template("TestDetail.html", path=pre)
    global is_executing
    print(is_executing)
    while 1:
        if not is_executing:
            return render_template("TestDetail.html")
