function bindEmailCaptchaClick(){
  // 用id: captcha-btn
  $("#captcha-btn").click(function (event){
    // $this：代表的是当前按钮的jquery对象
    var $this = $(this);
    // 阻止默认的事件(默认的事件可能是将这个表单提交给服务器)
    event.preventDefault();

    var email = $("input[name='email']").val(); //获取输入的邮箱
    // $是jquery库的一个缩写
    $.ajax({
      // http://127.0.0.1:500     /auth/captcha/email?email=xx@qq.com
      url: "/auth/captcha/email?email="+email,
      method: "GET",
      success: function (result){
        var code = result['code']; //拿到返回的代码

        if(code == 200){ //如果验证码获取成功
          var countdown = 5; //5s内不能再次获取验证码
          $this.off("click");// 开始倒计时之前，就取消按钮的点击事件
          var timer = setInterval(function (){
            $this.text(countdown);
            countdown -= 1;
            // 倒计时结束的时候执行
            if(countdown <= 0){
              clearInterval(timer);   // 清掉定时器
              $this.text("获取验证码"); // 将按钮的文字重新修改回来
              bindEmailCaptchaClick();// 重新绑定点击事件
            }
          }, 1000); // 每1000ms就执行一下这个函数
          // alert("邮箱验证码发送成功！");
        }else{
          alert(result['message']);
        }
      },
      fail: function (error){
        console.log(error);
      }
    })
  });
}


// 整个网页所有的内容都加载完毕后再执行的
$(function (){
  bindEmailCaptchaClick();
});