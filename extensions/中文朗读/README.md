# VITSAIChatVtube
## 说明：
+ **基于 [VITS语音在线合成](https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai)**
+ **只支持Windows平台**
## 介绍：
***使用VITS语音合成，使用ChatGPT作为AI***
- - -
## python环境
+ [Anaconda](https://www.anaconda.com/) 作为python环境
+ [Python官方地址](https://www.python.org/) 3.9.13
+ [Pip下载地址](https://pypi.python.org/pypi/pip#downloads) 使用 `python setup.py install` 进行安装， 下载依赖库 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` 清华大学镜像
## Model模型
+ `model/G_953000.pth` 请去 [VITS语音在线合成](https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai) 下载，放到model文件夹下
## 用法
+ 将openai key写在根目录下的key.txt里，没有自己创建
+ `python main.py`
## 错误解决方法
+ 如果直接跑会报错`No module named 'monotonic_align.core'`。[按照官方的说法]（https://github.com/jaywalnut310/vits），需要先在命令行 `cd` 到 `monotonic_align` 文件夹，然后开始编译，也就是在命令行中输入 `python setup.py build_ext --inplace`
## 功能:
<details>
<summary>✔️语音合成</summary>
  
  - 通过模型进行语音合成
  - 文字转语音
</details>
  
<details>
<summary>✔️ChatGPT</summary>
  
  - 通过request方式请求chatGPT
  - 获取回复
</details>
  
<details>
<summary>✔️播放语音</summary>
  
  - 通过mpv.exe进行播放
</details>

## 链接:
+ [vits](https://github.com/jaywalnut310/vits)
+ [vits-uma-genshin-honkai](https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai)
