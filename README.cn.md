## 1. 概述

本文的详细介绍编译运行本demo的环境以及如何使用本demo程序。

## 2. 安装Visual Studio社区版
[Visual Studio社区版在线安装的下载链接](https://visualstudio.microsoft.com/zh-hans/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false)

正常安装即可，具体安装过程不详细描述。

## 3. 安装包管理工具VCPKG
```powershell
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg ; .\bootstrap-vcpkg.bat
.\vcpkg.exe integrate install
```
完成后将vcpkg目录添加到系统环境变量PATH中。

## 4. 安装QNN SDK(AI Engine Direct)
登录高通官网下载并安装[QNN SDK软件包](https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct)，这里选择安装2.24.0.240626版本。

安装完成，需配置环境，相关指令如下：
```powershell
& "C:\Qualcomm\AIStack\QAIRT\2.24.0.240626\bin\envsetup.ps1"
```

## 5. 下载本源码并进行编译
- 下载源码
克隆本开源项目至C:\Users\HCKTest\source\repos，并将文件夹重命名为demo_yolov5_based_qnn_cpp。

- 编译appbuilder
```powershell
cd C:\Users\HCKTest\source\repos\demo_yolov5_based_qnn_cpp
cd appbuilder\src
mkdir build
cd build
cmake .. -T ClangCL -A ARM64
cmake --build . --config Debug
cmake --build . --config Release
cd ..\..
```

- 编译本demo

双击文件demo_yolov5_based_qnn_cpp\demo_yolov5_based_qnn_cpp.sln，启动Visual Studio，选择配置ARM64+Debug或者ARM64+Release，并选择Debug->Start Without Debugging运行即可。

本demo程序先对images目录中两图片进行目标检测，并展示目标检测结果。随后按n键退出图片检测，并开始采集USB摄像头并进行目标检测，同时实时展示目标检测结果。