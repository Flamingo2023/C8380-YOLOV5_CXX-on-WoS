## 1. Overview

This document provides a detailed introduction to the environment for compiling and running this demo, as well as how to use the demo program.

## 2. Install Visual Studio Community Edition
[Download link for online installation of Visual Studio Community Edition](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false)

Simply install it normally; the specific installation process is not described in detail.

## 3. Install Package Management Tool VCPKG
```powershell
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg ; .\bootstrap-vcpkg.bat
.\vcpkg.exe integrate install
```
After completion, add the vcpkg directory to the system environment variable PATH.

## 4. Install QNN SDK (AI Engine Direct)
Log in to Qualcomm's official website to download and install the [QNN SDK package](https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct). Choose to install version 2.24.0.240626.

After installation, configure the environment with the following command:
```powershell
& "C:\Qualcomm\AIStack\QAIRT\2.24.0.240626\bin\envsetup.ps1"
```

## 5. Download the Source Code and Compile
- Download the Source Code
Clone this open-source project to `C:\Users\HCKTest\source\repos` and rename the folder to `demo_yolov5_based_qnn_cpp`.

- Compile AppBuilder
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

- Compile this Demo

Double-click the file `demo_yolov5_based_qnn_cpp\demo_yolov5_based_qnn_cpp.sln` to start Visual Studio, select the configuration ARM64+Debug or ARM64+Release, and run it using Debug -> Start Without Debugging.

This demo program first performs object detection on two images in the `images` directory and displays the detection results. Then, by pressing the 'n' key, it exits image detection and starts capturing from the USB camera for object detection, while simultaneously displaying the detection results in real-time.