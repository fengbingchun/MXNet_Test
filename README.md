# MXNet_Test
**The main role of the project: MXNet's usage**  
**Note: Clone this repository to E:/GitCode/ in windows**
- open source library version:
	- MXNet: 1.3.0, [GitHub](https://github.com/apache/incubator-mxnet/releases)
	- OpenBLAS: 0.3.3, [GitHub](https://github.com/xianyi/OpenBLAS/releases)
	- dlmask: master, commit: bee4d1d, [GitHub](https://github.com/dmlc/dlpack)
	- mshadow: master, commit: 2e3a895, [GitHub](https://github.com/dmlc/mshadow)
	- dmlc-core: 0.3, [GitHub](https://github.com/dmlc/dmlc-core/releases)
	- tvm: 0.4, [GitHub](https://github.com/dmlc/tvm/releases)
- test code include:
	- C++(windows and linux)
		- MXNet's third library usage: dmlc-core、OpenBLAS、mshadow
		- MNIST train
	- python(windows and linux)

**The project support platform:** 
- windows10 64 bits: It can be directly build with VS2017 in windows10 64bits.
- Linux 
	- ThirdPartyLibrary_Test support cmake build(file position: prj/linux_cmake_ThirdPartyLibrary_Test)
	- MXNet_Test support cmake build(file position: prj/linux_cmake_MXNet_Test)

**Windows VS Screenshot:**  
![](https://github.com/fengbingchun/MXNet_Test/blob/master/prj/x86_x64/Screenshot.png)

**Blog:** [fengbingchun](https://blog.csdn.net/fengbingchun/article/category/8523737)
