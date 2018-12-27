#! /bin/bash

real_path=$(realpath $0)
dir_name=`dirname "${real_path}"`
echo "real_path: ${real_path}, dir_name: ${dir_name}"

data_dir="data"
if [ -d ${dir_name}/${data_dir} ]; then
	rm -rf ${dir_name}/${data_dir}
fi

ln -s ${dir_name}/./../../${data_dir} ${dir_name}

new_dir_name=${dir_name}/build
mkdir -p ${new_dir_name}
cd ${new_dir_name}
echo "pos: ${new_dir_name}"
if [ "$(ls -A ${new_dir_name})" ]; then
	echo "directory is not empty: ${new_dir_name}"
	#rm -r *
else
	echo "directory is empty: ${new_dir_name}"
fi

cd -
# build openblas
echo "========== start build openblas =========="
openblas_path=${dir_name}/../../src/openblas
if [ -f ${openblas_path}/build/lib/libopenblas.so ]; then
	echo "openblas dynamic library already exists without recompiling"
else
	mkdir -p ${openblas_path}/build
	cd ${openblas_path}/build
	cmake  -DBUILD_SHARED_LIBS=ON ..
	make
fi

ln -s ${openblas_path}/build/lib/libopenblas* ${new_dir_name}
echo "========== finish build openblas =========="

cd -
rc=$?
if [[ ${rc} != 0 ]]; then
	echo "########## Error: some of thess commands have errors above, please check"
	exit ${rc}
fi

cd -
cd ${new_dir_name}
cmake ..
make

cd -
# use all *.o files include openblas and mxnet to generate mxnet dynamic library
# Note: need to modify mxnet_home_dir value in different machines
mxnet_home_dir=/home/likewise-open/xxxx/fengbc/Other/MXNet_Test/src
g++ -shared -o ${dir_name}/libmxnet.so ${openblas_path}/build/driver/others/CMakeFiles/driver_others.dir/*.c.o \
					${openblas_path}/build/driver/level2/CMakeFiles/driver_level2.dir/CMakeFiles/*.c.o \
					${openblas_path}/build/driver/level3/CMakeFiles/driver_level3.dir/CMakeFiles/*.c.o \
					${openblas_path}/build/kernel/CMakeFiles/kernel.dir/CMakeFiles/*.c.o \
					${openblas_path}/build/interface/CMakeFiles/interface.dir/CMakeFiles/*.c.o \
					${openblas_path}/build/kernel/CMakeFiles/kernel.dir/CMakeFiles/*.S.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/storage/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/executor/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/profiler/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/engine/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/nnvm/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/kvstore/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/common/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/ndarray/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/io/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/operator/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/operator/quantization/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/operator/quantization/mkldnn/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/operator/nnpack/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/operator/contrib/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/operator/random/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/operator/nn/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/operator/nn/cudnn/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/operator/nn/mkldnn/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/operator/tensor/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/operator/custom/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/imperative/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/mxnet/src/c_api/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/tvm/nnvm/src/core/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/tvm/nnvm/src/pass/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/tvm/nnvm/src/c_api/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/dmlc-core/src/*.cc.o \
					${dir_name}/build/CMakeFiles/mxnet.dir${mxnet_home_dir}/dmlc-core/src/io/*.cc.o \
					-lrt 

mxnet_python_dir=${dir_name}/../../src/mxnet/python
#echo "mxnet python dir: ${mxnet_python_dir}"
cp libmxnet.so ${mxnet_python_dir}
cd ${mxnet_python_dir}
python3 setup.py install --user

cd -
