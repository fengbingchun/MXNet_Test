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
	rm -r *
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
# build dmlc-core
echo "========== start build dmlc-core =========="
dmlc_path=${dir_name}/../../src/dmlc-core
if [ -f ${dmlc_path}/build/libdmlc.a ]; then
	echo "dmlc static library already exists without recompiling"
else
	mkdir -p ${dmlc_path}/build
	cd ${dmlc_path}/build
	cmake ..
	make
fi

ln -s ${dmlc_path}/build/libdmlc.a ${new_dir_name}
echo "========== finish build dmlc-core =========="

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

