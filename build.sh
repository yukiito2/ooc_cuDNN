#!/bin/sh

PWD={pwd}
DIR=`dirname ${0}`

cd $DIR
cd ./src
make -f Makefile_cudnn
make -f Makefile_cuda
make -f Makefile_cublas
cd ..

mkdir include
cp ./src/*.h ./include/

mkdir lib64
cp ./src/*.so ./lib64/

cd $PWD