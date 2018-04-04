#!/bin/sh

PWD={pwd}
DIR=`dirname ${0}`

cd $DIR
cd ./src
make -f Makefile_cudnn clean
make -f Makefile_cuda clean
make -f Makefile_cublas clean
cd ..

rm ./include/*
rm -r ./include

rm ./lib64/*
rm -r ./lib64

cd $PWD