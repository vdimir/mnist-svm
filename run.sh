#!/usr/bin/env bash
CMAKE_BUILD_DIR=./cmake-build-debug
PYTHON_EXE=./env/bin/python

if [ ! -d $CMAKE_BUILD_DIR ]; then
    mkdir $CMAKE_BUILD_DIR
    cd $CMAKE_BUILD_DIR
    cmake ..
    cd -
fi
cmake --build $CMAKE_BUILD_DIR --target task -- -j 4

EXE=$CMAKE_BUILD_DIR/task

echo "=======Raw image======="
$EXE train raw input-train.txt model.dat &&
$EXE test raw input-test.txt model.dat output.txt &&
$PYTHON_EXE ./evaluation.py output.txt


echo "=======Preprocessed image======="
$EXE train proc input-train.txt model.dat &&
$EXE test proc input-test.txt model.dat output.txt &&
$PYTHON_EXE ./evaluation.py output.txt