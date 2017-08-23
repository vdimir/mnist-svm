#!/usr/bin/env bash
if [ ! -d "mnist_png/mnist_png" ]; then
    tar xfv mnist_png/mnist_png.tar.gz --directory=mnist_png
fi

find mnist_png/mnist_png/training -type f -name "*.png" | sed -r 's!^.+/([0-9])/.+$!\0 \1!' > input-train.txt
find mnist_png/mnist_png/testing -type f -name "*.png" | sed -r 's!^.+/([0-9])/.+$!\0 \1!' > input-test.txt
