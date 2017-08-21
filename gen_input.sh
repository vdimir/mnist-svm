if [ ! -d "mnist_png/mnist_png" ]; then
    tar --directory=mnist_png xfv mnist_png/mnist_png.tar.gz
fi

find mnist_png/mnist_png/training -type f | sed -r 's!^.+/([0-9])/.+$!\0 \1!' > input-train.txt
find mnist_png/mnist_png/testing -type f | sed -r 's!^.+/([0-9])/.+$!\0 \1!' > input-test.txt
