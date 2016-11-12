# FactorG: a matrix factorizer with stochastic gradient descent on GPU

This repository is a CUDA implementation of matrix factorization with
stochastic gradient descent.

## Requirements

These are the base requirements to build and use FactorG: 

  * POSIX-standard shell
  * GNU-compatible Make
  * G++ compiler with C++11 support
  * CUDA toolkit 7.5

## Quick start

```sh
make
cd bin
./mf -t train-file [-e test-file] [-c config-file] [-d gpu-id]
```

