# g-factor: a matrix factorizer with stochastic gradient descent on GPU

**g-factor** is a matrix factorization implementation with parallel stochastic gradient descent on AMD or NVIDIA GPUs. It uses HIP and CUDA languages.

## Requirements

These are the base requirements to build and use **g-factor**: 

  * POSIX-standard shell
  * GNU-compatible Make
  * G++ compiler with C++11 support
  * HIP or CUDA toolkit

## Quick start

```sh
make
cd bin
./mf -t train-file [-e test-file] [-c config-file] [-d gpu-id]
```
`-t train-file`: Train data file in the COO format. The first line is a (number-of-users, number-of-items, number-of-ratings) tuple, where the elements are divided by spaces/tabs. Each following line is a (user-id, item-id, rating) tuple, where the elements are divided by spaces/tabs. Comment lines start with '%'s.

`[-e test-file]`: Test data file in the COO format. The first line is a (number-of-users, number-of-items, number-of-ratings) tuple, where the elements are divided by spaces/tabs. Each following line is a (user-id, item-id, rating) tuple, where the elements are divided by spaces/tabs. Comment lines start with '%'s.

`[-c config-file]`: Configuration file. Comment lines start with '#'s.

`[-d gpu-id]`: GPU used to accelerate computation. If skipped, CPU is used.
