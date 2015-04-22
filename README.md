# MCMC Performance Primitives

A library for high performance Markov Chain Monte Carlo (MCMC)

## build instructions

You will need a compiler that supports C++11 standard. Eigen and Boost libraries are also required.

Here are the steps

    git clone https://github.com/tbs1980/MCMC-Performance-Primitives.git
    cd MCMC-Performance-Primitives
    mkdir build
    cd build
    cmake -DEIGEN3_INCLUDE_DIR=path_to_eigen_core -DBOOST_ROOT=path_to_boost -DCMAKE_INSTALL_PREFIX=path_to_install ../
    make
    make install

where `path_to_eigen_core` is the path to Eigen header files, in particular `Eigen/Core`, `path_to_boost` is the path to Boost header files and `path_to_install` is the installation path.

If every step goes well, you will see the files in `path_to_install`
