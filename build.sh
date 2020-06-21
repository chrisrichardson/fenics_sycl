#!/bin/bash
ffcx poisson.ufl
EIGEN=/usr/include/eigen3
UFC=`python3 -c 'import ffcx.codegeneration ; print(ffcx.codegeneration.get_include_path())'`
gcc -c -I$UFC -I$EIGEN poisson.c
g++ -c -I$UFC -I$EIGEN main.cpp
g++ -o rhsass main.o poisson.o
