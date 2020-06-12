#!/bin/bash
gcc -c -I`python3 -c 'import ffcx.codegeneration ; print(ffcx.codegeneration.get_include_path())'` poisson.c
g++ -c -I`python3 -c 'import ffcx.codegeneration ; print(ffcx.codegeneration.get_include_path())'` main.cpp
g++ -o rhsass main.o poisson.o