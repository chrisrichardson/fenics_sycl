# Some set up for CUDA or OneAPI on CSD3

# Setup for GPU

module use /home/cnr12/rds/hpc-work/modules
module load fenicsx/dev
module load gcc/9
module load computecpp

EIGEN=`pwd`/eigen
UFC=`python3 -c 'import ffcx.codegeneration ; print(ffcx.codegeneration.get_include_path())'`
export CPATH=$CPATH:$UFC:$EIGEN

# For CUDA
# module load cuda/10.2
# cmake -DComputeCpp_DIR=$ComputeCPP_DIR -DCOMPUTECPP_BITCODE=ptx64 .

# For Intel OneAPI
export ONEAPI=/home/cnr12/rds/hpc-work/packages/intel/inteloneapi
cmake -DComputeCpp_DIR=$ComputeCPP_DIR -DOpenCL_INCLUDE_DIR=$ONEAPI/compiler/latest/linux/include/sycl -DOpenCL_LIBRARY=$ONEAPI/compiler/latest/linux/lib/libOpenCL.so .
