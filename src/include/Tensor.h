#ifndef _TENSOR_HEADER_
#define _TENSOR_HEADER_

#include "include.h"

/*
To add a new tensor function:
1. Create the new kernel in tensor_kernels.cl
2. Add it to beginning of namespace Tensor as an extern
3. Add function prototype without result to Tensor.h
4. Add function prototype with result to Tensor.h
5. Add kernel to beginning of Tensor.cpp
6. Add kernel initialization code to init fxn in Tensor.cpp
7. Define the full fxn with result input in Tensor.cpp
8. Use the created "result input fxn" to define the fxn with no result input in Tensor.cpp
*/

namespace Tensor
{
    extern cl::Kernel addKernel;
    extern cl::Kernel subKernel;
    extern cl::Kernel multKernel;
    extern cl::Kernel convKernel;

    extern cl_int err;
    
    class Tensor
    {
    public:
        cl::Buffer storageBuffer;
        vector <int> dim;
        int total_size;

        Tensor(vector <int>& dim, string str, int val);
        void setValue(vector <float>& values);
        vector <float>& getValue();

        void print();
    };

    void init();
    void check_error();

    Tensor& add(Tensor& T1, Tensor& T2);
    Tensor& sub(Tensor& T1, Tensor& T2);
    Tensor& mult(Tensor& T1, Tensor& T2);
    Tensor& conv(Tensor& T, Tensor& filter, Tensor& bias, pair<int,int> stride);

    void add(Tensor& T1, Tensor& T2, Tensor& result);
    void sub(Tensor& T1, Tensor& T2, Tensor& result);
    void mult(Tensor& T1, Tensor& T2, Tensor& result);
    void conv(Tensor& T, Tensor& filter, Tensor& bias, pair<int,int> stride, Tensor& result);
};

#endif