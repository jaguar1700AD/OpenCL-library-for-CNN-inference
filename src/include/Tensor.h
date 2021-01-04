#ifndef _TENSOR_HEADER_
#define _TENSOR_HEADER_

#include "include.h"

/*
To add a new tensor function:
1. Create the new kernel in tensor_kernels.cl
2. Add it to beginning of namespace Tensor as an extern
3. Add function prototype to Tensor.h
4. Add kernel to beginning of Tensor.cpp
5. Add kernel initialization code to init fxn in Tensor.cpp
6. Define the full fxn in Tensor.cpp
*/

namespace Tensor
{
    extern cl::Kernel addKernel;
    extern cl::Kernel subKernel;
    extern cl::Kernel multKernel;
    extern cl::Kernel convKernel;
    extern cl::Kernel convOptim1Kernel;
    extern cl::Kernel convOptim2Kernel;
    extern cl::Kernel reluKernel;
    extern cl::Kernel maxPoolKernel;
    extern cl::Kernel avgPoolKernel;
    extern cl::Kernel matMultKernel;
    extern cl::Kernel fcMult1Kernel;
    extern cl::Kernel fcMult2Kernel;
    extern cl::Kernel fcReduceKernel;
    extern cl::Kernel padKernel;
    extern cl::Kernel transposeKernel;
    extern cl::Kernel begProcessKernel;

    extern cl_int err;
    
    class Tensor
    {
    public:
        cl::Buffer storageBuffer;
        vector <int> dim;
        int total_size;

        Tensor(vector <int> dim, string str, int val);
        Tensor (vector <int> dim, uint8_t* values); // Make a tensor from raw stb image
        void setValue(vector <float>& values);
        vector <float>* getValue(); // User needs to free mem after using it
        void clear();

        void flatten(int one_dim);
        void reshape(vector <int> new_dim);

        void print_dim();
        void print();

        float max();
        int max_ind();
    };

    void init();
    void check_error();

    Tensor add(Tensor& T1, Tensor& T2);
    Tensor sub(Tensor& T1, Tensor& T2);
    Tensor mult(Tensor& T1, Tensor& T2);
    Tensor conv(Tensor& T, Tensor& filter, Tensor& bias, pair<int,int> stride);
    Tensor relu(Tensor& T);
    Tensor maxPool(Tensor& T, pair <int,int> filter_size, pair<int,int> stride);
    Tensor avgPool(Tensor& T, pair <int,int> filter_size, pair<int,int> stride);
    Tensor matMult(Tensor& T, Tensor& weight);
    Tensor pad(Tensor& T, pair<int,int> amt, float pad_val);
    Tensor fc(Tensor& T, Tensor& weight);
    Tensor begProcess(Tensor& T, pair <int,int> new_size, Tensor& mean, Tensor& std);
};

#endif