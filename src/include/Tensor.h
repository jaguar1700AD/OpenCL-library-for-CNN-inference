#ifndef _TENSOR_HEADER_
#define _TENSOR_HEADER_

#include "include.h"

namespace Tensor
{
    extern cl::Kernel addKernel;
    extern cl_int err;
    
    class Tensor
    {
    public:
        cl::Buffer storageBuffer;
        vector <int> dim;
        int total_size;

        Tensor(vector <int>& dim);
        void setValue(vector <float>& values);
        vector <float>& getValue();

        void print();
    };

    void init();
    void check_error();

    Tensor& add(Tensor& T1, Tensor& T2);
    void add(Tensor& T1, Tensor& T2, Tensor& result);
};

#endif