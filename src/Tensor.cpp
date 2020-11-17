#include "include/include.h"
#include "include/Tensor.h"

namespace Tensor
{
    cl_int err;
    cl::Kernel addKernel;
    cl::Kernel subKernel;
    cl::Kernel multKernel;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Tensor::Tensor(vector <int>& dim)
    {
        this->dim = dim;
        total_size = 1;
        for(int i = 0; i < dim.size(); i++) total_size *= dim[i];
        storageBuffer = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float)*total_size);
        //(OpenCL::clqueue).enqueueWriteBuffer(storageBuffer, CL_TRUE, 0, sizeof(float)*total_size, input.data());
    }

    void Tensor::setValue(vector <float>& values)
    {
        assert(values.size() == total_size);
        err = (OpenCL::clqueue).enqueueWriteBuffer(storageBuffer, CL_TRUE, 0, sizeof(float)*total_size, values.data());
        check_error();
    }
    vector <float>& Tensor::getValue()
    {
        vector<float>* result = new vector<float>(total_size, -1);
        err = (OpenCL::clqueue).enqueueReadBuffer(storageBuffer, CL_TRUE, 0, sizeof(float)*total_size, result->data());
        check_error();

        return *result;
    }

    void Tensor::print()
    {
        vector<float> result = getValue();
        cout << "------------- Tensor Values ----------------" << endl;
        for(int i = 0; i < result.size(); i++) cout << result[i] << " ";
        cout << endl;
        cout << "--------------------------------------------" << endl;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void init()
    {
        addKernel = cl::Kernel(OpenCL::clprogram, "tensor_add", &err); check_error();
        subKernel = cl::Kernel(OpenCL::clprogram, "tensor_sub", &err); check_error();
        multKernel = cl::Kernel(OpenCL::clprogram, "tensor_mult", &err); check_error();
    }

    void check_error()
    {
        string str;
        if(err != CL_SUCCESS)
        {
            switch(err)
            {
                // run-time and JIT compiler errors
                case 0: str = "CL_SUCCESS"; break; 
                case -1: str = "CL_DEVICE_NOT_FOUND"; break;
                case -2: str = "CL_DEVICE_NOT_AVAILABLE"; break;
                case -3: str = "CL_COMPILER_NOT_AVAILABLE"; break;
                case -4: str = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
                case -5: str = "CL_OUT_OF_RESOURCES"; break;
                case -6: str = "CL_OUT_OF_HOST_MEMORY"; break;
                case -7: str = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
                case -8: str = "CL_MEM_COPY_OVERLAP"; break;
                case -9: str = "CL_IMAGE_FORMAT_MISMATCH"; break;
                case -10: str = "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
                case -11: str = "CL_BUILD_PROGRAM_FAILURE"; break;
                case -12: str = "CL_MAP_FAILURE"; break;
                case -13: str = "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
                case -14: str = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
                case -15: str = "CL_COMPILE_PROGRAM_FAILURE"; break;
                case -16: str = "CL_LINKER_NOT_AVAILABLE"; break;
                case -17: str = "CL_LINK_PROGRAM_FAILURE"; break;
                case -18: str = "CL_DEVICE_PARTITION_FAILED"; break;
                case -19: str = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; break;

                // compile-time errors
                case -30: str = "CL_INVALID_VALUE"; break;        
                case -31: str = "CL_INVALID_DEVICE_TYPE"; break;
                case -32: str = "CL_INVALID_PLATFORM"; break;
                case -33: str = "CL_INVALID_DEVICE"; break;
                case -34: str = "CL_INVALID_CONTEXT"; break;
                case -35: str = "CL_INVALID_QUEUE_PROPERTIES"; break;
                case -36: str = "CL_INVALID_COMMAND_QUEUE"; break;
                case -37: str = "CL_INVALID_HOST_PTR"; break;
                case -38: str = "CL_INVALID_MEM_OBJECT"; break;
                case -39: str = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
                case -40: str = "CL_INVALID_IMAGE_SIZE"; break;
                case -41: str = "CL_INVALID_SAMPLER"; break;
                case -42: str = "CL_INVALID_BINARY"; break;
                case -43: str = "CL_INVALID_BUILD_OPTIONS"; break;
                case -44: str = "CL_INVALID_PROGRAM"; break;
                case -45: str = "CL_INVALID_PROGRAM_EXECUTABLE"; break;
                case -46: str = "CL_INVALID_KERNEL_NAME"; break;
                case -47: str = "CL_INVALID_KERNEL_DEFINITION"; break;
                case -48: str = "CL_INVALID_KERNEL"; break;
                case -49: str = "CL_INVALID_ARG_INDEX"; break;
                case -50: str = "CL_INVALID_ARG_VALUE"; break;
                case -51: str = "CL_INVALID_ARG_SIZE"; break;
                case -52: str = "CL_INVALID_KERNEL_ARGS"; break;
                case -53: str = "CL_INVALID_WORK_DIMENSION"; break;
                case -54: str = "CL_INVALID_WORK_GROUP_SIZE"; break;
                case -55: str = "CL_INVALID_WORK_ITEM_SIZE"; break;
                case -56: str = "CL_INVALID_GLOBAL_OFFSET"; break;
                case -57: str = "CL_INVALID_EVENT_WAIT_LIST"; break;
                case -58: str = "CL_INVALID_EVENT"; break;
                case -59: str = "CL_INVALID_OPERATION"; break;
                case -60: str = "CL_INVALID_GL_OBJECT"; break;
                case -61: str = "CL_INVALID_BUFFER_SIZE"; break;
                case -62: str = "CL_INVALID_MIP_LEVEL"; break;
                case -63: str = "CL_INVALID_GLOBAL_WORK_SIZE"; break;
                case -64: str = "CL_INVALID_PROPERTY"; break;
                case -65: str = "CL_INVALID_IMAGE_DESCRIPTOR"; break;
                case -66: str = "CL_INVALID_COMPILER_OPTIONS"; break;
                case -67: str = "CL_INVALID_LINKER_OPTIONS"; break;
                case -68: str = "CL_INVALID_DEVICE_PARTITION_COUNT"; break;

                // extension errors
                case -1000: str = "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR"; break;
                case -1001: str = "CL_PLATFORM_NOT_FOUND_KHR"; break;
                case -1002: str = "CL_INVALID_D3D10_DEVICE_KHR"; break;
                case -1003: str = "CL_INVALID_D3D10_RESOURCE_KHR"; break;
                case -1004: str = "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR"; break;
                case -1005: str = "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR"; break;
                default: str = "Unknown OpenCL error";
            }

            cout << str << endl;
            exit(0);
        }
    }

    Tensor& add(Tensor& T1, Tensor& T2)
    {
        assert(T1.dim == T2.dim); 

        Tensor* result = new Tensor(T1.dim);

        addKernel.setArg(0, T1.storageBuffer);
        addKernel.setArg(1, T2.storageBuffer);
        addKernel.setArg(2, result->storageBuffer);

        err = (OpenCL::clqueue).enqueueNDRangeKernel(addKernel, cl::NullRange, cl::NDRange(T1.total_size), cl::NullRange);
        check_error();

        return *result;
    }

    void add(Tensor& T1, Tensor& T2, Tensor& result)
    {
        assert(T1.dim == T2.dim); 
        assert(T2.dim == result.dim);

        addKernel.setArg(0, T1.storageBuffer);
        addKernel.setArg(1, T2.storageBuffer);
        addKernel.setArg(2, result.storageBuffer);

        err = (OpenCL::clqueue).enqueueNDRangeKernel(addKernel, cl::NullRange, cl::NDRange(T1.total_size), cl::NullRange);
        check_error();
    }

    Tensor& sub(Tensor& T1, Tensor& T2)
    {
        assert(T1.dim == T2.dim); 

        Tensor* result = new Tensor(T1.dim);

        subKernel.setArg(0, T1.storageBuffer);
        subKernel.setArg(1, T2.storageBuffer);
        subKernel.setArg(2, result->storageBuffer);

        err = (OpenCL::clqueue).enqueueNDRangeKernel(subKernel, cl::NullRange, cl::NDRange(T1.total_size), cl::NullRange);
        check_error();

        return *result;
    }

    void sub(Tensor& T1, Tensor& T2, Tensor& result)
    {
        assert(T1.dim == T2.dim); 
        assert(T2.dim == result.dim);

        subKernel.setArg(0, T1.storageBuffer);
        subKernel.setArg(1, T2.storageBuffer);
        subKernel.setArg(2, result.storageBuffer);

        err = (OpenCL::clqueue).enqueueNDRangeKernel(subKernel, cl::NullRange, cl::NDRange(T1.total_size), cl::NullRange);
        check_error();
    }

    Tensor& mult(Tensor& T1, Tensor& T2)
    {
        assert(T1.dim == T2.dim); 

        Tensor* result = new Tensor(T1.dim);

        multKernel.setArg(0, T1.storageBuffer);
        multKernel.setArg(1, T2.storageBuffer);
        multKernel.setArg(2, result->storageBuffer);

        err = (OpenCL::clqueue).enqueueNDRangeKernel(multKernel, cl::NullRange, cl::NDRange(T1.total_size), cl::NullRange);
        check_error();

        return *result;
    }

    void mult(Tensor& T1, Tensor& T2, Tensor& result)
    {
        assert(T1.dim == T2.dim); 
        assert(T2.dim == result.dim);

        multKernel.setArg(0, T1.storageBuffer);
        multKernel.setArg(1, T2.storageBuffer);
        multKernel.setArg(2, result.storageBuffer);

        err = (OpenCL::clqueue).enqueueNDRangeKernel(multKernel, cl::NullRange, cl::NDRange(T1.total_size), cl::NullRange);
        check_error();
    }
};