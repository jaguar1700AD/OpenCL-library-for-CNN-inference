#include "include/include.h"
#include "include/Tensor.h"

namespace Tensor
{
    cl_int err;
    cl::Kernel addKernel;
    cl::Kernel subKernel;
    cl::Kernel multKernel;
    cl::Kernel convKernel;
    cl::Kernel convOptimKernel;
    cl::Kernel reluKernel;
    cl::Kernel maxPoolKernel;
    cl::Kernel avgPoolKernel;
    cl::Kernel matMultKernel;
    cl::Kernel padKernel;
    cl::Kernel begProcessKernel;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Tensor::Tensor(vector <int> dim, string str, int val)
    {
        this->dim = dim;
        total_size = 1;
        for(int i = 0; i < dim.size(); i++) total_size *= dim[i];
        storageBuffer = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float)*total_size);

        if (str == "inc")
        {
            vector <float> data;
            for(int i = 0; i < total_size; i++) data.push_back(i);
            setValue(data);
        }
        else if (str == "const")
        {
            vector <float> data(total_size, val);
            setValue(data);
        }
        
    }
    Tensor::Tensor(vector <int> dim, uint8_t* values) // Make a tensor from raw stb image
    {
        this->dim = dim;
        total_size = 1;
        for(int i = 0; i < dim.size(); i++) total_size *= dim[i];
        storageBuffer = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(uint8_t)*total_size);
        err = (OpenCL::clqueue).enqueueWriteBuffer(storageBuffer, CL_TRUE, 0, sizeof(uint8_t)*total_size, values);
        check_error();
    }

    void Tensor::setValue(vector <float>& values)
    {
        assert(values.size() == total_size);
        err = (OpenCL::clqueue).enqueueWriteBuffer(storageBuffer, CL_TRUE, 0, sizeof(float)*total_size, values.data());
        check_error();
    }
    vector <float>* Tensor::getValue() 
    {
        vector<float>* result = new vector<float>(total_size, -1);
        if (total_size > 0)
        {
            err = (OpenCL::clqueue).enqueueReadBuffer(storageBuffer, CL_TRUE, 0, sizeof(float)*total_size, result->data());
            check_error();
        }

        return result;
    }

    void Tensor::clear()
    {
        storageBuffer = NULL;
        dim.clear();
        total_size = 0;
    }

    void Tensor::flatten(int one_dim)
    {
        dim.clear();
        if (one_dim == 0)
        {
            dim.push_back(1);
            dim.push_back(total_size);
        }
        else if (one_dim == 1)
        {
            dim.push_back(total_size);
            dim.push_back(1);
        }
        else
        {
            cout << "Invalid Option in flatten" << endl;
        }
    }

    void Tensor::reshape(vector <int> new_dim)
    {
        int new_tot_size = 1;
        for(int i = 0; i < new_dim.size(); i++) new_tot_size *= new_dim[i];
        assert(total_size == new_tot_size);

        dim = new_dim;
    }

    void Tensor::print_dim()
    {
        cout << "------------- Tensor Dimensions ----------------" << endl;
        cout << "Dim: ";
        for(int i = 0; i < dim.size(); i++) cout << dim[i] << " ";
        cout << endl;
        cout << "--------------------------------------------" << endl;
    }

    void Tensor::print()
    {
        vector<float>* result = getValue();
        cout << "------------- Tensor Values ----------------" << endl;
        cout << "Dim: ";
        for(int i = 0; i < dim.size(); i++) cout << dim[i] << " ";
        cout << endl;
        for(int i = 0; i < (*result).size(); i++) cout << (*result)[i] << " ";
        cout << endl;
        cout << "--------------------------------------------" << endl;
        delete(result);
    }

    float Tensor::max()
    {
        vector<float>* result = getValue();
        float max_val = (*result)[0];
        for(int i = 1; i < total_size; i++) max_val = std::max(max_val, (*result)[i]);
        delete(result);
        return max_val;
    }

    int Tensor::max_ind()
    {
        assert(dim.size() == 1);
        vector<float>* result = getValue();
        float max_val = (*result)[0];
        int max_ind = 0;
        for(int i = 1; i < total_size; i++) 
        {
            if ((*result)[i] > max_val)
            {
                max_val = (*result)[i];
                max_ind = i;
            }
        }
        delete(result);
        return max_ind;
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void init()
    {
        addKernel = cl::Kernel(OpenCL::clprogram, "tensor_add", &err); check_error();
        subKernel = cl::Kernel(OpenCL::clprogram, "tensor_sub", &err); check_error();
        multKernel = cl::Kernel(OpenCL::clprogram, "tensor_mult", &err); check_error();
        convKernel = cl::Kernel(OpenCL::clprogram, "tensor_conv", &err); check_error();
        convOptimKernel = cl::Kernel(OpenCL::clprogram, "tensor_conv_optim", &err); check_error();
        reluKernel = cl::Kernel(OpenCL::clprogram, "tensor_relu", &err); check_error();
        maxPoolKernel = cl::Kernel(OpenCL::clprogram, "tensor_maxPool", &err); check_error();
        avgPoolKernel = cl::Kernel(OpenCL::clprogram, "tensor_avgPool", &err); check_error();
        matMultKernel = cl::Kernel(OpenCL::clprogram, "tensor_matMult", &err); check_error();
        padKernel = cl::Kernel(OpenCL::clprogram, "tensor_pad", &err); check_error();
        begProcessKernel = cl::Kernel(OpenCL::clprogram, "tensor_begProcess", &err); check_error();
    }

    void check_error()
    {
        static int iter = 0;

        iter++;
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

            cout << "My Error " << iter << " : " << str << endl;
            exit(0);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Tensor add(Tensor& T1, Tensor& T2)
    {
        assert(T1.dim == T2.dim); 

        Tensor result(T1.dim, "", -1);

        addKernel.setArg(0, T1.storageBuffer);
        addKernel.setArg(1, T2.storageBuffer);
        addKernel.setArg(2, result.storageBuffer);

        err = (OpenCL::clqueue).enqueueNDRangeKernel(addKernel, cl::NullRange, cl::NDRange(T1.total_size), cl::NullRange);
        check_error();

        return result;
    }

    Tensor sub(Tensor& T1, Tensor& T2)
    {
        assert(T1.dim == T2.dim); 

        Tensor result(T1.dim, "", -1);

        subKernel.setArg(0, T1.storageBuffer);
        subKernel.setArg(1, T2.storageBuffer);
        subKernel.setArg(2, result.storageBuffer);

        err = (OpenCL::clqueue).enqueueNDRangeKernel(subKernel, cl::NullRange, cl::NDRange(T1.total_size), cl::NullRange);
        check_error();

        return result;
    }

    Tensor mult(Tensor& T1, Tensor& T2)
    {
        assert(T1.dim == T2.dim); 
        
        Tensor result(T1.dim, "", -1);

        multKernel.setArg(0, T1.storageBuffer);
        multKernel.setArg(1, T2.storageBuffer);
        multKernel.setArg(2, result.storageBuffer);

        err = (OpenCL::clqueue).enqueueNDRangeKernel(multKernel, cl::NullRange, cl::NDRange(T1.total_size), cl::NullRange);
        check_error();

        return result;
    }

    Tensor conv(Tensor& T, Tensor& filter, Tensor& bias,  pair<int,int> stride)
    {
        assert(T.dim.size() == 3);
        assert(filter.dim.size() == 4);
        assert(bias.dim.size() == 1);
        assert(T.dim[0] == filter.dim[1]); 
        assert(T.dim[1] >= filter.dim[2]); 
        assert(T.dim[2] >= filter.dim[3]);
        assert(filter.dim[0] == bias.dim[0]); 

        // global float *image, global float* filters, global float* bias, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int strider, stridec

        int num_filters = filter.dim[0];
        int inr = T.dim[1], inc = T.dim[2];
        int kr = filter.dim[2], kc = filter.dim[3];
        int inz = T.dim[0];
        int strider = stride.first, stridec = stride.second;

        int outr = (inr - kr) / strider + 1;
        int outc = (inc - kc) / stridec + 1;

        Tensor result(vector <int> {num_filters, outr, outc}, "", -1);

        // convKernel.setArg(0, T.storageBuffer);
        // convKernel.setArg(1, filter.storageBuffer);
        // convKernel.setArg(2, bias.storageBuffer);
        // convKernel.setArg(3, result.storageBuffer);
        // convKernel.setArg(4, inr);
        // convKernel.setArg(5, inc);
        // convKernel.setArg(6, inz);
        // convKernel.setArg(7, kr);
        // convKernel.setArg(8, kc);
        // convKernel.setArg(9, outr);
        // convKernel.setArg(10, outc);
        // convKernel.setArg(11, strider);
        // convKernel.setArg(12, stridec);

        // cl::NDRange global_dim = cl::NDRange(num_filters, outr, outc);
        // err = (OpenCL::clqueue).enqueueNDRangeKernel(convKernel, cl::NullRange, global_dim, cl::NullRange);
        // check_error();

        int depth_per_iter = 2;
        int eff_depth = min(depth_per_iter, inz);

        int localRow = 8; // Number of consecutive output row positions to be computed in one work group using local memory
        int localCol = 8; // Number of consecutive output col positions to be computed in one work group using local memory
        int local_image_mem_size = eff_depth * (kr + strider * (localRow - 1)) * (kc + stridec * (localCol - 1));
        int local_filter_mem_size = eff_depth * kr * kc;

        // global float *image, int depth_per_iter, global float* filters, local float* image_local, local float* filter_local, global float* bias, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int oz, int strider, int stridec

        convOptimKernel.setArg(0, T.storageBuffer);
        convOptimKernel.setArg(1, depth_per_iter);
        convOptimKernel.setArg(2, filter.storageBuffer);
        convOptimKernel.setArg(3, sizeof(float) * local_image_mem_size, nullptr);
        convOptimKernel.setArg(4, sizeof(float) * local_filter_mem_size, nullptr);
        convOptimKernel.setArg(5, bias.storageBuffer);
        convOptimKernel.setArg(6, result.storageBuffer);
        convOptimKernel.setArg(7, inr);
        convOptimKernel.setArg(8, inc);
        convOptimKernel.setArg(9, inz);
        convOptimKernel.setArg(10, kr);
        convOptimKernel.setArg(11, kc);
        convOptimKernel.setArg(12, outr);
        convOptimKernel.setArg(13, outc);
        convOptimKernel.setArg(14, num_filters);
        convOptimKernel.setArg(15, strider);
        convOptimKernel.setArg(16, stridec);
        
        int globalRow = localRow * ceil(outr / (float) localRow);
        int globalCol = localCol * ceil(outc / (float) localCol);
        cl::NDRange global_dim = cl::NDRange(num_filters, globalRow, globalCol);
        cl::NDRange local_dim = cl::NDRange(1, localRow, localCol);
        err = (OpenCL::clqueue).enqueueNDRangeKernel(convOptimKernel, cl::NullRange, global_dim, local_dim);
        check_error();

        // OpenCL::clqueue.finish();
        // cout << "Reached" << endl;

        return result;
    }

    Tensor relu(Tensor& T)
    {
        Tensor result(T.dim, "", -1);
        
        reluKernel.setArg(0, T.storageBuffer);
        reluKernel.setArg(1, result.storageBuffer);

        err = (OpenCL::clqueue).enqueueNDRangeKernel(reluKernel, cl::NullRange, cl::NDRange(T.total_size), cl::NullRange);
        check_error();

        return result;
    }

    Tensor maxPool(Tensor& T, pair<int,int> filter_size, pair<int,int> stride)
    {
        assert(T.dim.size() == 3); 
        assert(T.dim[1] >= filter_size.first); 
        assert(T.dim[2] >= filter_size.second);

        // global float *image, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int strider, stridec

        int inr = T.dim[1], inc = T.dim[2];
        int kr = filter_size.first, kc = filter_size.second;
        int inz = T.dim[0];
        int strider = stride.first, stridec = stride.second;

        int outr = (inr - kr) / strider + 1;
        int outc = (inc - kc) / stridec + 1;

        Tensor result(vector <int> {inz, outr, outc}, "", -1);

        maxPoolKernel.setArg(0, T.storageBuffer);
        maxPoolKernel.setArg(1, result.storageBuffer);
        maxPoolKernel.setArg(2, inr);
        maxPoolKernel.setArg(3, inc);
        maxPoolKernel.setArg(4, inz);
        maxPoolKernel.setArg(5, kr);
        maxPoolKernel.setArg(6, kc);
        maxPoolKernel.setArg(7, outr);
        maxPoolKernel.setArg(8, outc);
        maxPoolKernel.setArg(9, strider);
        maxPoolKernel.setArg(10, stridec);

        cl::NDRange global_dim = cl::NDRange(inz, outr, outc);
        err = (OpenCL::clqueue).enqueueNDRangeKernel(maxPoolKernel, cl::NullRange, global_dim, cl::NullRange);
        check_error();

        return result;
    }

    Tensor avgPool(Tensor& T, pair<int,int> filter_size, pair<int,int> stride)
    {
        assert(T.dim.size() == 3); 
        assert(T.dim[1] >= filter_size.first); 
        assert(T.dim[2] >= filter_size.second);

        // global float *image, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int strider, stridec

        int inr = T.dim[1], inc = T.dim[2];
        int kr = filter_size.first, kc = filter_size.second;
        int inz = T.dim[0];
        int strider = stride.first, stridec = stride.second;

        int outr = (inr - kr) / strider + 1;
        int outc = (inc - kc) / stridec + 1;

        Tensor result(vector <int> {inz, outr, outc}, "", -1);

        avgPoolKernel.setArg(0, T.storageBuffer);
        avgPoolKernel.setArg(1, result.storageBuffer);
        avgPoolKernel.setArg(2, inr);
        avgPoolKernel.setArg(3, inc);
        avgPoolKernel.setArg(4, inz);
        avgPoolKernel.setArg(5, kr);
        avgPoolKernel.setArg(6, kc);
        avgPoolKernel.setArg(7, outr);
        avgPoolKernel.setArg(8, outc);
        avgPoolKernel.setArg(9, strider);
        avgPoolKernel.setArg(10, stridec);

        cl::NDRange global_dim = cl::NDRange(inz, outr, outc);
        err = (OpenCL::clqueue).enqueueNDRangeKernel(avgPoolKernel, cl::NullRange, global_dim, cl::NullRange);
        check_error();

        return result;
    }

    Tensor matMult(Tensor& T, Tensor& weight)
    {
        assert(T.dim.size() == 2);
        assert(weight.dim.size() == 2);
        assert(T.dim[1] == weight.dim[0]);

        int m = T.dim[0];
        int size = T.dim[1];
        int n = weight.dim[1];

        Tensor result(vector <int> {m, n}, "", -1);

        // global float* image, global float* weights, global float* out, int size, int m, int n

        matMultKernel.setArg(0, T.storageBuffer);
        matMultKernel.setArg(1, weight.storageBuffer);
        matMultKernel.setArg(2, result.storageBuffer);
        matMultKernel.setArg(3, size);
        matMultKernel.setArg(4, m);
        matMultKernel.setArg(5, n);

        cl::NDRange global_dim = cl::NDRange(m, n);
        err = (OpenCL::clqueue).enqueueNDRangeKernel(matMultKernel, cl::NullRange, global_dim, cl::NullRange);
        check_error();

        return result;
    }

    Tensor pad(Tensor& T, pair<int,int> amt, float pad_val)
    {
        assert(T.dim.size() == 3);
        
        int iz = T.dim[0], ir = T.dim[1], ic = T.dim[2];
        int padr = amt.first, padc = amt.second;
        
        int outr = ir + 2 * padr;
        int outc = ic + 2 * padc;

        Tensor result(vector <int> {iz, outr, outc}, "", -1);

        // global float* image, global float* out, int ir, int ic, int iz, int padr, int padc, float pad_val

        padKernel.setArg(0, T.storageBuffer);
        padKernel.setArg(1, result.storageBuffer);
        padKernel.setArg(2, ir);
        padKernel.setArg(3, ic);
        padKernel.setArg(4, iz);
        padKernel.setArg(5, padr);
        padKernel.setArg(6, padc);
        padKernel.setArg(7, pad_val);

        cl::NDRange global_dim = cl::NDRange(iz, outr, outc);
        err = (OpenCL::clqueue).enqueueNDRangeKernel(padKernel, cl::NullRange, global_dim, cl::NullRange);
        check_error();

        return result;
    }

    Tensor fc(Tensor& T, Tensor& weight)
    {

        T.dim.push_back(1);
        Tensor result = matMult(weight, T);
        T.dim.pop_back();
        result.dim.pop_back();

        return result;
    }

    Tensor begProcess(Tensor& T, pair <int,int> new_size, Tensor& mean, Tensor& std)
    {
        // Used for resizing and processing raw images provided by stbi library and bringing them to correct format
        // Input contains uint8_t data

        assert(T.dim.size() == 3); 

        int ir = T.dim[0], ic = T.dim[1], iz = T.dim[2]; // Different from normal tensor format in which channels come first
        int outr = new_size.first, outc = new_size.second;

        assert(mean.dim.size() == 1); assert(mean.total_size == iz);
        assert(std.dim.size() == 1); assert(std.total_size == iz);

        Tensor result(vector <int> {iz, outr, outc}, "", -1); // Normal tensor format

        // global uint8_t* image, global float* out, int ir, int ic, int iz, int or, int oc, float* mean, float* std

        begProcessKernel.setArg(0, T.storageBuffer);
        begProcessKernel.setArg(1, result.storageBuffer);
        begProcessKernel.setArg(2, ir);
        begProcessKernel.setArg(3, ic);
        begProcessKernel.setArg(4, iz);
        begProcessKernel.setArg(5, outr);
        begProcessKernel.setArg(6, outc);
        begProcessKernel.setArg(7, mean.storageBuffer);
        begProcessKernel.setArg(8, std.storageBuffer);

        err = (OpenCL::clqueue).enqueueNDRangeKernel(begProcessKernel, cl::NullRange, cl::NDRange(outr, outc), cl::NullRange);
        check_error();

        return result;
    }
};