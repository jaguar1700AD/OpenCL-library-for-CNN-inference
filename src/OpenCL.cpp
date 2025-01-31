
#include "include/include.h"
#include "include/OpenCL.h"
#include "include/util.h"

using namespace std;

cl::Program OpenCL::clprogram;
cl::Context OpenCL::clcontext;
cl::CommandQueue OpenCL::clqueue;

void OpenCL::initialize_OpenCL() {
	// get all platforms (drivers), e.g. NVIDIA
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	// get default device (CPUs, GPUs) of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}

	for(int i = 0; i < all_devices.size(); i++) cout << all_devices[i].getInfo<CL_DEVICE_NAME>() << endl;
	
	cl::Device default_device = all_devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
	std::cout << "\t-------------------------" << std::endl;

	std::string s;
	default_device.getInfo(CL_DEVICE_NAME, &s);
	std::cout << "\t\tName: " << s << std::endl;

	//default_device.getInfo(CL_DEVICE_OPENCL_C_VERSION, &s);
	//std::cout << "\t\tVersion: " << s << std::endl;

	int i;
	default_device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &i);
	std::cout << "\t\tMax. Compute Units: " << i << std::endl;

	size_t size;
	default_device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	std::cout << "\t\tLocal Memory Size: " << size / 1024 << " KB" << std::endl;

	default_device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &size);
	std::cout << "\t\tGlobal Memory Size: " << size / (1024 * 1024) << " MB" << std::endl;

	default_device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
	std::cout << "\t\tMax Alloc Size: " << size / (1024 * 1024) << " MB" << std::endl;

	default_device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
	std::cout << "\t\tMax Work-group Total Size: " << size << std::endl;

	std::vector<size_t> d;
	default_device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &d);
	std::cout << "\t\tMax Work-group Dims: (";
	for (std::vector<size_t>::iterator st = d.begin(); st != d.end(); st++)
		std::cout << *st << " ";
	std::cout << "\x08)" << std::endl;

	std::cout << "\t-------------------------" << std::endl;

	// a context is like a "runtime link" to the device and platform;
	// i.e. communication is possible
	OpenCL::clcontext=cl::Context(default_device);

	// create the program that we want to execute on the device
	cl::Program::Sources sources;

	
	std::string src, src2, src3, src4;
	
	src = util::loadProgram("tensor_kernels.cl");
	sources.push_back({ src.c_str(), src.length() });

	OpenCL::clprogram=cl::Program(OpenCL::clcontext, sources);
	OpenCL::clprogram.build({ default_device });
	auto buildInfo = OpenCL::clprogram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device);
	cout << buildInfo << std::endl << std::endl;
		

	OpenCL::clqueue=cl::CommandQueue(OpenCL::clcontext, default_device);

}

void OpenCL::gen_m2s_binary()
{

	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	cout << "Num Platforms: " << all_platforms.size() << endl;

	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	cout << default_platform.getInfo<CL_PLATFORM_EXTENSIONS>() << endl;

	/* create context */
	cl_context_properties cprops[5] = 
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties) (default_platform) (),
		CL_CONTEXT_OFFLINE_DEVICES_AMD,
		(cl_context_properties) 1,
		(cl_context_properties) 0
	};
	//cl_int err;


	cl::Context myContext(CL_DEVICE_TYPE_ALL, cprops, NULL, NULL, &Tensor::err); Tensor::check_error(); 
	// vector <cl::Device> devices = myContext.getInfo<CL_CONTEXT_DEVICES>();

	// cout << devices.size() << endl;
	// for(int i = 0; i < devices.size(); i++)
	// {
	// 	cout << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
	// }

	// cl_context context = clCreateContextFromType(cprops, CL_DEVICE_TYPE_ALL, NULL, NULL, &Tensor::err); Tensor::check_error();
	// get number of devices
	// cl_int n_all_devices;
	// err = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(n_all_devices), &n_all_devices, NULL);
	// // get all device IDs
	// cl_device_id* all_devices = (cl_device_id*) malloc(sizeof(cl_device_id)*n_all_devices);
	// err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id)*n_all_devices, all_devices, NULL);

	
}



