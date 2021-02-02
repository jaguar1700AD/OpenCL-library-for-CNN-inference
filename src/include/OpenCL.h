#pragma once

class OpenCL {


public:

	static cl::Program clprogram;
	static cl::CommandQueue clqueue;
	static cl::Context clcontext;

	static void initialize_OpenCL();

	static void gen_m2s_binary();
};



