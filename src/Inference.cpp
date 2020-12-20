#define __CL_ENABLE_EXCEPTIONS

#include "include/OpenCL.h"
#include "include/util.h"
#include "include/Tensor.h"
#include "include/ConvNets.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdint.h>
#include <stdio.h>
#include <dirent.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "include/stb_image_resize.h"

#define IW 224
#define IH 224

void read_Mnist(std::string filename, std::vector<std::vector<float>> &vec);
void read_Mnist_Label(std::string filename, std::vector<std::vector<float>> &vec, std::vector<float> &testtargets, bool testflag);
void printInput(std::vector<float> &inputs);
void read_CIFAR10(cv::Mat &trainX, cv::Mat &testX, cv::Mat &trainY, cv::Mat &testY);

void test_add()
{
	vector <int> vec {2, 2};
	Tensor::Tensor T1(vec, "inc", -1); 
	T1.print();

	Tensor::Tensor T2(vec, "inc", -1); 
	T2.print();

	Tensor::Tensor result = Tensor::add(T1, T2);
	result.print();
}

void test_sub()
{
	vector <int> vec {2, 2};
	Tensor::Tensor T1(vec, "inc", -1); 
	T1.print();

	Tensor::Tensor T2(vec, "const", 1); 
	T2.print();

	Tensor::Tensor result = Tensor::sub(T1, T2);
	result.print();

	result = Tensor::relu(result);
	result.print();
}

void test_mult()
{
	vector <int> vec {2, 2};
	Tensor::Tensor T1(vec, "inc", -1); 
	T1.print();

	Tensor::Tensor T2(vec, "inc", -1); 
	T2.print();

	Tensor::Tensor result = Tensor::mult(T1, T2);
	result.print();
}

void test_conv()
{
	vector <int> vec {2, 2, 3, 3};
	Tensor::Tensor filters(vec, "inc", -1); 
	filters.print();

	vector <int> vec2 {2, 13, 13};
	Tensor::Tensor input(vec2, "inc", -1); 
	input.print();

	vector <int> vec3 {2};
	Tensor::Tensor bias(vec3, "const", 0);

	pair <int, int> stride {2, 3};
	Tensor::Tensor result = Tensor::conv(input, filters, bias, stride);
	result.print();

	pair <int, int> filter_size {3, 2};
	stride = make_pair(2,2);
	result = Tensor::avgPool(result, filter_size, stride);
	result.print();
}

void test_matMult()
{
	vector <int> vec {2, 3};
	Tensor::Tensor T1(vec, "inc", -1); 
	T1.print();

	vector <int> vec2 {3, 2};
	Tensor::Tensor T2(vec2, "inc", -1); 
	T2.print();

	Tensor::Tensor result = Tensor::matMult(T1, T2);
	vector <int> vec3 {1,2,2};
	result.reshape(vec3);
	result.print();

	Tensor::Tensor result1 = Tensor::pad(result, make_pair(1, 2), 1);
	result1.print();
}

vector <float>& process(uint8_t* input_image, int width, int height, int depth)
{
	float mean[] = {0.485, 0.456, 0.406};
	float std[] = {0.229, 0.224, 0.225};

	uint8_t* image = (uint8_t*) malloc (sizeof(uint8_t) * IW * IH * depth);
	stbir_resize_uint8(input_image, width, height, 0, image, IW, IH, 0, depth);

	vector <float>* ret_image = new vector<float>;
	for(int channel = 0; channel < depth; channel++)
	{
		for(int row = 0; row < IH; row++)
		{
			for(int col = 0; col < IW; col++)
			{
				float value = image[row * IW * depth + col * depth + channel] / 255.0;
				value = (value - mean[channel]) / std[channel];
				ret_image->push_back(value);
			}
		}
	}
	return *ret_image;
}

void check_accuracy()
{
	int width, height, depth;

	char path[] = "./src/data/Datasets/Imagenet/train/n01440764/ILSVRC2012_val_00010306.JPEG";
    uint8_t* rgb_image = stbi_load(path, &width, &height, &depth, 3);
	cout << height << " " << width << " " << depth << endl;
	
	vector <float> image = process(rgb_image, width, height, depth);
	for(int i = 0; i < depth; i++)
	{
		cout << image[i * IW * IH] << endl;
	}

	stbi_image_free(rgb_image);

	struct dirent *entry = nullptr;
    DIR *dp = nullptr;
	char base[] = "./src/data/Datasets/Imagenet/val/";
    dp = opendir(base);
	while ((entry = readdir(dp)))
	{
		if (entry->d_name[0] == '.') continue;

		char sub_name[1000] = "";
		strcat(sub_name, base);
		strcat(sub_name, entry->d_name);

		cout << entry->d_name << "-------------" << endl;
		
		DIR* sub_dp = opendir(sub_name);
		while((entry = readdir(sub_dp)))
		{
			if (entry->d_name[0] == '.') continue;
			cout << entry->d_name << endl;
		}
		closedir(sub_dp);

	}
    closedir(dp);

}

int main(void)
{


	// try {

	// 	OpenCL::initialize_OpenCL();

	// 	util::Timer timer;

	// 	timer.reset();




	// 	std::vector<std::vector<float> > inputs, targets;
	// 	std::vector<std::vector<float> > testinputs;
	// 	std::vector<float> testtargets;



	// 	/*//////////////////////////////////
	// 	std::vector<float> intemp(28 * 28);

	// 	for (int i = 0; i < 28 * 28; i++) {
	// 		intemp.at(i) = 0.5;
	// 	}

	// 	for (int j = 0; j < 10000; j++)
	// 		inputs.push_back(intemp);


	// 	std::vector<float> temp(10);

	// 	for (int i = 0; i < 1; i++)
	// 		temp.at(i) = 0;
	// 	temp.at(1) = 1;

	// 	for (int j = 0; j < 10000; j++)
	// 		targets.push_back(temp);

	// 	testinputs = inputs;
	// 	for (int i = 0; i < 10000; i++)
	// 		testtargets.push_back(1);
		

	// 	////////////////////////////////////////////////////////*/

	// 	///MNIST
	// 	/*//////////////////////////////////////////////////
	// 	read_Mnist("train-images.idx3-ubyte", inputs);
	// 	read_Mnist_Label("train-labels.idx1-ubyte", targets,testtargets,0);


	// 	std::cout << "MNIST loaded in: " <<timer.getTimeMilliseconds()/1000.0 <<" s"<<std::endl;

	// 	timer.reset();
	// 	read_Mnist("t10k-images.idx3-ubyte", testinputs);
	// 	read_Mnist_Label("t10k-labels.idx1-ubyte", targets, testtargets, 1);

	// 	//for (int i = 0; i < 30; i++)
	// 		//std::cout << " " <<testtargets[i];
	// 	std::cout << "MNIST test loaded in: " << timer.getTimeMilliseconds() / 1000.0 << " s" << std::endl;

	// 	//printInput(inputs[54]);

	// 	////////////////////////////////////////////////////*/


	// 	///CIFAR10
	// 	/////////////////////////////////////////////////////////
	// 	cv::Mat trainX, testX;

	// 	cv::Mat trainY, testY;

	// 	trainX = cv::Mat::zeros(1024, 50000, CV_32FC1);

	// 	testX = cv::Mat::zeros(1024, 10000, CV_32FC1);

	// 	trainY = cv::Mat::zeros(1, 50000, CV_32FC1);

	// 	testY = cv::Mat::zeros(1, 10000, CV_32FC1);


	// 	//read_CIFAR10(trainX, testX, trainY, testY);



	// 	std::cout << "Cifar10 loaded in: " << timer.getTimeMilliseconds() / 1000.0 << " s" << std::endl;

		


	// 	timer.reset();


	// 	for (int i = 0; i < 50000; i++) {
	// 		inputs.push_back(trainX.col(i));

	// 		std::vector<float> tempvec(10);

	// 		for (int j = 0; j < 10; j++) {
	// 			if (j == trainY.col(i).at<float>(0))
	// 				tempvec[j] = (float)1.0;
	// 			else
	// 				tempvec[j] = (float) 0.0;
	// 		}
	// 		targets.push_back(tempvec);

	// 	}
	// 	for (int i = 0; i < 10000; i++) {
	// 		testinputs.push_back(testX.col(i));
	// 		testtargets.push_back(testY.col(i).at<float>(0));

	// 	}



	// 	std::cout << "Cifar10 converted in: " << timer.getTimeMilliseconds() / 1000.0 << " s" << std::endl;

	// 	timer.reset();




	// 	////////////////////////////////////////////////////////*/

	// 	///CNN
	// 	//////////////////////////////////////////////////////////

	// 	ConvNN m_nn;
	// 	m_nn.createConvNN(7, 7, 32);//num of filters,filterdim,imagedim


	// 	//todo::many filters  3d kernel
	// 	std::vector<int> netVec;
	// 	netVec = { 169 * 7,10 };
	// 	m_nn.createFullyConnectedNN(netVec, 0, 32);

	

	// 	m_nn.train(inputs, targets, testinputs, testtargets, 1000000);

	// 	std::cout << "trained in : " << timer.getTimeMilliseconds() / 1000.0 << " s" << std::endl;



	// 	//////////////////////////////////////////////////////////////////////////////*/

	// 	/// FCNN
	// 	/*////////////////////////////////////////////////////

	//    ConvNN m_nn;

	//    std::vector<int> netVec;
	//    netVec = { 1024,10 };
	//    m_nn.createFullyConnectedNN(netVec, 1, 32);


	//    //m_nn.forwardFCNN(inputs[0]);


	//    m_nn.trainFCNN(inputs, targets, testinputs, testtargets, 50000);

	//    std::cout << "trained in : " << timer.getTimeMilliseconds() / 1000.0 << " s" << std::endl;

	//    m_nn.trainingAccuracy(testinputs, testtargets, 2000, 1);
	//    /////////////////////////////////////////////////////////////*/

	// }
	

	// catch (cl::Error e) 
	// {
	// 	std::cout << "opencl error: " << e.what() << std::endl;
	// 	std::cout << "error number: " << e.err() << std::endl;
	// }
	// catch (int e)
	// {
	// 	std::cout << "An exception occurred. Exception Nr. " << e << '\n';
	// }

	// OpenCL::initialize_OpenCL();
	// Tensor::init();

	// AlexNet CNN(false);
	// CNN.readData("Alexnet");
	// CNN.printModel();

	// for(int i = 0; i < 1000; i++)
	// {
	// 	Tensor::Tensor input(vector<int> {3,224,224}, "inc", -1);
	// 	input = CNN.forward(input);
	// 	cout << i << endl;
	// }

    return 0;
}

