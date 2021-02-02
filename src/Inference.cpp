#define __CL_ENABLE_EXCEPTIONS

#include "include/include.h"
#include "include/OpenCL.h"
#include "include/util.h"
#include "include/Tensor.h"
#include "include/ConvNets.h"

#include <stdint.h>
#include <stdio.h>
#include <dirent.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "include/stb_image_resize.h"

#define ID 3    // Image Depth
#define IW 224  // Image Width
#define IH 224  // Image Height

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool char_compare(char* str1, char* str2)
{
	string s1 = str1, s2 = str2;
	if (s1 < s2) return true;
	return false;
}

char* concat(const char* str1, const char* str2, bool is_dir)
{
	char* ret = new char[1000];
	strcpy(ret, str1);
	strcat(ret, str2);

	if (is_dir) strcat(ret, "/");

	return ret;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////

vector <char*>* sorted_dir_entries(char* path)
{
	vector <char*>* result = new vector <char*>;

	struct dirent *entry = nullptr;
    DIR *dp = nullptr;
    dp = opendir(path);
	while ((entry = readdir(dp)))
	{
		if (entry->d_name[0] == '.') continue;
		char* str = new char[strlen(entry->d_name) + 1];
		strcpy(str, entry->d_name);
		result->push_back(str);
	}
	closedir(dp);

	sort(result->begin(), result->end(), char_compare);

	return result;
}

void check_accuracy()
{
	vector <float> mean = {0.485, 0.456, 0.406};
	vector <float> std = {0.229, 0.224, 0.225};
	Tensor::Tensor tensor_mean(vector <int> {ID}, "", -1); tensor_mean.setValue(mean); 
	Tensor::Tensor tensor_std(vector <int> {ID}, "", -1); tensor_std.setValue(std); 
	
	int image_upp_bound = 1000;
	int num_timer = 3;
	int CNN_timer_size = 27;
	util::Timer timer[num_timer];
	vector <util::Timer> CNN_timer(CNN_timer_size, util::Timer());

	AlexNet CNN(false);
	CNN.readData("Alexnet");
	//CNN.printModel();

	char base[] = "./src/data/Datasets/Imagenet/train/";
	vector <char*>* sub_dirs = sorted_dir_entries(base);

	int category = 0;
	int correct = 0;
	int tot_image = 0;
	for(int i = 0; i < (*sub_dirs).size(); i++)
	{
		char* sub_dir_path = concat(base, (*sub_dirs)[i], true);
		
		DIR* sub_dp = opendir(sub_dir_path);
		struct dirent* entry = nullptr;
		while((entry = readdir(sub_dp)))
		{
			timer[0].reset();

			if (entry->d_name[0] == '.') continue;

			int width, height, depth;
			char* path = concat(sub_dir_path, entry->d_name, false);

			timer[1].reset();
			uint8_t* rgb_image = stbi_load(path, &width, &height, &depth, ID);
			timer[1].record();
			
			Tensor::Tensor X(vector <int> {height, width, ID}, rgb_image);
			X = Tensor::begProcess(X, make_pair(IH, IW), tensor_mean, tensor_std);

			stbi_image_free(rgb_image);
			
			timer[2].reset();
			X = CNN.timed_forward(X, CNN_timer);
			timer[2].record();
			
			int predn = X.max_ind();
			//int predn = 0;

			if (predn == category) correct++;
			tot_image++;
			if (tot_image > image_upp_bound) break;

			cout << "\r" << correct << " / " << tot_image << " ( " << (correct * 100.0 / tot_image) << "% ) " << flush;

			delete(path);

			timer[0].record();
		}

		delete(sub_dir_path);
		closedir(sub_dp);
		category++;

		if (tot_image > image_upp_bound) break;
	}

	sub_dirs->clear();
	delete(sub_dirs);

	util::Timer tot_expected_time;
	util::Timer tot_CNN_time;

	cout << endl;
	timer[0].print("Total time");
	for(int i = 1; i < num_timer; i++) 
	{
		timer[i].print("");
		tot_expected_time.recorded_time += timer[i].recorded_time;
	}
	tot_expected_time.print("Total expected time");
	//vector <int> relevant_inds = {0, 1, 4, 5, 8, 9, 11, 12, 14, 15};
	vector <int> relevant_inds = {19, 20, 22, 23, 25, 26};
	//vector <int> relevant_inds; for (int i = 0; i < CNN_timer_size; i++) relevant_inds.push_back(i);
	for(int i = 0; i < relevant_inds.size(); i++) 
	{
		CNN_timer[relevant_inds[i]].print(to_string(relevant_inds[i]));
		//tot_CNN_time.recorded_time += CNN_timer[i].recorded_time;
	}
	//tot_CNN_time.print("Total CNN expected time");
}

int main(void)
{
	OpenCL::initialize_OpenCL();
	Tensor::init();
	check_accuracy();

	// OpenCL::gen_m2s_binary();

    return 0;
}

