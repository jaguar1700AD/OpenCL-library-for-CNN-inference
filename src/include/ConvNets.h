#ifndef _ConvNets_HEADER_
#define _ConvNets_HEADER_

#include "include.h"
#include "Tensor.h"

//////////////////////////////////////////////////////////////////////////////

class ConvNet
{
private:
    void readTensorSet(vector <Tensor::Tensor>& tensor_store, string file_name);

public:
    vector <Tensor::Tensor> filter;
    vector <Tensor::Tensor> weight;
    vector <Tensor::Tensor> filter_bias;
    vector <Tensor::Tensor> weight_bias;

    virtual Tensor::Tensor forward(Tensor::Tensor input) = 0;

    void readData(string name);
    void printModel();
    void printModelWeights();
};

void ConvNet::readTensorSet(vector <Tensor::Tensor>& tensor_store, string file_name)
{
    string global_path = "./src/data/Models/";

    tensor_store.clear();

    string line;
    ifstream file (global_path + file_name);
    if (!file.is_open()) cout << "Unable to open file\n";

    while (getline(file,line))
    {
        cout << "\r" << tensor_store.size() << flush;

        vector <int> size;
        istringstream ss_size(line);
        string word; 
        while (ss_size >> word) size.push_back(stoi(word));

        getline(file, line);
        vector <float> values;
        istringstream ss_values(line);
        while (ss_values >> word) values.push_back(stof(word));

        tensor_store.push_back(Tensor::Tensor (size, "", -1));
        tensor_store[tensor_store.size() - 1].setValue(values);

        getline(file, line);
    }
    
    cout << "\r" << flush;
    file.close();
}

void ConvNet::readData(string name)
{
    readTensorSet(filter, name + "/filters.txt");
    cout << "Read all filters" << endl;
    readTensorSet(weight, name + "/weights.txt");
    cout << "Read all weights" << endl;
    readTensorSet(filter_bias, name + "/filters_bias.txt");
    cout << "Read all filter biases" << endl;
    readTensorSet(weight_bias, name + "/weights_bias.txt");
    cout << "Read all weight biases" << endl;
}

void ConvNet::printModel()
{
    cout << "~~~~~~~~~~~~~~~~~~~~~~ Model Layers ~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

    cout << "//////////////////////// Filters ////////////////////////////" << endl;
    for(int i = 0; i < filter.size(); i++) filter[i].print_dim();
    cout << "/////////////////////////////////////////////////////////////" << endl;
    cout << "//////////////////////// Weights ////////////////////////////" << endl;
    for(int i = 0; i < weight.size(); i++) weight[i].print_dim();
    cout << "/////////////////////////////////////////////////////////////" << endl;
    cout << "///////////////////// Filter Biases /////////////////////////" << endl;
    for(int i = 0; i < filter_bias.size(); i++) filter_bias[i].print_dim();
    cout << "/////////////////////////////////////////////////////////////" << endl;
    cout << "///////////////////// Weight Biases /////////////////////////" << endl;
    for(int i = 0; i < weight_bias.size(); i++) weight_bias[i].print_dim();
    cout << "/////////////////////////////////////////////////////////////" << endl;

    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
}

void ConvNet::printModelWeights()
{
    cout << "~~~~~~~~~~~~~~~~~~~~~~ Model Layers ~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

    cout << "//////////////////////// Filters ////////////////////////////" << endl;
    for(int i = 0; i < filter.size(); i++) filter[i].print();
    cout << "/////////////////////////////////////////////////////////////" << endl;
    cout << "//////////////////////// Weights ////////////////////////////" << endl;
    for(int i = 0; i < weight.size(); i++) weight[i].print();
    cout << "/////////////////////////////////////////////////////////////" << endl;
    cout << "///////////////////// Filter Biases /////////////////////////" << endl;
    for(int i = 0; i < filter_bias.size(); i++) filter_bias[i].print();
    cout << "/////////////////////////////////////////////////////////////" << endl;
    cout << "///////////////////// Weight Biases /////////////////////////" << endl;
    for(int i = 0; i < weight_bias.size(); i++) weight_bias[i].print();
    cout << "/////////////////////////////////////////////////////////////" << endl;

    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
}

//////////////////////////////////////////////////////////////////////////////

class AlexNet : public ConvNet
{
public:
    AlexNet(bool setValues);
    Tensor::Tensor forward(Tensor::Tensor input);
};

AlexNet::AlexNet(bool setValues)
{
    if (!setValues) return;

    filter.push_back(Tensor::Tensor (vector <int> {64, 3, 11, 11}, "inc", -1));
    filter.push_back(Tensor::Tensor (vector <int> {192, 64, 5, 5}, "inc", -1));
    filter.push_back(Tensor::Tensor (vector <int> {384, 192, 3, 3}, "inc", -1));
    filter.push_back(Tensor::Tensor (vector <int> {256, 384, 3, 3}, "inc", -1));
    filter.push_back(Tensor::Tensor (vector <int> {256, 256, 3, 3}, "inc", -1));

    weight.push_back(Tensor::Tensor (vector <int> {4096, 9216}, "inc", -1));
    weight.push_back(Tensor::Tensor (vector <int> {4096, 4096}, "inc", -1));
    weight.push_back(Tensor::Tensor (vector <int> {1000, 4096}, "inc", -1));

    filter_bias.push_back(Tensor::Tensor (vector <int> {64}, "inc", -1));
    filter_bias.push_back(Tensor::Tensor (vector <int> {192}, "inc", -1));
    filter_bias.push_back(Tensor::Tensor (vector <int> {384}, "inc", -1));
    filter_bias.push_back(Tensor::Tensor (vector <int> {256}, "inc", -1));
    filter_bias.push_back(Tensor::Tensor (vector <int> {256}, "inc", -1));
    
    weight_bias.push_back(Tensor::Tensor (vector <int> {4096}, "inc", -1));
    weight_bias.push_back(Tensor::Tensor (vector <int> {4096}, "inc", -1));
    weight_bias.push_back(Tensor::Tensor (vector <int> {1000}, "inc", -1));
}

Tensor::Tensor AlexNet::forward(Tensor::Tensor input)
{
    assert(input.dim.size() == 3);
    assert(input.dim[0] == 3);
    assert(input.dim[1] == 224);
    assert(input.dim[2] == 224);

    input = Tensor::conv(input, filter[0], filter_bias[0], make_pair(4, 4));
    input = Tensor::pad(input, make_pair(2, 2), 0);
    input = Tensor::relu(input);
    input = Tensor::maxPool(input, make_pair(3,3), make_pair(2,2));

    input = Tensor::conv(input, filter[1], filter_bias[1], make_pair(1, 1));
    input = Tensor::pad(input, make_pair(2, 2), 0);
    input = Tensor::relu(input);
    input = Tensor::maxPool(input, make_pair(3,3), make_pair(2,2));

    input = Tensor::conv(input, filter[2], filter_bias[2], make_pair(1, 1));
    input = Tensor::pad(input, make_pair(1, 1), 0);
    input = Tensor::relu(input);
    
    input = Tensor::conv(input, filter[3], filter_bias[3], make_pair(1, 1));
    input = Tensor::pad(input, make_pair(1, 1), 0);
    input = Tensor::relu(input);
    
    input = Tensor::conv(input, filter[4], filter_bias[4], make_pair(1, 1));
    input = Tensor::pad(input, make_pair(1, 1), 0);
    input = Tensor::relu(input);
    input = Tensor::maxPool(input, make_pair(3,3), make_pair(2,2));

    input.reshape(vector <int> {9216});
    
    input = Tensor::fc(input, weight[0]); 
    input = Tensor::add(input, weight_bias[0]);
    input = Tensor::relu(input);

    input = Tensor::fc(input, weight[1]); 
    input = Tensor::add(input, weight_bias[1]);
    input = Tensor::relu(input);

    input = Tensor::fc(input, weight[2]); 
    input = Tensor::add(input, weight_bias[2]);

    return input;
}

//////////////////////////////////////////////////////////////////////////////

#endif
