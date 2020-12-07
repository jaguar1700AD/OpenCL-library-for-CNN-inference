#ifndef _ConvNets_HEADER_
#define _ConvNets_HEADER_

#include "include.h"
#include "Tensor.h"

//////////////////////////////////////////////////////////////////////////////

class AlexNet
{
public:
    vector <Tensor::Tensor> filter;
    vector <Tensor::Tensor> weight;
    vector <Tensor::Tensor> filter_bias;
    vector <Tensor::Tensor> weight_bias;

    AlexNet();
    Tensor::Tensor forward(Tensor::Tensor input);
};

AlexNet::AlexNet()
{
    filter.push_back(Tensor::Tensor (vector <int> {64, 3, 11, 11}, "inc", -1));
    filter.push_back(Tensor::Tensor (vector <int> {192, 64, 5, 5}, "inc", -1));
    filter.push_back(Tensor::Tensor (vector <int> {384, 192, 3, 3}, "inc", -1));
    filter.push_back(Tensor::Tensor (vector <int> {256, 384, 3, 3}, "inc", -1));
    filter.push_back(Tensor::Tensor (vector <int> {256, 256, 3, 3}, "inc", -1));

    weight.push_back(Tensor::Tensor (vector <int> {256*6*6, 4096}, "inc", -1));
    weight.push_back(Tensor::Tensor (vector <int> {4096, 4096}, "inc", -1));
    weight.push_back(Tensor::Tensor (vector <int> {4096, 1000}, "inc", -1));

    filter_bias.push_back(Tensor::Tensor (vector <int> {64}, "inc", -1));
    filter_bias.push_back(Tensor::Tensor (vector <int> {192}, "inc", -1));
    filter_bias.push_back(Tensor::Tensor (vector <int> {384}, "inc", -1));
    filter_bias.push_back(Tensor::Tensor (vector <int> {256}, "inc", -1));
    filter_bias.push_back(Tensor::Tensor (vector <int> {256}, "inc", -1));
    
    weight_bias.push_back(Tensor::Tensor (vector <int> {1, 4096}, "inc", -1));
    weight_bias.push_back(Tensor::Tensor (vector <int> {1, 4096}, "inc", -1));
    weight_bias.push_back(Tensor::Tensor (vector <int> {1, 1000}, "inc", -1));
}

Tensor::Tensor AlexNet::forward(Tensor::Tensor input)
{
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

    input.reshape(vector <int> {1, 256*6*6});
    
    input = Tensor::matMult(input, weight[0]); 
    input = Tensor::add(input, weight_bias[0]);
    input = Tensor::relu(input);

    input = Tensor::matMult(input, weight[1]); 
    input = Tensor::add(input, weight_bias[1]);
    input = Tensor::relu(input);

    input = Tensor::matMult(input, weight[2]); 
    input = Tensor::add(input, weight_bias[2]);

    return input;
}

//////////////////////////////////////////////////////////////////////////////

#endif
