#pragma once

#ifndef INCLUDE_H
#define INCLUDE_H

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120

#pragma OPENCL EXTENSION cl_amd_offline_devices : enable
#pragma OPENCL EXTENSION all : enable
// #define CL_CONTEXT_OFFLINE_DEVICES_AMD 0x403f

#include <CL/cl.hpp>

#include "OpenCL.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <math.h>
#include <bits/stdc++.h>

using namespace std;

#endif // INCLUDE_H
