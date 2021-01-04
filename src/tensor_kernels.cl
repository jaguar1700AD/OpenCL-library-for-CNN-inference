int myMin(int a, int b, int c)
{
    return min(a, min(b, c));
}

kernel void tensor_sub(global float *T1, global float* T2, global float* dest)
{
    const int id = get_global_id(0);
    dest[id] = T1[id] - T2[id];    
}

kernel void tensor_add(global float *T1, global float* T2, global float* dest)
{
    const int id = get_global_id(0);
    dest[id] = T1[id] + T2[id];    
}


kernel void tensor_mult(global float *T1, global float* T2, global float* dest)
{
    const int id = get_global_id(0);
    dest[id] = T1[id] * T2[id];    
}

kernel void tensor_conv(global float *image, global float* filters, global float* bias, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int strider, int stridec)
{
    // kr, kc -> Num of rows and columns in the kernel
    // ir, ic -> Num of rows and columns in the image
    // iz -> Num of planes in the image = Num of planes in the kernel
    // or, oc -> Num of rows and columns in the output
    
    const int filtId = get_global_id(0); // filter id
    const int rId = get_global_id(1); // row id
    const int cId = get_global_id(2); // column id

    float sum = 0;
    for (int z = 0; z < iz; z++)
    {
        for (int r = 0; r < kr; r++)
        {
            for (int c = 0; c < kc; c++)
            {
                int r_image = r + strider * rId;
                int c_image = c + stridec * cId;
                sum += filters[filtId*iz*kr*kc + z*kr*kc + r*kc + c] * image[z*ir*ic + r_image*ic + c_image];  
            }
        }
    }
    sum += bias[filtId];

    out[filtId*or*oc + rId*oc + cId] = sum;
 }

 kernel void tensor_conv_optim1(global float *image, int depth_per_iter, global float* filters, local float* image_local, local float* filter_local, global float* bias, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int oz, int strider, int stridec)
{
    // Assume -> strider <= kr, stridec <= kc
    // Consecutive filter and image positions processed by a single work group
    // Uncoalesced memory access

    // kr, kc -> Num of rows and columns in the kernel
    // ir, ic -> Num of rows and columns in the image
    // iz -> Num of planes in the image = Num of planes in the kernel
    // or, oc -> Num of rows and columns in the output
    // oz -> Number of filters = Depth of output

    const int filtId = get_global_id(0); // filter id 
    const int rId = get_global_id(1); // row id
    const int cId = get_global_id(2); // column id
    // get_local_id(0) = 1
    const int localrId = get_local_id(1);
    const int localcId = get_local_id(2);
    const int numRowThreads = get_local_size(1); // Num consecutive filters for a work group in row dirn
    const int numColThreads = get_local_size(2); // Num consecutive filters for a work group in col dirn
    const int workrId = get_group_id(1);
    const int workcId = get_group_id(2);

    for(int iz_beg = 0; iz_beg < iz; iz_beg += depth_per_iter)
    {
        int iz_end = min(iz_beg + depth_per_iter, iz);
        int iz_len = iz_end - iz_beg;

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Image data and filter data for neighbouring positions is stored in local memory

        // Copy image to local memory

        const int r_image_beg = workrId * numRowThreads * strider;
        const int c_image_beg = workcId * numColThreads * stridec;
        const int rlim = kr + strider * (numRowThreads - 1);
        const int clim = kc + stridec * (numColThreads - 1);
        const int rowCopySize =  rlim / numRowThreads + 1;
        const int colCopySize =  clim / numColThreads + 1;
        const int rbeg = localrId * rowCopySize;
        const int cbeg = localcId * colCopySize;

        for(int z = 0; z < iz_len; z++)
        {
            for(int r = rbeg; r < myMin(rbeg + rowCopySize, rlim, ir - r_image_beg); r++)
            {
                for(int c = cbeg; c < myMin(cbeg + colCopySize, clim, ic - c_image_beg); c++)
                {
                    int r_image = r_image_beg + r;
                    int c_image = c_image_beg + c;
                    int z_image = iz_beg + z;
                    image_local[z*rlim*clim + r * clim + c] = image[z_image*ir*ic + r_image*ic + c_image];
                }
            }
        }

        // Copy filter to local memory

        const int filtlim = iz_len * kr * kc;
        const int numThreads = numRowThreads * numColThreads;
        const int copySize = filtlim / numThreads + 1;
        const int beg = (localrId * numColThreads + localcId) * copySize;
        for(int id = beg; id < min(beg + copySize, filtlim); id++)
        {
            filter_local[id] = filters[filtId * iz * kr * kc + iz_beg * kr * kc + id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (rId < or && cId < oc)
        {
            float sum = 0;
            for (int z = 0; z < iz_len; z++)
            {
                for (int r = 0; r < kr; r++)
                {
                    for (int c = 0; c < kc; c++)
                    {
                        const int r_image = r + strider * localrId;
                        const int c_image = c + stridec * localcId;
                        sum += image_local[z*rlim*clim + r_image * clim + c_image] * filter_local[z*kr*kc + r * kc + c];
                    }
                }
            }
            if (iz_beg == 0) out[filtId*or*oc + rId*oc + cId] = sum;
            else out[filtId*or*oc + rId*oc + cId] += sum;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (rId < or && cId < oc) out[filtId*or*oc + rId*oc + cId] += bias[filtId];

 }

  kernel void tensor_conv_optim2(global float *image, int depth_per_iter, global float* filters, local float* image_local, local float* filter_local, global float* bias, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int oz, int strider, int stridec)
{
    // Assume -> strider <= kr, stridec <= kc
    // Consecutive filter and image positions processed by a single work group
    // Coalesced memory access

    // kr, kc -> Num of rows and columns in the kernel
    // ir, ic -> Num of rows and columns in the image
    // iz -> Num of planes in the image = Num of planes in the kernel
    // or, oc -> Num of rows and columns in the output
    // oz -> Number of filters = Depth of output

    const int filtId = get_global_id(0); // filter id 
    const int rId = get_global_id(1); // row id
    const int cId = get_global_id(2); // column id
    // get_local_id(0) = 1
    const int localrId = get_local_id(1);
    const int localcId = get_local_id(2);
    const int numRowThreads = get_local_size(1); // Num consecutive filters for a work group in row dirn
    const int numColThreads = get_local_size(2); // Num consecutive filters for a work group in col dirn
    const int workrId = get_group_id(1);
    const int workcId = get_group_id(2);

    for(int iz_beg = 0; iz_beg < iz; iz_beg += depth_per_iter)
    {
        int iz_end = min(iz_beg + depth_per_iter, iz);
        int iz_len = iz_end - iz_beg;

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Image data and filter data for neighbouring positions is stored in local memory

        // Copy image to local memory

        const int r_image_beg = workrId * numRowThreads * strider;
        const int c_image_beg = workcId * numColThreads * stridec;
        const int rlim = min(kr + strider * (numRowThreads - 1), ir - r_image_beg);
        const int clim = min(kc + stridec * (numColThreads - 1), ic - c_image_beg);

        const int imagelim = rlim * clim;
        int numThreads = numRowThreads * numColThreads;
        int beg = localrId * numColThreads + localcId;

        for(int z = 0; z < iz_len; z++)
        {
            for(int id = beg; id < imagelim; id += numThreads)
            {
                    int r_image = r_image_beg + id / clim;
                    int c_image = c_image_beg + id % clim;
                    int z_image = iz_beg + z;
                    image_local[z*rlim*clim + id] = image[z_image*ir*ic + r_image*ic + c_image];
            }
        }

        // Copy filter to local memory

        const int filtlim = iz_len * kr * kc;
        numThreads = numRowThreads * numColThreads;
        beg = localrId * numColThreads + localcId;
        for(int id = beg; id < filtlim; id += numThreads)
        {
            filter_local[id] = filters[filtId * iz * kr * kc + iz_beg * kr * kc + id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (rId < or && cId < oc)
        {
            float sum = 0;
            for (int z = 0; z < iz_len; z++)
            {
                for (int r = 0; r < kr; r++)
                {
                    for (int c = 0; c < kc; c++)
                    {
                        const int r_image = r + strider * localrId;
                        const int c_image = c + stridec * localcId;
                        sum += image_local[z*rlim*clim + r_image * clim + c_image] * filter_local[z*kr*kc + r * kc + c];
                    }
                }
            }
            if (iz_beg == 0) out[filtId*or*oc + rId*oc + cId] = sum;
            else out[filtId*or*oc + rId*oc + cId] += sum;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (rId < or && cId < oc) out[filtId*or*oc + rId*oc + cId] += bias[filtId];

 }

kernel void tensor_relu(global float* image, global float* dest)
{
    const int id = get_global_id(0);
    if (image[id] > 0) dest[id] = image[id];
    else dest[id] = 0;
}

kernel void tensor_maxPool(global float *image, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int strider, int stridec)
{
    // kr, kc -> Num of rows and columns in the kernel
    // ir, ic -> Num of rows and columns in the image
    // iz -> Num of planes in the image = Num of planes in the kernel
    // or, oc -> Num of rows and columns in the output
    
    const int planeId = get_global_id(0); // plane id
    const int rId = get_global_id(1); // row id
    const int cId = get_global_id(2); // column id

    float max_val = FLT_MIN;
    for (int r = 0; r < kr; r++)
    {
        for (int c = 0; c < kc; c++)
        {
            int r_image = r + strider * rId;
            int c_image = c + stridec * cId;
            max_val = max(max_val, image[planeId*ir*ic + r_image*ic + c_image]);  
        }
    }

    out[planeId*or*oc + rId*oc + cId] = max_val;
 }

 kernel void tensor_avgPool(global float *image, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int strider, int stridec)
{
    // kr, kc -> Num of rows and columns in the kernel
    // ir, ic -> Num of rows and columns in the image
    // iz -> Num of planes in the image = Num of planes in the kernel
    // or, oc -> Num of rows and columns in the output
    
    const int planeId = get_global_id(0); // plane id
    const int rId = get_global_id(1); // row id
    const int cId = get_global_id(2); // column id

    float sum = 0;
    for (int r = 0; r < kr; r++)
    {
        for (int c = 0; c < kc; c++)
        {
            int r_image = r + strider * rId;
            int c_image = c + stridec * cId;
            sum += image[planeId*ir*ic + r_image*ic + c_image];  
        }
    }
    sum = sum / (kr * kc);

    out[planeId*or*oc + rId*oc + cId] = sum;
 }

kernel void tensor_matMult(global float* image, global float* weights, global float* out, int size, int m, int n)
{
    // Image -> (m, size)
    // weights -> (size, n)
    // out -> (m, n)

    const int rid = get_global_id(0); // rid row selected in image
    const int cid = get_global_id(1); // cid column selected in weights

    float sum = 0;
    for(int i = 0; i < size; i++)
    {
        sum += image[rid * size + i] * weights[i * n + cid];
    }

    out[rid * n + cid] = sum;
}

kernel void tensor_fcMult1(global float* weight, global float* act, global float* out, local float* act_local, int m, int n, int p)
{
    // weight -> m X n matrix
    // act -> n X 1 matrix
    // So weight X act gives m X 1 matrix
    // out is m X p matrix (Stored in row major order with values to be added placed along a column)

    int numThreads = get_local_size(0);
    // get_local_size(1) = 1
    int colId = get_global_id(1);
    int rowId = get_global_id(0);
    int localId = get_local_id(0);
    // get_local_id(1) = 0
    int colOffset = colId * p;
    int numComp = min(p, n - colOffset); // Number of values to compute dot product of

    // Load act for work group into local memory

    for(int i = localId; i < numComp; i += numThreads)
    {
        act_local[i] = act[i + colOffset];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform dot product

    if (rowId < m)
    {
        float sum = 0;
        for(int i = 0; i < numComp; i++)
        {
            int j = (i + localId) % numComp; // Use j instead of i to avoid bank conflicts
            sum += weight[rowId * n + colOffset + j] * act_local[j];
        }

        out[colId * m + rowId] = sum;
    }
}

kernel void tensor_fcMult1(global float* weight, global float* act, global float* out, local float* act_local, int m, int n, int p)
{
    // weight -> m X n matrix
    // act -> n X 1 matrix
    // So weight X act gives m X 1 matrix
    // out is m X p matrix (Stored in row major order with values to be added placed along a column)

    int numThreads = get_local_size(0);
    // get_local_size(1) = 1
    int colId = get_global_id(1);
    int rowId = get_global_id(0);
    int localId = get_local_id(0);
    // get_local_id(1) = 0
    int colOffset = colId * p;
    int numComp = min(p, n - colOffset); // Number of values to compute dot product of

    // Load act for work group into local memory

    for(int i = localId; i < numComp; i += numThreads)
    {
        act_local[i] = act[i + colOffset];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform dot product

    if (rowId < m)
    {
        float sum = 0;
        for(int i = 0; i < numComp; i++)
        {
            int j = (i + localId) % numComp; // Use j instead of i to avoid bank conflicts
            sum += weight[rowId * n + colOffset + j] * act_local[j];
        }

        out[colId * m + rowId] = sum;
    }
}

kernel void tensor_fcReduce(global float* input, global float* output, int ir, int ic)
{
    // All values along each column are added
    int id = get_global_id(0);

    float sum = 0;
    for(int r = 0; r < ir; r++)
    {
        sum += input[r * ic + id];
    }
    output[id] = sum;
}

kernel void tensor_pad(global float* image, global float* out, int ir, int ic, int iz, int padr, int padc, float pad_val)
{
    // Image -> (iz, ir, ic)
    // out -> (iz, ir + 2*padr, ic + 2*padc)

    const int zId = get_global_id(0);
    const int rId = get_global_id(1);
    const int cId = get_global_id(2);

    float value = pad_val;
    int rimage = rId - padr;
    int cimage = cId - padc;
    int oz = iz;
    int or = ir + 2*padr;
    int oc = ic + 2*padc;

    if (rimage >= 0 && rimage < ir && cimage >= 0 && cimage < ic) 
    {
        value = image[zId * ir * ic + rimage * ic + cimage];
    }

    out[zId * or * oc + rId * oc + cId] = value;
}

kernel void tensor_begProcess(global uchar* image, global float* out, int ir, int ic, int iz, int or, int oc, global float* mean, global float* std)
{
    // Used for resizing and processing raw images provided by stbi library and bringing them to correct format
    // Input contains uint8_t data

    // Image -> (ir, ic, iz) -> Different from normal tensor format
    // Output -> (iz, or, oc) -> Normal tensor format

    int idr = get_global_id(0);
    int idc = get_global_id(1);

    // Positions in original image
    float rf = (idr / (float) or) * ir;
    float cf = (idc / (float) oc) * ic;

    int rlow = floor(rf); int rhigh = rlow + 1;
    int clow = floor(cf); int chigh = clow + 1;
    float deltar = rf - rlow;
    float deltac = cf - clow;

    for(int channel = 0; channel < iz; channel++)
    {
        float ans = 0;
        ans += image[rlow * ic * iz + clow * iz + channel] * (1-deltar) * (1 - deltac);
        ans += image[rhigh * ic * iz + clow * iz + channel] * (deltar) * (1 - deltac);
        ans += image[rlow * ic * iz + chigh * iz + channel] * (1-deltar) * (deltac);
        ans += image[rhigh * ic * iz + chigh * iz + channel] * (deltar) * (deltac);
        ans /= 255.0;
        ans -= mean[channel];
        ans /= std[channel];
        out[channel * or * oc + idr * oc + idc] = ans;
    }
}