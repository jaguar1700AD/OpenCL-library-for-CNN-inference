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

kernel void tensor_conv(global float *image, global float* filters, global float* bias, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int strider, stridec)
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

kernel void tensor_relu(global float* image, global float* dest)
{
    const int id = get_global_id(0);
    if (image[id] > 0) dest[id] = image[id];
    else dest[id] = 0;
}

kernel void tensor_maxPool(global float *image, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int strider, stridec)
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

 kernel void tensor_avgPool(global float *image, global float* out, int ir, int ic, int iz, int kr, int kc, int or, int oc, int strider, stridec)
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