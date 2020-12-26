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