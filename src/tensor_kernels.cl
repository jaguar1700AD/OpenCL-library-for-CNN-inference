kernel void tensor_add(global float *T1, global float* T2, global float* dest)
{
    const int id = get_global_id(0);
    dest[id] = T1[id] + T2[id];    
}