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