#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;
template <typename scalar_t>
int cuda_read_array(scalar_t **arr, string file)
{
    std::ifstream input(file, ios::in | ios::binary);
    input.seekg(0, input.end);
    int length = input.tellg();
    input.seekg(0, input.beg);
    int cnt = length / sizeof(scalar_t);
    // *arr = new scalar_t[cnt];
    cudaMallocManaged(arr, cnt * sizeof(scalar_t));

    input.read((char *)*arr, length);
    input.close();
    // *arr = reinterpret_cast<float *>(&buffer[0]);
    return cnt;
}
template <typename scalar_t>
int read_array(scalar_t **arr, string file)
{
    std::ifstream input(file, ios::in | ios::binary);
    input.seekg(0, input.end);
    int length = input.tellg();
    input.seekg(0, input.beg);
    int cnt = length / sizeof(scalar_t);
    *arr = new scalar_t[cnt];
    input.read((char *)*arr, length);
    input.close();
    return cnt;
}

__global__ void spmm_kernel_ref(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int begin = ptr[tid], end = ptr[tid + 1];
    for (int j = 0; j < INFEATURE; ++j)
    {
        float result = 0.0f;
        for (int i = begin; i < end; ++i)
        {
            result += vin[idx[i] * INFEATURE + j] * val[i];
        }
        vout[tid * INFEATURE + j] = result;
    }
}

__global__ void spmm_kernel_sp(int *ptr, int *idx, float *val, float *vin, int *vin_loc, float *vout, int num_v, int INFEATURE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int begin = ptr[tid], end = ptr[tid + 1];
    for (int j = 0; j < INFEATURE; ++j)
    {
        float result = 0.0f;
        for (int i = begin; i < end; ++i)
        {
            result += vin[idx[i] * INFEATURE + j] * val[i];
        }
        vout[tid * INFEATURE + j] = result;
    }
}


int main(int argc, char *argv[])
{
    float *data;
    int e_num = cuda_read_array(&data, "/home/xiexi/py_projects/py1/data.bin");
    int *indptr, *indices;
    int v_num = cuda_read_array(&indptr, "/home/xiexi/py_projects/py1/indptr.bin") - 1;
    cuda_read_array(&indices, "/home/xiexi/py_projects/py1/indices.bin");
    int dim = 32;
    float *vin, *vout, *ref_vout;
    read_array(&ref_vout, "/home/xiexi/py_projects/py1/vout.bin");
    cudaMallocManaged(&vin, v_num * dim * sizeof(float));
    cudaMallocManaged(&vout, v_num * dim * sizeof(float));
    for (int i = 0; i < v_num * dim; i++)
    {
        vin[i] = i;
        vout[i] = 0;
    }

    int BLOCK_SIZE = 128;
    dim3 grid, block;
    grid.x = (v_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block.x = BLOCK_SIZE;

    cout << e_num << " " << v_num << endl;

    spmm_kernel_ref<<<grid, block>>>(indptr, indices, data, vin, vout, v_num, dim);
    cudaDeviceSynchronize();

    float err = 0;
    for (int i = 0; i < v_num * dim; i++)
    {
        err += abs(ref_vout[i] - vout[i]);
    }
    cout << "err sum = " << err << endl;

    cudaFree(indptr);
    cudaFree(indices);
    cudaFree(data);
    cudaFree(vin);
    cudaFree(vout);

    delete ref_vout;

    return 0;
}
