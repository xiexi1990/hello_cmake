#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

const int WARPSIZE = 32;

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

vector<vector<int>> build_part(int partSize, vector<int> indptr)
{
    int num_nodes = indptr.size() - 1;
    int degree, thisNumParts, numParts = 0;

    for (int i = 0; i < num_nodes; i++)
    {
        degree = indptr[i + 1] - indptr[i];
        if (degree % partSize == 0)
            thisNumParts = degree / partSize;
        else
            thisNumParts = degree / partSize + 1;
        numParts += thisNumParts;
    }

    auto partPtr = vector<int>(numParts + 1, 0);
    auto part2Node = vector<int>(numParts, 0);

    int part_counter = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        int degree = indptr[i + 1] - indptr[i];
        if (degree % partSize == 0)
            thisNumParts = degree / partSize;
        else
            thisNumParts = degree / partSize + 1;

        for (int pid = 0; pid < thisNumParts; pid++)
        {
            int partBeg = indptr[i] + pid * partSize;
            int partEnd = partBeg + partSize < indptr[i + 1] ? partBeg + partSize : indptr[i + 1];
            partPtr[part_counter] = partBeg;
            part2Node[part_counter++] = i;
            if (i == num_nodes - 1 && partEnd == indptr[i + 1])
                partPtr[part_counter] = partEnd;
        }
    }
    return {partPtr, part2Node};
}

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_e, int feat_in)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int warpid = tid / WARPSIZE;
    if (warpid >= num_e)
        return;
    int lane_id = tid & (WARPSIZE - 1);

    float left = __ldg(val + warpid);
    int row = __ldg(ptr + warpid), col = __ldg(idx + warpid);

    float right = __ldg(vin + col * feat_in + lane_id);

    float result = left * right;

    int right_loc = row * feat_in + lane_id;

    atomicAdd(vout + right_loc, result);
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
    // int v_num = cuda_read_array(&indptr, "/home/xiexi/py_projects/py1/indptr.bin") - 1;
    int v_num = 40;
    cuda_read_array(&indptr, "/home/xiexi/py_projects/py1/indptr.bin");
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
    grid.x = (e_num * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block.x = BLOCK_SIZE;

    cout << e_num << " " << v_num << endl;

    vector<int> indptr_vec(indptr, indptr + v_num + 1);

    auto parts = build_part(13, indptr_vec);
    vector<int> partPtr = parts[0];
    vector<int> part2Node = parts[1];

    return 0;

    spmm_kernel_opt<<<grid, block>>>(indptr, indices, data, vin, vout, e_num, dim);
    cudaDeviceSynchronize();

    float err = 0;
    for (int i = 0; i < v_num * dim; i++)
    {
        //    cout << vout[i] << " ";
        err += abs(ref_vout[i] - vout[i]);
    }
    // cout << endl;
    // for (int i = 0; i < v_num * dim; i++)
    // {
    //     cout << ref_vout[i] << " ";
    // }
    // cout << endl;
    cout << "err sum = " << err << endl;

    cudaFree(indptr);
    cudaFree(indices);
    cudaFree(data);
    cudaFree(vin);
    cudaFree(vout);

    delete ref_vout;

    return 0;
}
