cuda_template_1_3 = lambda fwd_expr, bwd_expr_r1, bwd_expr_c1, bwd_expr_c2, bwd_expr_c3: f"""

#include <torch/torch.h>
#include <cstdio>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)
#define INDEX3D(a, b, c, db, dc) (((a) * (db) * (dc) + (b) * (dc) + (c)))

#define ALL_THREADS_IN_WARP 0xFFFFFFFF

__global__ void fwd_kernel(const torch::PackedTensorAccessor64<float, 2> a_r1,
                           const torch::PackedTensorAccessor64<float, 2> a_c1,
                           const torch::PackedTensorAccessor64<float, 2> a_c2,
                           const torch::PackedTensorAccessor64<float, 2> a_c3,
                           torch::PackedTensorAccessor64<float, 2> a_result,
                           int M, int N, int K, int numThreads)
{{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // x.size(): (batch_size, in_feats)

    if (idx < numThreads)
    {{
        int im = idx / N;
        int in = idx % N;

        float res = 0;
        for (int ik = 0; ik < K; ik++)
        {{
            float r1 = a_r1[im][ik];
            float c1 = a_c1[in][ik];
            float c2 = a_c2[in][ik];
            float c3 = a_c3[in][ik];

            res += {fwd_expr};
        }}
        a_result[im][in] = res;
    }}
}}

__global__ void bwd_row_kernel(const torch::PackedTensorAccessor64<float, 2> a_gout,
                               const torch::PackedTensorAccessor64<float, 2> a_r1,
                               const torch::PackedTensorAccessor64<float, 2> a_c1,
                               const torch::PackedTensorAccessor64<float, 2> a_c2,
                               const torch::PackedTensorAccessor64<float, 2> a_c3,
                               torch::PackedTensorAccessor64<float, 2> a_grad_r1,
                               int M, int N, int K, int numThreads)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numThreads)
    {{
        int im = idx / K;
        int ik = idx % K;

        float r1 = a_r1[im][ik];

        float grad_r1 = 0.0f;
        for (int in = 0; in < N; in++)
        {{
            float c1 = a_c1[in][ik];
            float c2 = a_c2[in][ik];
            float c3 = a_c3[in][ik];

            float gout = a_gout[im][in];

            grad_r1 += ({bwd_expr_r1}) * gout;
        }}
        a_grad_r1[im][ik] = grad_r1;
    }}
}}

__global__ void bwd_col_kernel(const torch::PackedTensorAccessor64<float, 2> a_gout,
                               const torch::PackedTensorAccessor64<float, 2> a_r1,
                               const torch::PackedTensorAccessor64<float, 2> a_c1,
                               const torch::PackedTensorAccessor64<float, 2> a_c2,
                               const torch::PackedTensorAccessor64<float, 2> a_c3,
                               torch::PackedTensorAccessor64<float, 2> a_grad_c1,
                               torch::PackedTensorAccessor64<float, 2> a_grad_c2,
                               torch::PackedTensorAccessor64<float, 2> a_grad_c3,
                               int M, int N, int K, int numThreads)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numThreads)
    {{
        int in = idx / K;
        int ik = idx % K;

        float c1 = a_c1[in][ik];
        float c2 = a_c2[in][ik];
        float c3 = a_c3[in][ik];

        float grad_c1 = 0.0f;
        float grad_c2 = 0.0f;
        float grad_c3 = 0.0f;
        for (int im = 0; im < M; im++)
        {{
            float r1 = a_r1[im][ik];

            float gout = a_gout[im][in];

            grad_c1 += ({bwd_expr_c1}) * gout;
            grad_c2 += ({bwd_expr_c2}) * gout;
            grad_c3 += ({bwd_expr_c3}) * gout;
        }}
        a_grad_c1[in][ik] = grad_c1;
        a_grad_c2[in][ik] = grad_c2;
        a_grad_c3[in][ik] = grad_c3;
    }}
}}

void fwd_launcher(const torch::PackedTensorAccessor64<float, 2> r1,
                  const torch::PackedTensorAccessor64<float, 2> c1,
                  const torch::PackedTensorAccessor64<float, 2> c2,
                  const torch::PackedTensorAccessor64<float, 2> c3,
                  torch::PackedTensorAccessor64<float, 2> result,
                  int M, int N, int K)
{{
    int numThreads = M * N;
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    fwd_kernel<<<blockSize, threadSize>>>(r1, c1, c2, c3, result, M, N, K, numThreads);
}}

void bwd_launcher(const torch::PackedTensorAccessor64<float, 2> gout,
                  const torch::PackedTensorAccessor64<float, 2> r1,
                  const torch::PackedTensorAccessor64<float, 2> c1,
                  const torch::PackedTensorAccessor64<float, 2> c2,
                  const torch::PackedTensorAccessor64<float, 2> c3,
                  torch::PackedTensorAccessor64<float, 2> grad_r1,
                  torch::PackedTensorAccessor64<float, 2> grad_c1,
                  torch::PackedTensorAccessor64<float, 2> grad_c2,
                  torch::PackedTensorAccessor64<float, 2> grad_c3,
                  int M, int N, int K)
{{

    int numThreads_row = M * K;
    int numThreads_col = N * K;
    dim3 blockSize_row(DIVUP(numThreads_row, THREADS_PER_BLOCK));
    dim3 blockSize_col(DIVUP(numThreads_col, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    bwd_row_kernel<<<blockSize_row, threadSize>>>(gout, r1, c1, c2, c3,
                                                  grad_r1, M, N, K, numThreads_row);
    bwd_col_kernel<<<blockSize_col, threadSize>>>(gout, r1, c1, c2, c3,
                                                  grad_c1, grad_c2, grad_c3, M, N, K, numThreads_col);
}}

"""