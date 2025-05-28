def operand_params(nrow, ncol):
    res = ""
    for r in range(1, nrow + 1):
        res += f"const torch::PackedTensorAccessor64<float, 2> r{r},\n"
    for c in range(1, ncol + 1):
        res += f"const torch::PackedTensorAccessor64<float, 2> c{c},\n"
    return res


def grad_params(nrow, ncol):
    res = ""
    for r in range(1, nrow + 1):
        res += f"torch::PackedTensorAccessor64<float, 2> grad_r{r},\n"
    for c in range(1, ncol + 1):
        res += f"torch::PackedTensorAccessor64<float, 2> grad_c{c},\n"
    return res


def def_forward(nrow, ncol):
    res = ""
    for r in range(1, nrow + 1):
        res += f"const torch::PackedTensorAccessor64<float, 2> a_r{r},\n"
    for c in range(1, ncol + 1):
        res += f"const torch::PackedTensorAccessor64<float, 2> a_c{c},\n"
    return res


def fetch_operands(nrow, ncol):
    res = ""
    for r in range(1, nrow + 1):
        res += f"float r{r} = a_r{r}[im][ik];\n"
    for c in range(1, ncol + 1):
        res += f"float c{c} = a_c{c}[in][ik];\n"
    return res


def fetch_row_operands(nrow):
    res = ""
    for r in range(1, nrow + 1):
        res += f"float r{r} = a_r{r}[im][ik];\n"
    return res


def fetch_col_operands(ncol):
    res = ""
    for c in range(1, ncol + 1):
        res += f"float c{c} = a_c{c}[in][ik];\n"
    return res


def backward_row(nrow):
    res = ""
    for r in range(1, nrow + 1):
        res += f"torch::PackedTensorAccessor64<float, 2> a_grad_r{r},\n"
    return res


def backward_col(ncol):
    res = ""
    for c in range(1, ncol + 1):
        res += f"torch::PackedTensorAccessor64<float, 2> a_grad_c{c},\n"
    return res


def diff_row(nrow, diffs):
    res = ""
    for r in range(1, nrow + 1):
        k = f"r{r}"
        res += f"grad_r{r} += ({diffs[k]}) * gout;\n"
    return res


def diff_col(nrow, diffs):
    res = ""
    for c in range(1, nrow + 1):
        k = f"c{c}"
        res += f"grad_c{c} += ({diffs[k]}) * gout;\n"
    return res


def def_result_row(nrow):
    res = ""
    for r in range(1, nrow + 1):
        res += f"float grad_r{r} = 0.0f;\n"
    return res


def def_result_col(ncol):
    res = ""
    for c in range(1, ncol + 1):
        res += f"float grad_c{c} = 0.0f;\n"
    return res


def store_results_row(nrow):
    res = ""
    for r in range(1, nrow + 1):
        res += f"a_grad_r{r}[im][ik] = grad_r{r};\n"
    return res


def store_results_col(ncol):
    res = ""
    for c in range(1, ncol + 1):
        res += f"a_grad_c{c}[in][ik] = grad_c{c};\n"
    return res


def call_forward(nrow, ncol):
    res = []
    for r in range(1, nrow + 1):
        res.append(f"r{r}")
    for c in range(1, ncol + 1):
        res.append(f"c{c}")
    return ", ".join(res)


def call_backward_row(nrow):
    res = []
    for r in range(1, nrow + 1):
        res.append(f"grad_r{r}")
    return ", ".join(res)


def call_backward_col(ncol):
    res = []
    for c in range(1, ncol + 1):
        res.append(f"grad_c{c}")
    return ", ".join(res)


def generate_cuda(nrow, ncol, fwd_expr, bwd_expr_dict):
    # bwd expr is a dictionary
    return f"""

#include <torch/torch.h>
#include <cstdio>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)
#define INDEX3D(a, b, c, db, dc) (((a) * (db) * (dc) + (b) * (dc) + (c)))


__global__ void fwd_kernel(
                           {def_forward(nrow, ncol)}
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
            {fetch_operands(nrow, ncol)}

            res += {fwd_expr};
        }}
        a_result[im][in] = res;
    }}
}}

__global__ void bwd_row_kernel(const torch::PackedTensorAccessor64<float, 2> a_gout,
                               {def_forward(nrow, ncol)}
                               {backward_row(nrow)}
                               int M, int N, int K, int numThreads)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numThreads)
    {{
        int im = idx / K;
        int ik = idx % K;

        {fetch_row_operands(nrow)}

        {def_result_row(nrow)}
        for (int in = 0; in < N; in++)
        {{
            {fetch_col_operands(ncol)}

            float gout = a_gout[im][in];

            {diff_row(nrow, bwd_expr_dict)}
        }}
        {store_results_row(nrow)}
    }}
}}

__global__ void bwd_col_kernel(const torch::PackedTensorAccessor64<float, 2> a_gout,
                               {def_forward(nrow, ncol)}
                               {backward_col(ncol)}
                               int M, int N, int K, int numThreads)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numThreads)
    {{
        int in = idx / K;
        int ik = idx % K;

        {fetch_col_operands(ncol)}

        {def_result_col(ncol)}
        for (int im = 0; im < M; im++)
        {{
            {fetch_row_operands(nrow)}

            float gout = a_gout[im][in];

            {diff_col(ncol, bwd_expr_dict)}
        }}
        {store_results_col(ncol)}

    }}
}}

void fwd_launcher(
                  {operand_params(nrow, ncol)}
                  torch::PackedTensorAccessor64<float, 2> result,
                  int M, int N, int K)
{{
    int numThreads = M * N;
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    fwd_kernel<<<blockSize, threadSize>>>({call_forward(nrow, ncol)}, result, M, N, K, numThreads);
}}

void bwd_launcher(const torch::PackedTensorAccessor64<float, 2> gout,
                  {operand_params(nrow, ncol)}
                  {grad_params(nrow, ncol)}
                  int M, int N, int K)
{{

    int numThreads_row = M * K;
    int numThreads_col = N * K;
    dim3 blockSize_row(DIVUP(numThreads_row, THREADS_PER_BLOCK));
    dim3 blockSize_col(DIVUP(numThreads_col, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    bwd_row_kernel<<<blockSize_row, threadSize>>>(gout, {call_forward(nrow, ncol)},
                                                  {call_backward_row(nrow)}, M, N, K, numThreads_row);
    bwd_col_kernel<<<blockSize_col, threadSize>>>(gout, {call_forward(nrow, ncol)},
                                                  {call_backward_col(ncol)}, M, N, K, numThreads_col);
}}

"""
