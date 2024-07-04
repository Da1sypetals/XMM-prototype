cpp_code = """

#include <torch/extension.h>

#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <cuda_runtime.h>
#include <utility>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void gemm_caller_tn(int m, int n, int k,
                    float const *dA, int ldda,
                    float const *dB, int lddb,
                    float *dC, int lddc,
                    float alpha, float beta);

torch::Tensor gemm(torch::Tensor A, torch::Tensor B)
{
    CHECK_CUDA(A);
    CHECK_CUDA(B);

    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);

    int M = A.size(0);
    int N = B.size(0);
    int K = A.size(1);

    if (A.size(1) != B.size(1))
    {
        throw std::runtime_error("Sizes does not match!");
    }

    auto a = A.data_ptr<float>();
    auto b = B.data_ptr<float>();

    torch::Tensor C = torch::empty({{N, M}}, torch::device(torch::kCUDA).dtype(torch::kFloat));

    auto c = C.data_ptr<float>();

    gemm_caller_tn(M, N, K, a, K, b, K, c, M, 1.0f, 0.0f);

    return C.t();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &gemm, "forward");
}



"""