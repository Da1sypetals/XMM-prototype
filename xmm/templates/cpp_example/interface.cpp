#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <cuda_runtime.h>
#include <utility>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void fwd_launcher(const torch::PackedTensorAccessor64<float, 2> r1,
                  const torch::PackedTensorAccessor64<float, 2> c1,
                  const torch::PackedTensorAccessor64<float, 2> c2,
                  const torch::PackedTensorAccessor64<float, 2> c3,
                  torch::PackedTensorAccessor64<float, 2> result,
                  int M, int N, int K);

void bwd_launcher(const torch::PackedTensorAccessor64<float, 2> gout,
                  const torch::PackedTensorAccessor64<float, 2> r1,
                  const torch::PackedTensorAccessor64<float, 2> c1,
                  const torch::PackedTensorAccessor64<float, 2> c2,
                  const torch::PackedTensorAccessor64<float, 2> c3,
                  torch::PackedTensorAccessor64<float, 2> grad_r1,
                  torch::PackedTensorAccessor64<float, 2> grad_c1,
                  torch::PackedTensorAccessor64<float, 2> grad_c2,
                  torch::PackedTensorAccessor64<float, 2> grad_c3,
                  int M, int N, int K);

using Tensor4 = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

torch::Tensor forward_1_3(torch::Tensor r1, torch::Tensor c1, torch::Tensor c2, torch::Tensor c3)
{

    // first check inputs!
    CHECK_INPUT(r1);
    CHECK_INPUT(c1);
    CHECK_INPUT(c2);
    CHECK_INPUT(c3);

    // get metadata
    int M = r1.size(0);
    int K = r1.size(1);
    int N = c1.size(0);

    // get data accesser
    const auto r1_acc = r1.packed_accessor64<float, 2>();
    const auto c1_acc = c1.packed_accessor64<float, 2>();
    const auto c2_acc = c2.packed_accessor64<float, 2>();
    const auto c3_acc = c3.packed_accessor64<float, 2>();

    // create result tensor
    torch::Tensor result = torch::empty({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat));

    auto result_acc = result.packed_accessor64<float, 2>();

    fwd_launcher(r1_acc, c1_acc, c2_acc, c3_acc, result_acc, M, N, K);

    // cudaDeviceSynchronize();

    return result.sum({2});
}

Tensor4 backward_1_3(torch::Tensor gout, torch::Tensor r1, torch::Tensor c1, torch::Tensor c2, torch::Tensor c3)
{

    // first check inputs!
    CHECK_INPUT(r1);
    CHECK_INPUT(c1);
    CHECK_INPUT(c2);
    CHECK_INPUT(c3);

    // get metadata
    int M = r1.size(0);
    int K = r1.size(1);
    int N = c1.size(0);

    // get data accesser
    const auto gout_acc = gout.packed_accessor64<float, 2>();

    const auto r1_acc = r1.packed_accessor64<float, 2>();
    const auto c1_acc = c1.packed_accessor64<float, 2>();
    const auto c2_acc = c2.packed_accessor64<float, 2>();
    const auto c3_acc = c3.packed_accessor64<float, 2>();

    std::vector<int64_t> shape_row = {M, K};
    std::vector<int64_t> shape_col = {N, K};

    torch::Tensor grad_r1 = torch::empty(shape_row, torch::device(torch::kCUDA).dtype(torch::kFloat));
    torch::Tensor grad_c1 = torch::empty(shape_col, torch::device(torch::kCUDA).dtype(torch::kFloat));
    torch::Tensor grad_c2 = torch::empty(shape_col, torch::device(torch::kCUDA).dtype(torch::kFloat));
    torch::Tensor grad_c3 = torch::empty(shape_col, torch::device(torch::kCUDA).dtype(torch::kFloat));

    auto grad_r1_acc = grad_r1.packed_accessor64<float, 2>();
    auto grad_c1_acc = grad_c1.packed_accessor64<float, 2>();
    auto grad_c2_acc = grad_c2.packed_accessor64<float, 2>();
    auto grad_c3_acc = grad_c3.packed_accessor64<float, 2>();

    bwd_launcher(gout_acc, r1_acc, c1_acc, c2_acc, c3_acc, grad_r1_acc, grad_c1_acc, grad_c2_acc, grad_c3_acc, M, N, K);

    // cudaDeviceSynchronize();

    return Tensor4(grad_r1,
                   grad_c1,
                   grad_c2,
                   grad_c3);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &forward_1_3, "forward R=1,C=3");
    m.def("backward", &backward_1_3, "backward R=1,C=3");
}