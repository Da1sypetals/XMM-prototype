def check_inputs(nrow, ncol):
    res = ""
    for r in range(1, nrow + 1):
        res += f"CHECK_INPUT(r{r});\n"
    for c in range(1, ncol + 1):
        res += f"CHECK_INPUT(c{c});\n"
    return res


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


def using_tensor_tuple(n):
    tensors = ", ".join(["torch::Tensor"] * n)
    return f"using Tensor{n} = std::tuple<{tensors}>"


def accessors(nrow, ncol):
    res = ""
    for r in range(1, nrow + 1):
        res += f"const auto r{r}_acc = r{r}.packed_accessor64<float, 2>();\n"
    for c in range(1, ncol + 1):
        res += f"const auto c{c}_acc = c{c}.packed_accessor64<float, 2>();\n"
    return res


def def_forward(nrow, ncol):
    res = []
    for r in range(1, nrow + 1):
        res.append(f"torch::Tensor r{r}")
    for c in range(1, ncol + 1):
        res.append(f"torch::Tensor c{c}")
    return ", ".join(res)


def call_forward(nrow, ncol):
    res = []
    for r in range(1, nrow + 1):
        res.append(f"r{r}_acc")
    for c in range(1, ncol + 1):
        res.append(f"c{c}_acc")
    return ", ".join(res)


def create_grads(nrow, ncol):
    res = ""
    for r in range(1, nrow + 1):
        res += f"torch::Tensor grad_r{r} = torch::empty(shape_row, torch::device(torch::kCUDA).dtype(torch::kFloat));\n"
    for c in range(1, ncol + 1):
        res += f"torch::Tensor grad_c{c} = torch::empty(shape_col, torch::device(torch::kCUDA).dtype(torch::kFloat));\n"
    return res


def grad_accessors(nrow, ncol):
    res = ""
    for r in range(1, nrow + 1):
        res += f"auto grad_r{r}_acc = grad_r{r}.packed_accessor64<float, 2>();\n"
    for c in range(1, ncol + 1):
        res += f"auto grad_c{c}_acc = grad_c{c}.packed_accessor64<float, 2>();\n"
    return res


def call_backward(nrow, ncol):
    res = []
    for r in range(1, nrow + 1):
        res.append(f"r{r}_acc")
    for c in range(1, ncol + 1):
        res.append(f"c{c}_acc")
    for r in range(1, nrow + 1):
        res.append(f"grad_r{r}_acc")
    for c in range(1, ncol + 1):
        res.append(f"grad_c{c}_acc")
    return ", ".join(res)


def return_expr(nrow, ncol):
    res = []
    for r in range(1, nrow + 1):
        res.append(f"grad_r{r}")
    for c in range(1, ncol + 1):
        res.append(f"grad_c{c}")
    tensors = ", ".join(res)
    return f"Tensor{nrow + ncol}({tensors})"


def generate_cpp(nrow, ncol):
    return f"""

#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <cuda_runtime.h>
#include <utility>

#define CHECK_CUDA(x) \\
    TORCH_CHECK(x.is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \\
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \\
    CHECK_CUDA(x);     \\
    CHECK_CONTIGUOUS(x)

void fwd_launcher(
                  {operand_params(nrow, ncol)}
                  torch::PackedTensorAccessor64<float, 2> result,
                  int M, int N, int K);

void bwd_launcher(const torch::PackedTensorAccessor64<float, 2> gout,
                  {operand_params(nrow, ncol)}
                  {grad_params(nrow, ncol)}
                  int M, int N, int K);

{using_tensor_tuple(nrow + ncol)};

torch::Tensor forward_1_3({def_forward(nrow, ncol)})
{{

    // first check inputs!
    {check_inputs(nrow, ncol)}

    // get metadata
    int M = r1.size(0);
    int K = r1.size(1);
    int N = c1.size(0);

    // get data accesser
    {accessors(nrow, ncol)}

    // create result tensor
    torch::Tensor result = torch::empty({{M, N}}, torch::device(torch::kCUDA).dtype(torch::kFloat));

    auto result_acc = result.packed_accessor64<float, 2>();

    fwd_launcher({call_forward(nrow, ncol)}, result_acc, M, N, K);

    // cudaDeviceSynchronize();

    return result;
}}

Tensor{nrow + ncol} backward_1_3(torch::Tensor gout, {def_forward(nrow, ncol)})
{{

    // first check inputs!
    {check_inputs(nrow, ncol)}


    // get metadata
    int M = r1.size(0);
    int K = r1.size(1);
    int N = c1.size(0);

    // get data accesser
    const auto gout_acc = gout.packed_accessor64<float, 2>();

    {accessors(nrow, ncol)}

    std::vector<int64_t> shape_row = {{M, K}};
    std::vector<int64_t> shape_col = {{N, K}};

    {create_grads(nrow, ncol)}

    {grad_accessors(nrow, ncol)}

    bwd_launcher(gout_acc, {call_backward(nrow, ncol)}, M, N, K);

    // cudaDeviceSynchronize();

    return {return_expr(nrow, ncol)};
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{{
    m.def("forward", &forward_1_3, "forward R = {nrow}, C = {ncol}");
    m.def("backward", &backward_1_3, "backward R = {nrow}, C = {ncol}");
}}

"""
