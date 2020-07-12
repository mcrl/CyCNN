#include <torch/extension.h>

#include <vector>
#include <cstdio>


torch::Tensor cyconv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor workspace,
    int stride,
    int padding,
    int dilation);

std::vector<torch::Tensor> cyconv2d_cuda_backward(
    torch::Tensor input,
    torch::Tensor grad_output,
    torch::Tensor weight,
    torch::Tensor workspace,
    int stride,
    int padding,
    int dilation);

#define CHECK_CUDA(x) \
  AT_ASSERTM((x).type().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM((x).is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (0)

torch::Tensor cyconv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor workspace,
    int stride,
    int padding,
    int dilation)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(workspace);

  // printf("cyconv2d_forward called\n");

  return cyconv2d_cuda_forward(
      input, weight, workspace, stride, padding, dilation);
}

std::vector<torch::Tensor> cyconv2d_backward(
    torch::Tensor input,
    torch::Tensor grad_output,
    torch::Tensor weight,
    torch::Tensor workspace,
    int stride,
    int padding,
    int dilation)
{
  CHECK_INPUT(input);
  CHECK_INPUT(grad_output);
  CHECK_INPUT(weight);
  CHECK_INPUT(workspace);

  return cyconv2d_cuda_backward(
      input, grad_output, weight, workspace, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",
        &cyconv2d_forward,
        "CyConv2d forward (CUDA)",
        py::arg("input"),
        py::arg("weight"),
        py::arg("workspace"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"));
  m.def("backward",
        &cyconv2d_backward,
        "CyConv2d backward (CUDA)",
        py::arg("input"),
        py::arg("grad_output"),
        py::arg("weight"),
        py::arg("workspace"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"));
}
