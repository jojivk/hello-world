//            c_runtime_api.cc            
/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_runtime_api.cc
 * \brief Device specific implementations
 */
#include "./c_runtime_api.h"
#include "./cpu_device_api.h"
#include "./cuda_device_api.h"
#include "./runtime_base.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <thread>

namespace dlsys {
    namespace runtime {

        class DeviceAPIManager {
        public:
            static const int kMaxDeviceAPI = 8;

            // Get API
            static DeviceAPI *Get(DLContext ctx) {
                return Global()->GetAPI(ctx.device_type);
            }

        private:
            std::array<DeviceAPI *, kMaxDeviceAPI> api_;

            DeviceAPIManager() {
                std::fill(api_.begin(), api_.end(), nullptr);
                static CPUDeviceAPI cpu_device_api_inst;
                static CUDADeviceAPI gpu_device_api_inst;
                api_[kCPU] = static_cast<DeviceAPI *>(&cpu_device_api_inst);
                api_[kGPU] = static_cast<DeviceAPI *>(&gpu_device_api_inst);
            }

            // Get global static variable.
            static DeviceAPIManager *Global() {
                static DeviceAPIManager inst;
                return &inst;
            }

            // Get API.
            DeviceAPI *GetAPI(DLDeviceType type) {
                if (api_[type] == nullptr) {
                    std::cerr << "Device API not supported" << std::endl;
                    exit(EXIT_FAILURE);
                }
                return api_[type];
            }
        };

        inline DLArray *DLArrayCreate_() {
            DLArray *arr = new DLArray();
            arr->shape = nullptr;
            arr->ndim = 0;
            arr->data = nullptr;
            return arr;
        }

        inline void DLArrayFree_(DLArray *arr) {
            if (arr != nullptr) {
                // ok to delete nullptr
                delete[] arr->shape;
                if (arr->data != nullptr) {
                    DeviceAPIManager::Get(arr->ctx)->FreeDataSpace(arr->ctx, arr->data);
                }
            }
            delete arr;
        }

        inline size_t GetDataSize(DLArray *arr) {
            size_t size = 1;
            for (index_t i = 0; i < arr->ndim; ++i) {
                size *= arr->shape[i];
            }
            // assume 32-bit float
            size *= 4;
            return size;
        }

        inline size_t GetDataAlignment(DLArray *arr) {
            // assume 32-bit float
            return 8;
        }

    } // namespace runtime
} // namespace dlsys

using namespace dlsys::runtime;

int DLArrayAlloc(const index_t *shape, index_t ndim, DLContext ctx,
                 DLArrayHandle *out) {
    DLArray *arr = nullptr;
    API_BEGIN() ;
        // shape
        arr = DLArrayCreate_();
        // ndim
        arr->ndim = ndim;
        index_t *shape_copy = new index_t[ndim];
        std::copy(shape, shape + ndim, shape_copy);
        arr->shape = shape_copy;
        // ctx
        arr->ctx = ctx;
        size_t size = GetDataSize(arr);
        size_t alignment = GetDataAlignment(arr);
        arr->data = DeviceAPIManager::Get(ctx)->AllocDataSpace(ctx, size, alignment);
        *out = arr;
    API_END_HANDLE_ERROR(DLArrayFree_(arr));
}

int DLArrayFree(DLArrayHandle handle) {
    API_BEGIN() ;
        DLArray *arr = handle;
        DLArrayFree_(arr);
    API_END();
}

int DLArrayReshape(const DLArrayHandle handle, const index_t *new_shape, index_t new_dim) {
    API_BEGIN() ;
        DLArray *arr = handle;

        index_t *shape_copy = new index_t[new_dim];
        std::copy(new_shape, new_shape + new_dim, shape_copy);
        arr->shape = shape_copy;
        arr->ndim = new_dim;
    API_END();
}

int DLArrayCopyFromTo(DLArrayHandle from, DLArrayHandle to,
                      DLStreamHandle stream) {
    API_BEGIN() ;
        size_t from_size = GetDataSize(from);
        size_t to_size = GetDataSize(to);
        // The size must exactly match
        assert(from_size == to_size);
        DLContext ctx = from->ctx;
        if (ctx.device_type == kCPU) {
            ctx = to->ctx;
        } else {
            // Can not copy across different ctx types directly
            assert((to->ctx.device_type == kCPU) ||
                   (to->ctx.device_type == from->ctx.device_type));
        }
        DeviceAPIManager::Get(ctx)->CopyDataFromTo(from->data, to->data, from_size,
                                                   from->ctx, to->ctx, stream);
    API_END();
}
==========================================================
            c_runtime_api.h            
==========================================================
/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_runtime_api.h
 * \brief DL runtime library.
 *
 */

#ifndef DLSYS_RUNTIME_C_RUNTIME_API_H_
#define DLSYS_RUNTIME_C_RUNTIME_API_H_

#ifdef __cplusplus
#define DLSYS_EXTERN_C extern "C"
#else
#define DLSYS_EXTERN_C
#endif

#include "dlarray.h"
#include <stddef.h>
#include <stdint.h>

DLSYS_EXTERN_C {
/*! \brief type of array index. */
typedef int64_t index_t;

/*! \brief the array handle */
typedef DLArray *DLArrayHandle;
/*!
 * \brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void *DLStreamHandle;

// Array related apis for quick proptying
/*!
 * \brief Allocate a nd-array's memory,
 *  including space of shape, of given spec.
 *
 * \param shape The shape of the array, the data content will be copied to out
 * \param ndim The number of dimension of the array.
 * \param ctx The ctx this array sits on.
 * \param out The output handle.
 * \return 0 when success, -1 when failure happens
 */
int DLArrayAlloc(const index_t *shape, index_t ndim, DLContext ctx,
                 DLArrayHandle *out);

/*!
 * \brief Free the DL Array.
 * \param handle The array handle to be freed.
 * \return 0 when success, -1 when failure happens
 */
int DLArrayFree(DLArrayHandle handle);

/*!
 * \brief Copy the array, both from and to must be valid during the copy.
 * \param from The array to be copied from.
 * \param to The target space.
 * \param stream The stream where the copy happens, can be NULL.
 * \return 0 when success, -1 when failure happens
 */
int DLArrayCopyFromTo(DLArrayHandle from, DLArrayHandle to,
                      DLStreamHandle stream);

/*!
 * \brief Set all array elements to given value.
 * \param arr The array to be Set.
 * \param value The target value.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuArraySet(DLArrayHandle arr, float value);


int DLArrayReshape(const DLArrayHandle handle, const index_t *new_shape, index_t new_dim);

/*!
 * \brief Broadcast input array to output array.
 * \param input The input array.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output);

/*!
 * \brief Reduce sum input array by axis=0 and store to output.
 * \param input The input array.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output);

/*!
 * \brief Elementwise add two matrices and store to output.
 * \param matA The left input array.
 * \param matB The right input array.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output);

/*!
 * \brief Add matrix by const and store to output.
 * \param input The input array.
 * \param val The constant.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output);


int DLGpuMatrixElementwiseSubtract(const DLArrayHandle matA,
                                   const DLArrayHandle matB, DLArrayHandle output);

int DLGpuMatrixElementwiseSubtractByConst(const DLArrayHandle input, float val,
                                          DLArrayHandle output);

/*!
 * \brief Elementwise multiply two matrices and store to output.
 * \param matA The left input array.
 * \param matB The right input array.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuMatrixElementwiseMultiply(
        const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output);

/*!
 * \brief Multiply matrix by const and store to output.
 * \param input The input array.
 * \param val The constant.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output);


// TODO: (upul) documentation
int DLGpuMatrixElementwiseDiv(const DLArrayHandle matA,
                              const DLArrayHandle matB,
                              DLArrayHandle output);

// TODO: (upul) documentation
int DLGpuMatrixElementwiseDivByConst(const DLArrayHandle matA, float val,
                                     DLArrayHandle output);

/*!
 * \brief Matrix multiply two matrices and store to output.
 * \param matA The left input array.
 * \param transposeA Whether matA needs to be transposed
 * \param matB The right input array.
 * \param transposeB Whether matB needs to be transposed
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC);

/*!
 * \brief Compute relu on all array elements, and store to output.
 * \param input The input array.
 * \param output The output value.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output);

/*!
 * \brief Compute relu gradient, and store to output.
 * \param input The input array.
 * \param in_grad The input gradients value.
 * \param output The output array.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output);

/*!
 * \brief Compute softmax on matrix, and store to output.
 * \param input The input array.
 * \param output The output value.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output);

/*!
 * \brief Compute softmax_cross_entropy.
 *  np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
 * \param input_a The y array.
 * \param input_b The y_ array.
 * \param output The output value.
 * \return 0 when success, -1 when failure happens
 */
int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output);

int DLGpuMatrixElementwiseSqrt(const DLArrayHandle input_a, DLArrayHandle output);

/*
* CUDNN....
*/
int cudnnReLUForward(const DLArrayHandle input, DLArrayHandle output);

int cudnnConv2DForward(const DLArrayHandle input,
                       const DLArrayHandle filter,
                       const DLArrayHandle bias,
                       const int stride_height,
                       const int stride_width,
                       const int padding_height,
                       const int padding_width,
                       DLArrayHandle output);

int cudnnPoolForward(const DLArrayHandle input,
                     const int pooling_height,
                     const int pooling_width,
                     const int stride_height,
                     const int stride_width,
                     const char *mode,
                     DLArrayHandle output);

int cudnnPoolBackward(const DLArrayHandle input,
                      const DLArrayHandle output_grads,
                      const DLArrayHandle output,
                      const int pooling_height,
                      const int pooling_width,
                      const int stride_height,
                      const int stride_width,
                      const char *mode,
                      DLArrayHandle pool_grad);

int cudnnConv2DBackwardFilter(const DLArrayHandle input,
                              const DLArrayHandle output_grads,
                              const int stride_height,
                              const int stride_width,
                              const int padding_height,
                              const int padding_width,
                              DLArrayHandle filter_grad);

int cudnnConv2DBackwardData(const DLArrayHandle filter,
                            const DLArrayHandle output_grads,
                            const int stride_height,
                            const int stride_width,
                            const int padding_height,
                            const int padding_width,
                            DLArrayHandle data_grad);

int cudnnConv2DBackwardBias(const DLArrayHandle output_grads,
                            DLArrayHandle bias_grads);

} // DLSYS_EXTERN_C

#endif // DLSYS_RUNTIME_C_RUNTIME_API_H_
==========================================================
            cpu_device_api.cc            
==========================================================
/*!
 *  Copyright (c) 2017 by Contributors
 * \file cpu_device_api.cc
 */
#include "./cpu_device_api.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace dlsys {
    namespace runtime {

        void *CPUDeviceAPI::AllocDataSpace(DLContext ctx, size_t size,
                                           size_t alignment) {
            // std::cout << "allocating cpu data" << std::endl;
            void *ptr;
            int ret = posix_memalign(&ptr, alignment, size);
            if (ret != 0)
                throw std::bad_alloc();
            return ptr;
        }

        void CPUDeviceAPI::FreeDataSpace(DLContext ctx, void *ptr) { free(ptr); }

        void CPUDeviceAPI::CopyDataFromTo(const void *from, void *to, size_t size,
                                          DLContext ctx_from, DLContext ctx_to,
                                          DLStreamHandle stream) {
            // std::cout << "copying cpu data" << std::endl;
            memcpy(to, from, size);
        }

        void CPUDeviceAPI::StreamSync(DLContext ctx, DLStreamHandle stream) {}

    } // namespace runtime
} // namespace dlsys
==========================================================
            cpu_device_api.h            
==========================================================
/*!
 *  Copyright (c) 2017 by Contributors
 * \file device_api.h
 * \brief Device specific API
 */
#ifndef DLSYS_RUNTIME_CPU_DEVICE_API_H_
#define DLSYS_RUNTIME_CPU_DEVICE_API_H_

#include "c_runtime_api.h"
#include "device_api.h"
#include <assert.h>
#include <string>

namespace dlsys {
    namespace runtime {

        class CPUDeviceAPI : public DeviceAPI {
        public:
            void *AllocDataSpace(DLContext ctx, size_t size, size_t alignment) final;

            void FreeDataSpace(DLContext ctx, void *ptr) final;

            void CopyDataFromTo(const void *from, void *to, size_t size,
                                DLContext ctx_from, DLContext ctx_to, DLStreamHandle stream) final;

            void StreamSync(DLContext ctx, DLStreamHandle stream) final;
        };

    } // namespace runtime
} // namespace dlsys
#endif // DLSYS_RUNTIME_CPU_DEVICE_API_H_
==========================================================
            cuda_device_api.cc            
==========================================================
/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */

#include "./cuda_device_api.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CALL(func)                                                        \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    assert((e == cudaSuccess) || (e == cudaErrorCudartUnloading));             \
  }

namespace dlsys {
    namespace runtime {

        static void GPUCopy(const void *from, void *to, size_t size,
                            cudaMemcpyKind kind, cudaStream_t stream) {
            if (stream != 0) {
                CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
            } else {
                CUDA_CALL(cudaMemcpy(to, from, size, kind));
            }
        }

        void *CUDADeviceAPI::AllocDataSpace(DLContext ctx, size_t size,
                                            size_t alignment) {
            //std::cout << "allocating cuda data" << std::endl;
            CUDA_CALL(cudaSetDevice(ctx.device_id));
            assert((256 % alignment) == 0U); // << "CUDA space is aligned at 256 bytes";
            void *ret;
            CUDA_CALL(cudaMalloc(&ret, size));
            return ret;
        }

        void CUDADeviceAPI::FreeDataSpace(DLContext ctx, void *ptr) {
            //std::cout << "releasing cuda data" << std::endl;
            CUDA_CALL(cudaSetDevice(ctx.device_id));
            CUDA_CALL(cudaFree(ptr));
        }

        void CUDADeviceAPI::CopyDataFromTo(const void *from, void *to, size_t size,
                                           DLContext ctx_from, DLContext ctx_to, DLStreamHandle stream) {
            //std::cout << "copying cuda data" << std::endl;
            cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
            if (ctx_from.device_type == kGPU && ctx_to.device_type == kGPU) {
                CUDA_CALL(cudaSetDevice(ctx_from.device_id));
                if (ctx_from.device_id == ctx_to.device_id) {
                    GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
                } else {
                    cudaMemcpyPeerAsync(to, ctx_to.device_id, from, ctx_from.device_id,
                                        size, cu_stream);
                }
            } else if (ctx_from.device_type == kGPU && ctx_to.device_type == kCPU) {
                CUDA_CALL(cudaSetDevice(ctx_from.device_id));
                GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
            } else if (ctx_from.device_type == kCPU && ctx_to.device_type == kGPU) {
                CUDA_CALL(cudaSetDevice(ctx_to.device_id));
                GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
            } else {
                std::cerr << "expect copy from/to GPU or between GPU" << std::endl;
            }
        }

        void CUDADeviceAPI::StreamSync(DLContext ctx, DLStreamHandle stream) {
            CUDA_CALL(cudaSetDevice(ctx.device_id));
            CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
        }

    } // namespace runtime
} // namespace dlsys
==========================================================
            cuda_device_api.h            
==========================================================
/*!
 *  Copyright (c) 2017 by Contributors
 * \file device_api.h
 * \brief Device specific API
 */
#ifndef DLSYS_RUNTIME_CUDA_DEVICE_API_H_
#define DLSYS_RUNTIME_CUDA_DEVICE_API_H_

#include "c_runtime_api.h"
#include "device_api.h"
#include <cuda_runtime.h>

#include <assert.h>
#include <string>

namespace dlsys {
    namespace runtime {

        class CUDADeviceAPI : public DeviceAPI {
        public:
            void *AllocDataSpace(DLContext ctx, size_t size, size_t alignment) final;

            void FreeDataSpace(DLContext ctx, void *ptr) final;

            void CopyDataFromTo(const void *from, void *to, size_t size,
                                DLContext ctx_from, DLContext ctx_to,
                                DLStreamHandle stream) final;

            void StreamSync(DLContext ctx, DLStreamHandle stream) final;
        };

    } // namespace runtime
} // namespace dlsys
#endif // DLSYS_RUNTIME_CUDA_DEVICE_API_H_
==========================================================
            cudnn_operations.cu            
==========================================================
#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <string>

#define checkCUDNN(expression)                                  \
{                                                               \
    cudnnStatus_t status = (expression);                        \
    if (status != CUDNN_STATUS_SUCCESS) {                       \
        std::cerr  << "Error on line " << __LINE__ << ": "      \
                   << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE);                                \
    }                                                           \
}

int setTensorDescriptor(cudnnTensorDescriptor_t activationDesc,
                        const int numDim,
                        const long shape[]) {
    int batchSize = 0;
    int channels = 0;
    switch (numDim) {
        case 2:
            batchSize = shape[0];
            channels = shape[1];
            checkCUDNN(cudnnSetTensor4dDescriptor(activationDesc,
                                                  CUDNN_TENSOR_NCHW,
                                                  CUDNN_DATA_FLOAT,
                                                  batchSize,
                                                  channels, 1, 1));
            break;

        case 4:
            batchSize = shape[0];
            channels = shape[1];
            int height = shape[2];
            int width = shape[3];
            checkCUDNN(cudnnSetTensor4dDescriptor(activationDesc,
                                                  CUDNN_TENSOR_NCHW,
                                                  CUDNN_DATA_FLOAT,
                                                  batchSize,
                                                  channels,
                                                  height,
                                                  width));
            break;
            // TODO: handle other cases and errors

    }
    return 0;
}

cudnnHandle_t cudnn_handler = NULL;

int cudnnReLUForward(const DLArrayHandle input, DLArrayHandle output) {
    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;

    assert(input->shape[0] == output->shape[0]);
    assert(input->shape[1] == output->shape[1]);

    if (!cudnn_handler) {
        cudnnCreate(&cudnn_handler);
    }

    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                            CUDNN_ACTIVATION_RELU, // type of activation
                                            CUDNN_PROPAGATE_NAN, // reluNanOpt
                                            0));  //relu_coef

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    setTensorDescriptor(input_descriptor, input->ndim, input->shape);

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    setTensorDescriptor(output_descriptor, output->ndim, output->shape);

    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnActivationForward(cudnn_handler,
                                      activation_descriptor,
                                      &alpha,
                                      input_descriptor,
                                      input_data,
                                      &beta,
                                      output_descriptor,
                                      output_data));

    cudnnDestroyActivationDescriptor(activation_descriptor);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);

    return 0;
}

int cudnnConv2DForward(const DLArrayHandle input,
                       const DLArrayHandle filter,
                       const DLArrayHandle bias,
                       const int stride_height,
                       const int stride_width,
                       const int padding_height,
                       const int padding_width,
                       DLArrayHandle output) {

    const int input_dim = input->ndim;
    const int output_dim = output->ndim;
    assert(input_dim == 4);
    assert(output_dim == 4);

    const int filter_shape = filter->ndim;
    assert(filter_shape == 4);
    const int num_filters = filter->shape[0];
    const int num_outputs = filter->shape[1];
    const int filter_height = filter->shape[2];
    const int filter_width = filter->shape[3];

    const int bias_dim = bias->ndim;
    assert(bias_dim == 1);
    assert(bias->shape[0] == num_filters);

    const float *input_data = (const float *) input->data;
    const float *filter_date = (const float *) filter->data;
    const float *bias_data = (const float *) bias->data;
    float *output_data = (float *) output->data;

    if (!cudnn_handler) {
        cudnnCreate(&cudnn_handler);
    }


    // creating input and output tensors
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    setTensorDescriptor(input_descriptor, input->ndim, input->shape);

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    setTensorDescriptor(output_descriptor, output->ndim, output->shape);

    // create filter tensors
    cudnnFilterDescriptor_t filter_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/num_filters,
            /*in_channels=*/num_outputs,
            /*kernel_height=*/filter_height,
            /*kernel_width=*/filter_width));
    // create convolution tensor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
            /*pad_height=*/padding_height,
            /*pad_width=*/padding_width,
            /*vertical_stride=*/stride_height,
            /*horizontal_stride=*/stride_width,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION,
            /*computeType=*/CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handler,
                                                   input_descriptor,
                                                   filter_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            /*memoryLimitInBytes=*/0,
                                                   &convolution_algorithm));

    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handler,
                                                       input_descriptor,
                                                       filter_descriptor,
                                                       convolution_descriptor,
                                                       output_descriptor,
                                                       convolution_algorithm,
                                                       &workspace_bytes));
    //std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;
    assert(workspace_bytes > 0);

    void *d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    const float alpha = 1.0f, beta = 0.0f;

    checkCUDNN(cudnnConvolutionForward(cudnn_handler,
                                       &alpha,
                                       input_descriptor,
                                       input_data,
                                       filter_descriptor,
                                       filter_date,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       output_data));

    // adding bias tensor
    cudnnTensorDescriptor_t bias_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
    //setTensorDescriptor(bias_descriptor, bias->ndim, bias->shape);
    checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          num_filters,
                                          1,
                                          1));
    checkCUDNN(cudnnAddTensor(cudnn_handler,
                              &alpha,
                              bias_descriptor,
                              bias_data,
                              &alpha,
                              output_descriptor,
                              output_data));

    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyTensorDescriptor(bias_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    return 0;
}

int cudnnConv2DBackwardBias(const DLArrayHandle output_grads,
                            DLArrayHandle bias_grads) {

    const float *output_grads_data = (const float *) output_grads->data;
    float *bias_grads_data = (float *) bias_grads->data;

    const int bias_grads_dim = bias_grads->ndim;
    assert(bias_grads_dim == 1);
    const int num_filters = bias_grads->shape[0];

    if (!cudnn_handler) {
        cudnnCreate(&cudnn_handler);
    }

    // creating output_grads descriptor
    cudnnTensorDescriptor_t output_grads_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_grads_descriptor));
    setTensorDescriptor(output_grads_descriptor, output_grads->ndim, output_grads->shape);

    // bias descriptor
    cudnnTensorDescriptor_t bias_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1,
                                          num_filters,
                                          1,
                                          1));

    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionBackwardBias(cudnn_handler,
                                            &alpha,
                                            output_grads_descriptor,
                                            output_grads_data,
                                            &beta,
                                            bias_descriptor,
                                            bias_grads_data
    ));

    cudnnDestroyTensorDescriptor(bias_descriptor);
    cudnnDestroyTensorDescriptor(output_grads_descriptor);

    return 0;
}


int cudnnConv2DBackwardData(const DLArrayHandle filter,
                            const DLArrayHandle output_grads,
                            const int stride_height,
                            const int stride_width,
                            const int padding_height,
                            const int padding_width,
                            DLArrayHandle data_grad) {

    //const int input_dim = input->ndim;
    const int data_grad_dim = data_grad->ndim;
    //assert(input_dim == 4);
    assert(data_grad_dim == 4);

    const int filter_shape = filter->ndim;
    assert(filter_shape == 4);

    const int num_filters = filter->shape[0];
    const int num_outputs = filter->shape[1];
    const int filter_height = filter->shape[2];
    const int filter_width = filter->shape[3];

    //const float *input_data = (const float *) input->data;
    const float *filter_date = (const float *) filter->data;
    const float *output_grads_data = (const float *) output_grads->data;
    float *data_grad_data = (float *) data_grad->data;

    if (!cudnn_handler) {
        cudnnCreate(&cudnn_handler);
    }

    // creating output_grads descriptor
    cudnnTensorDescriptor_t output_grads_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_grads_descriptor));
    setTensorDescriptor(output_grads_descriptor, output_grads->ndim, output_grads->shape);

    // create convolution tensor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
            /*pad_height=*/padding_height,
            /*pad_width=*/padding_width,
            /*vertical_stride=*/stride_height,
            /*horizontal_stride=*/stride_width,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION,
            /*computeType=*/CUDNN_DATA_FLOAT));
    // create filter tensors
    cudnnFilterDescriptor_t filter_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/num_filters,
            /*in_channels=*/num_outputs,
            /*kernel_height=*/filter_height,
            /*kernel_width=*/filter_width));

    cudnnTensorDescriptor_t data_grads_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&data_grads_descriptor));
    setTensorDescriptor(data_grads_descriptor, data_grad->ndim, data_grad->shape);

    cudnnConvolutionBwdDataAlgo_t backward_data_algo;
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handler,
                                                        filter_descriptor,
                                                        output_grads_descriptor,
                                                        convolution_descriptor,
                                                        data_grads_descriptor,
                                                        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                        0,
                                                        &backward_data_algo));

    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handler,
                                                            filter_descriptor,
                                                            output_grads_descriptor,
                                                            convolution_descriptor,
                                                            data_grads_descriptor,
                                                            backward_data_algo,
                                                            &workspace_bytes));

    //std::cout << "workspace size: " << workspace_bytes << std::endl;

    void *d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionBackwardData(cudnn_handler,
                                            &alpha,
                                            filter_descriptor,
                                            filter_date,
                                            output_grads_descriptor,
                                            output_grads_data,
                                            convolution_descriptor,
                                            backward_data_algo,
                                            d_workspace,
                                            workspace_bytes,
                                            &beta,
                                            data_grads_descriptor,
                                            data_grad_data));

    // Release resources
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(data_grads_descriptor);
    cudnnDestroyTensorDescriptor(output_grads_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    //std::cout << "leaveing cudnnConv2DBackwardData" << std::endl;
    return 0;

}


int cudnnConv2DBackwardFilter(const DLArrayHandle input,
                              const DLArrayHandle output_grads,
                              const int stride_height,
                              const int stride_width,
                              const int padding_height,
                              const int padding_width,
                              DLArrayHandle filter_grad) {


    const int input_dim = input->ndim;
    const int filter_dim = filter_grad->ndim;
    //const int filter_grad_dim = filter_grad->ndim;
    assert(input_dim == 4);
    assert(filter_dim == 4);
    //assert(filter_grad_dim == filter_dim);

    const int num_filters = filter_grad->shape[0];
    const int num_outputs = filter_grad->shape[1];
    const int filter_height = filter_grad->shape[2];
    const int filter_width = filter_grad->shape[3];

    const float *input_data = (const float *) input->data;
    const float *output_grads_data = (const float *) output_grads->data;
    //const float *filter_date = (const float *) filter->data;
    float *filter_grad_data = (float *) filter_grad->data;

    //cudnnHandle_t cudnn;
    //cudnnCreate(&cudnn);
    if (!cudnn_handler) {
        cudnnCreate(&cudnn_handler);
    }

    // creating input descriptor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    setTensorDescriptor(input_descriptor, input->ndim, input->shape);

    // creating output_grads descriptor
    cudnnTensorDescriptor_t output_grads_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_grads_descriptor));
    setTensorDescriptor(output_grads_descriptor, output_grads->ndim, output_grads->shape);

    // create convolution tensor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
            /*pad_height=*/padding_height,
            /*pad_width=*/padding_width,
            /*vertical_stride=*/stride_height,
            /*horizontal_stride=*/stride_width,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION,
            /*computeType=*/CUDNN_DATA_FLOAT));


    // create filter tensors
    cudnnFilterDescriptor_t filter_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/num_filters,
            /*in_channels=*/num_outputs,
            /*kernel_height=*/filter_height,
            /*kernel_width=*/filter_width));


    cudnnConvolutionBwdFilterAlgo_t backward_filter_algo;
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handler,
                                                          input_descriptor,
                                                          output_grads_descriptor,
                                                          convolution_descriptor,
                                                          filter_descriptor,
                                                          CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                          0,
                                                          &backward_filter_algo));

    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handler,
                                                              input_descriptor,
                                                              output_grads_descriptor,
                                                              convolution_descriptor,
                                                              filter_descriptor,
                                                              backward_filter_algo,
                                                              &workspace_bytes));
    void *d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handler,
                                              &alpha,
                                              input_descriptor,
                                              input_data,
                                              output_grads_descriptor,
                                              output_grads_data,
                                              convolution_descriptor,
                                              backward_filter_algo,
                                              d_workspace,
                                              workspace_bytes,
                                              &beta,
                                              filter_descriptor,
                                              filter_grad_data));


    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_grads_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    return 0;
}


int cudnnPoolForward(const DLArrayHandle input,
                     const int pooling_height,
                     const int pooling_width,
                     const int stride_height,
                     const int stride_width,
                     const char *mode,
                     DLArrayHandle output) {

    const int input_dim = input->ndim;
    const int output_dim = output->ndim;
    assert(input_dim == 4);
    assert(output_dim == 4);

    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;

    cudnnPoolingMode_t pooling_mode = CUDNN_POOLING_MAX;
    std::string str_mode(mode);
    if (str_mode.compare("average") == 0) {
        std::cout << str_mode << std::endl;
        pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        std::cout << pooling_mode << std::endl;
    }

    if (!cudnn_handler) {
        cudnnCreate(&cudnn_handler);
    }

    // creating input and output tensors
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    setTensorDescriptor(input_descriptor, input->ndim, input->shape);

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    setTensorDescriptor(output_descriptor, output->ndim, output->shape);

    cudnnPoolingDescriptor_t pooling_descriptor;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    checkCUDNN(cudnnSetPooling2dDescriptor(pooling_descriptor,
                                           pooling_mode,
                                           CUDNN_PROPAGATE_NAN,
                                           pooling_height,
                                           pooling_width,
                                           0, // TODO: parameterize vertical padding
                                           0, // TODO: parameterize horizontal padding
                                           stride_height,
                                           stride_width));


    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnPoolingForward(cudnn_handler,
                                   pooling_descriptor,
                                   &alpha,
                                   input_descriptor,
                                   input_data,
                                   &beta,
                                   output_descriptor,
                                   output_data));

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyPoolingDescriptor(pooling_descriptor);

    return 0;
}

int cudnnPoolBackward(const DLArrayHandle input,
                      const DLArrayHandle output_grads,
                      const DLArrayHandle output,
                      const int pooling_height,
                      const int pooling_width,
                      const int stride_height,
                      const int stride_width,
                      const char *mode,
                      DLArrayHandle pool_grad) {


    const int input_dim = input->ndim;
    const int output_dim = output->ndim;
    const int output_grads_dim = output_grads->ndim;
    const int pool_grad_dim = pool_grad->ndim;
    assert(input_dim == 4);
    assert(output_dim == 4);
    assert(output_grads_dim == 4);
    assert(pool_grad_dim == 4);

    const float *input_data = (const float*) input->data;
    const float *output_data = (const float*) output->data;
    const float *output_grads_data = (const float*) output_grads->data;
    float *pool_grad_data = (float*) pool_grad->data;

    cudnnPoolingMode_t pooling_mode = CUDNN_POOLING_MAX;
    std::string str_mode(mode);
    if (str_mode.compare("average") == 0) {
        std::cout << str_mode << std::endl;
        pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        std::cout << pooling_mode << std::endl;
    }

    if (!cudnn_handler) {
        cudnnCreate(&cudnn_handler);
    }

    // input descriptor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    setTensorDescriptor(input_descriptor, input->ndim, input->shape);

    // ouput descriptor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    setTensorDescriptor(output_descriptor, output->ndim, output->shape);

    // output grad descriptor
    cudnnTensorDescriptor_t output_grad_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_grad_descriptor));
    setTensorDescriptor(output_grad_descriptor, output_grads->ndim, output_grads->shape);

    // pool grad descriptor
    cudnnTensorDescriptor_t pool_grad_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&pool_grad_descriptor));
    setTensorDescriptor(pool_grad_descriptor, pool_grad->ndim, pool_grad->shape);

    // TODO: reuse already defined pooling descriptor in forward pass
    cudnnPoolingDescriptor_t pooling_descriptor;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    checkCUDNN(cudnnSetPooling2dDescriptor(pooling_descriptor,
                                           pooling_mode,
                                           CUDNN_PROPAGATE_NAN,
                                           pooling_height,
                                           pooling_width,
                                           0, // TODO: parameterize vertical padding
                                           0, // TODO: parameterize horizontal padding
                                           stride_height,
                                           stride_width));


    const float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnPoolingBackward(cudnn_handler,
                                   pooling_descriptor,
                                   &alpha,
                                   output_descriptor,
                                   output_data,
                                   output_grad_descriptor,
                                   output_grads_data,
                                   input_descriptor,
                                   input_data,
                                   &beta,
                                   pool_grad_descriptor,
                                   pool_grad_data));

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyTensorDescriptor(output_grad_descriptor);
    cudnnDestroyTensorDescriptor(pool_grad_descriptor);
    cudnnDestroyPoolingDescriptor(pooling_descriptor);

    return 0;
}





==========================================================
            device_api.h            
==========================================================
/*!
 *  Copyright (c) 2017 by Contributors
 * \file device_api.h
 * \brief Device specific API
 */
#ifndef DLSYS_RUNTIME_DEVICE_API_H_
#define DLSYS_RUNTIME_DEVICE_API_H_

#include "c_runtime_api.h"
#include <assert.h>
#include <string>

namespace dlsys {
    namespace runtime {

        class DeviceAPI {
        public:
            /*! \brief virtual destructor */
            virtual ~DeviceAPI() {}

            /*!
             * \brief Allocate a data space on device.
             * \param ctx The device context to perform operation.
             * \param size The size of the memory
             * \param alignment The alignment of the memory.
             * \return The allocated device pointer
             */
            virtual void *AllocDataSpace(DLContext ctx, size_t size,
                                         size_t alignment) = 0;

            /*!
             * \brief Free a data space on device.
             * \param ctx The device context to perform operation.
             * \param ptr The data space.
             * \tparam xpu The device mask.
             */
            virtual void FreeDataSpace(DLContext ctx, void *ptr) = 0;

            /*!
             * \brief copy data from one place to another
             * \param dev The device to perform operation.
             * \param from The source array.
             * \param to The target array.
             * \param size The size of the memory
             * \param ctx_from The source context
             * \param ctx_to The target context
             */
            virtual void CopyDataFromTo(const void *from, void *to, size_t size,
                                        DLContext ctx_from, DLContext ctx_to,
                                        DLStreamHandle stream) = 0;

            /*!
             * \brief Synchronize the stream
             * \param ctx The context to perform operation.
             * \param stream The stream to be sync.
             */
            virtual void StreamSync(DLContext ctx, DLStreamHandle stream) = 0;
        };

    } // namespace runtime
} // namespace dlsys
#endif // DLSYS_RUNTIME_DEVICE_API_H_
==========================================================
            dlarray.h            
==========================================================
/*!
 *  Copyright (c) 2017 by Contributors
 * \file dlarray.h
 * \brief Header that defines array struct.
 */
#ifndef DLSYS_H_
#define DLSYS_H_

#ifdef __cplusplus
#define DLSYS_EXTERN_C extern "C"
#else
#define DLSYS_EXTERN_C
#endif

#include <stddef.h>
#include <stdint.h>

DLSYS_EXTERN_C {
/*!
 * \brief The device type in DLContext.
 */
typedef enum {
    kCPU = 1,
    kGPU = 2,
} DLDeviceType;

/*!
 * \brief A Device context for array.
 */
typedef struct {
    /*! \brief The device index */
    int device_id;
    /*! \brief The device type used in the device. */
    DLDeviceType device_type;
} DLContext;

/*!
 * \brief Plain C Array object, does not manage memory.
 */
typedef struct {
    /*!
     * \brief The opaque data pointer points to the allocated data.
     *  This will be CUDA device pointer or cl_mem handle in OpenCL.
     *  This pointer is always aligns to 256 bytes as in CUDA.
     */
    void *data;
    /*! \brief The device context of the tensor */
    DLContext ctx;
    /*! \brief Number of dimensions */
    int ndim;
    /*! \brief The shape of the tensor */
    int64_t *shape;
} DLArray;

} // DLSYS_EXTERN_C
#endif // DLSYS_H_
==========================================================
            gpu_op.cu            
==========================================================
#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a, const float *input_b, float *output) {
    // Dynamic shared memory, size provided at kernel launch.
    extern __shared__ float loss_per_row[];
    // Two dimensional thread blocks.
    int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x
            + threadIdx.x;
    if (y >= nrow) {
        return;
    }
    input_a += y * ncol;
    input_b += y * ncol;
    float maxval = *input_a;
    // Find max for a row.
    for (int x = 1; x < ncol; ++x) {
        maxval = max(maxval, input_a[x]);
    }
    // Deduct by max for a row, and raise to exp.
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
        sum += exp(input_a[x] - maxval);
    }
    // Compute per-row loss.
    float loss = 0;
    for (int x = 0; x < ncol; ++x) {
        loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
    }
    loss_per_row[y] = loss;
    __syncthreads();
    // Compute reduce_mean across rows.
    float mean_loss = 0;
    // Use a single thread to reduce mean across rows.
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        for (int i = 0; i < nrow; ++i) {
            mean_loss += loss_per_row[i];
        }
        mean_loss /= nrow;
        output[0] = mean_loss;
    }
}


__global__ void array_set_kernel(float *array, float value, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        array[index] = value;
    }
}


int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < arr->ndim; i++) {
        n = n * arr->shape[i];
    }

    float *array_data = (float *) arr->data;

    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    array_set_kernel << < num_blocks, threads_per_block >> > (array_data, value, n);
    return 0;
}


__global__ void broadcast_to_kernel(const float *input_data,
                                    float *output_data,
                                    index_t input_n,
                                    index_t output_n) {
    index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < output_n) {
        output_data[idx] = input_data[idx % input_n];
    }
}


int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    index_t input_n = 1;
    for (int i = 0; i < input->ndim; i++)
        input_n *= input->shape[i];

    index_t output_n = 1;
    for (int i = 0; i < output->ndim; i++)
        output_n *= output->shape[i];

    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;

    int thread_per_block = 512;
    int n_blocks = (output_n + thread_per_block - 1) / thread_per_block;
    broadcast_to_kernel << < n_blocks, thread_per_block >> > (input_data, output_data,
            input_n, output_n);
    return 0;
}

__global__ void reduced_sum_axis_zero(const float *input_data, float *output_data, int input_n, int output_n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < output_n) {
        output_data[idx] = 0.0;
        for (int i = 0; i < input_n / output_n; i++) {
            output_data[idx] += input_data[i * output_n + idx];
        }
    }
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    int input_n = 1;
    for (int i = 0; i < input->ndim; i++) {
        input_n *= input->shape[i];
    }

    int output_n = 1;
    for (int i = 0; i < output->ndim; i++) {
        output_n *= output->shape[i];
    }

    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;

    int thread_per_block = 1024;
    int n_blocks = (output_n + thread_per_block - 1) / thread_per_block;

    reduced_sum_axis_zero << < n_blocks, thread_per_block >> > (input_data, output_data, input_n, output_n);
    return 0;
}

__global__ void matrix_elementwise_add(const float *a, const float *b, float *c,
                                       int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
    /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < output->ndim; i++) {
        n = n * output->shape[i];
    }
    const float *data_A = (const float *) matA->data;
    const float *data_B = (const float *) matB->data;
    float *data_output = (float *) output->data;

    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    matrix_elementwise_add << < num_blocks, threads_per_block >> > (data_A, data_B,
            data_output, n);
    return 0;
}

__global__
void matrix_elementwise_subtract(const float *a, const float *b, float *c,
                                 int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] - b[index];
    }
}

int DLGpuMatrixElementwiseSubtract(const DLArrayHandle matA,
                                   const DLArrayHandle matB, DLArrayHandle output) {
    /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < output->ndim; i++) {
        n = n * output->shape[i];
    }
    const float *data_A = (const float *) matA->data;
    const float *data_B = (const float *) matB->data;
    float *data_output = (float *) output->data;

    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    matrix_elementwise_subtract << < num_blocks, threads_per_block >> > (data_A, data_B,
            data_output, n);
    return 0;
}

__global__
void matrix_elementwise_division(const float *a, const float *b, float *result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        result[index] = a[index] / b[index];
    }
}

int DLGpuMatrixElementwiseDiv(const DLArrayHandle matA, const DLArrayHandle matB,
                              DLArrayHandle output) {
    int n = 1;
    for (int i = 0; i < output->ndim; i++) {
        n = n * output->shape[i];
    }
    const float *data_A = (const float *) matA->data;
    const float *data_B = (const float *) matB->data;
    float *data_output = (float *) output->data;

    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    matrix_elementwise_division << < num_blocks, threads_per_block >> > (data_A, data_B,
            data_output, n);
    return 0;

}

__global__ void matrix_elementwise_add_by_const_kernal(const float *d_in,
                                                       float *d_out, float val, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        d_out[index] = d_in[index] + val;
    }
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
    /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < output->ndim; i++) {
        n = n * output->shape[i];
    }
    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    matrix_elementwise_add_by_const_kernal << < num_blocks, threads_per_block >> > (
            input_data, output_data, val, n);
    return 0;
}

__global__
void matrix_elementwise_subtract_by_const_kernal(const float *d_in,
                                                 float *d_out, float val, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        d_out[index] = d_in[index] - val;
    }
}

int DLGpuMatrixElementwiseSubtractByConst(const DLArrayHandle input, float val,
                                          DLArrayHandle output) {
    /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < output->ndim; i++) {
        n = n * output->shape[i];
    }
    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    matrix_elementwise_subtract_by_const_kernal << < num_blocks, threads_per_block >> > (
            input_data, output_data, val, n);
    return 0;
}


__global__ void matrix_elementwise_div_by_const_kernal(const float *d_in,
                                                       float *d_out, float val, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        d_out[index] = d_in[index] / val;
    }
}

int DLGpuMatrixElementwiseDivByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
    /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < output->ndim; i++) {
        n = n * output->shape[i];
    }
    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    matrix_elementwise_div_by_const_kernal << < num_blocks, threads_per_block >> > (
            input_data, output_data, val, n);
    return 0;
}


__global__ void elementwise_mul_kernel(const float *data_a, const float *data_b,
                                       float *output, int n) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        output[index] = data_a[index] * data_b[index];
    }
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB, DLArrayHandle output) {
    /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < output->ndim; i++) {
        n = n * output->shape[i];
    }

    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    const float *mat_a_data = (const float *) matA->data;
    const float *mat_b_data = (const float *) matB->data;
    float *output_data = (float *) output->data;

    elementwise_mul_kernel << < num_blocks, threads_per_block >> > (mat_a_data,
            mat_b_data, output_data, n);

    return 0;
}

__global__
void matrix_elementwise_sqrt(const float *d_input, float *d_output, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        d_output[index] = sqrt(d_input[index]);
    }
}

int DLGpuMatrixElementwiseSqrt(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < input->ndim; i++) {
        n *= input->shape[i];
    }

    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    matrix_elementwise_sqrt << < num_blocks, threads_per_block >> > (input_data, output_data, n);
    return 0;
}


__global__ void marix_multiply_by_const(const float *d_input, float *d_output,
                                        float val, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        d_output[index] = d_input[index] * val;
    }
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
    /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < input->ndim; i++) {
        n *= input->shape[i];
    }

    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    marix_multiply_by_const << < num_blocks, threads_per_block >> > (input_data,
            output_data, val, n);
    return 0;
}

// int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
// 		const DLArrayHandle matB, bool transposeB, DLArrayHandle matC) {
// 	/* TODO: Your code here */
// 	// Hint: use cublas
// 	// cublas assume matrix is column major
//     cublasHandle_t handle;
//     cublasStatus_t stat = cublasCreate(&handle);
//     if (stat != CUBLAS_STATUS_SUCCESS)
//         printf("CUBLAS initialization failed\n");

//     const float *matA_data = (const float *) matA->data;
//     const float *matB_data = (const float *) matB->data;
//     float *matC_data = (float *) matC->data;

//     cublasOperation_t transa = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
//     cublasOperation_t transb = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

//     int m = transposeB ? matB->shape[0] : matB->shape[1];
//     int n = transposeA ? matA->shape[1] : matA->shape[0];
//     int k = transposeA ? matA->shape[0] : matA->shape[1];

//     float alpha = 1.0f;
//     float beta = 0.0f;
//     stat = cublasSgemm(handle, transb, transa,
//                        m, n, k,
//                        &alpha, matB_data, matB->shape[1],
//                        matA_data, matA->shape[1],
//                        &beta, matC_data, m);

//     if (stat != CUBLAS_STATUS_SUCCESS)
//         printf("CUBLAS kernel execution error.\n");

//     stat = cublasDestroy(handle);
//     if (stat != CUBLAS_STATUS_SUCCESS)
//         printf("CUBLAS shutdown error\n");

//     return 0;
// }
cublasHandle_t cublas_handle = NULL;

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
    /* TODO: Your code here */
    // Hint: use cublas
    // cublas assume matrix is column major
    // op(A) * op(B) = C
    // op(B)T * op(A)T = CT

    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }

    float one = 1.0f;
    float zero = 0.0f;
    int m = matC->shape[1];
    int n = matC->shape[0];
    int k = transposeA ? matA->shape[0] : matA->shape[1];

    cublasSgemm(cublas_handle,
                transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
                m, n, k,
                &one,
                (const float *) matB->data, !transposeB ? m : k,
                (const float *) matA->data, !transposeA ? k : n,
                &zero,
                (float *) matC->data, m
    );
    return 0;
}

__global__ void relu_kernel(const float *input, float *output, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        float element = input[index];
        if (element <= 0) {
            output[index] = 0;
        } else {
            output[index] = element;
        }
    }
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < input->ndim; i++) {
        n *= input->shape[i];
    }

    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    relu_kernel << < num_blocks, threads_per_block >> > (input_data, output_data, n);
    return 0;
}

__global__ void relu_gradient_kernel(const float *input, float *output,
                                     const float *in_grad, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        float element = input[index];
        if (element <= 0) {
            output[index] = 0;
        } else {
            output[index] = in_grad[index];
        }
    }
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
    /* TODO: Your code here */
    int n = 1;
    for (int i = 0; i < input->ndim; i++) {
        n *= input->shape[i];
    }

    const float *input_data = (const float *) input->data;
    float *output_data = (float *) output->data;
    const float *in_grad_data = (const float *) in_grad->data;
    int threads_per_block = 1024;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    relu_gradient_kernel << < num_blocks, threads_per_block >> > (input_data,
            output_data, in_grad_data, n);
    return 0;
}

__global__ void softmax_kernel(int64_t nrow, int64_t ncol,
                               const float *input_data,
                               float *output_data) {

// two dimensional thread blocks.
    int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (y >= nrow) {
        return;
    }
    // y_th row of input data
    input_data += y * ncol;
    output_data += y * ncol;
    // find max for a row.
    float maxval = *input_data;
    for (int x = 1; x < ncol; ++x) {
        maxval = max(maxval, input_data[x]);
    }
    // Deduct by max for a row, and raise to exp.
    // in case of too large of exp, and the result will not be affected
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
        sum += exp(input_data[x] - maxval);
    }
    // Compute per-row softmax.
    for (int x = 0; x < ncol; ++x) {
        output_data[x] = exp(input_data[x] - maxval) / sum;
    }
}


int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
    /* TODO: Your code here */
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    int64_t nrow = input->shape[0];
    int64_t ncol = input->shape[1];
    float *input_data = (float *) input->data;
    float *output_data = (float *) output->data;
    dim3 threads;
    if (nrow < 1024) {
        threads.x = nrow;
    } else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    softmax_kernel << < 1, threads >> > (nrow, ncol, input_data, output_data);
    return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b, DLArrayHandle output) {
    assert(input_a->ndim == 2);
    assert(input_b->ndim == 2);
    assert(output->ndim == 1);
    assert(
            input_a->shape[0] == input_b->shape[0]
            && input_a->shape[1] == input_b->shape[1]);
    int nrow = input_a->shape[0];
    // Maximum x- or y-dimension of a block = 1024
    // But we need 'nrow' shared memory, and max shared memory is 48KB.
    // Conservatively allow max 16KB shared memory.
    assert(nrow <= 1024 * 4);
    int ncol = input_a->shape[1];
    const float *input_data_a = (const float *) input_a->data;
    const float *input_data_b = (const float *) input_b->data;
    float *output_data = (float *) output->data;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
    } else {
        threads.x = 1024;
        threads.y = (nrow + 1023) / 1024;
    }
    // 1 block, each block with 'threads' number of threads with 'nrow' shared
    // memory size
    matrix_softmax_cross_entropy_kernel << < 1, threads, nrow * sizeof(float) >> > (
            nrow, ncol, input_data_a, input_data_b, output_data);
    return 0;
}
==========================================================
            runtime_base.h            
==========================================================
/*!
 *  Copyright (c) 2017 by Contributors
 * \file runtime_base.h
 * \brief Base of all C APIs
 */
#ifndef DLSYS_RUNTIME_RUNTIME_BASE_H_
#define DLSYS_RUNTIME_RUNTIME_BASE_H_

#include "c_runtime_api.h"
#include <stdexcept>

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*!
 * \brief every function starts with API_BEGIN(), and finishes with API_END()
 *  or API_END_HANDLE_ERROR
 */
#define API_END()                                                              \
  }                                                                            \
  catch (std::runtime_error & _except_) {                                      \
    return DLSYSAPIHandleException(_except_);                                  \
  }                                                                            \
  return 0;

/*!
 * \brief every function starts with API_BEGIN() and finishes with API_END() or
 * API_END_HANDLE_ERROR. The finally clause contains procedure to cleanup states
 * when an error happens.
 */
#define API_END_HANDLE_ERROR(Finalize)                                         \
  }                                                                            \
  catch (std::runtime_error & _except_) {                                      \
    Finalize;                                                                  \
    return DLSYSAPIHandleException(_except_);                                  \
  }                                                                            \
  return 0;

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
inline int DLSYSAPIHandleException(const std::runtime_error &e) {
    // TODO
    // TVMAPISetLastError(e.what());
    return -1;
}

#endif // DLSYS_RUNTIME_RUNTIME_BASE_H_
