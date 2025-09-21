#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cmath> // For M_PI, sin, etc.
#include <cfloat> // For FLT_MAX

namespace py = pybind11;

// I'm getting an error about the atomicMin function on floats so
// this function uses a Compare-and-Swap loop to safely
// perform a minimum operation on floating-point numbers.

__device__ static float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) < val) {
            break;
        }
        old = atomicCAS(address_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void projection_kernel(const float* points, int num_points, 
                                float* range_buffer, int* index_buffer, 
                                int image_height, int image_width,
                                float fov_up, float fov_down) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_points) {
        float x = points[index * 3 + 0];
        float y = points[index * 3 + 1];
        float z = points[index * 3 + 2];

        float r = sqrtf(x * x + y * y + z * z);
        if (r < 1e-6) return;

        float theta = atan2f(y, x);
        float phi = asinf(z / r);

        float vertical_fov = fov_up - fov_down;
        float u = (image_width - 1) * (1.0f - (theta + M_PI) / (2.0f * M_PI));
        float v = (image_height - 1) * (1.0f - (phi - fov_down) / vertical_fov);

        int u_int = static_cast<int>(roundf(u));
        int v_int = static_cast<int>(roundf(v));
        if (u_int < 0 || u_int >= image_width || v_int < 0 || v_int >= image_height) return;

        int pixel_index = v_int * image_width + u_int;
        
        float old_range = atomicMinFloat(&range_buffer[pixel_index], r);

        if (r < old_range) {
            index_buffer[pixel_index] = index;
        }
    }
}

__global__ void initialize_buffers_kernel(float* range_buffer, int* index_buffer, int num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        range_buffer[index] = FLT_MAX;
        index_buffer[index] = -1;
    }
}

py::array_t<int> spherical_projection_cuda(py::array_t<float, py::array::c_style | py::array::forcecast> points_py, 
                                           int image_height, int image_width, 
                                           float fov_up_deg, float fov_down_deg) {
    py::buffer_info points_buf = points_py.request();
    const float *points_ptr = (const float *)points_buf.ptr;
    int num_points = points_buf.shape[0];
    
    float *d_points, *d_range_buffer;
    int *d_index_buffer;
    size_t num_pixels = image_height * image_width;

    cudaMalloc(&d_points, num_points * 3 * sizeof(float));
    cudaMalloc(&d_range_buffer, num_pixels * sizeof(float));
    cudaMalloc(&d_index_buffer, num_pixels * sizeof(int));
    
    cudaMemcpy(d_points, points_ptr, num_points * 3 * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block_init = 256;
    int blocks_per_grid_init = (num_pixels + threads_per_block_init - 1) / threads_per_block_init;
    initialize_buffers_kernel<<<blocks_per_grid_init, threads_per_block_init>>>(d_range_buffer, d_index_buffer, num_pixels);
    
    int threads_per_block_proj = 256;
    int blocks_per_grid_proj = (num_points + threads_per_block_proj - 1) / threads_per_block_proj;
    
    float fov_up_rad = fov_up_deg * M_PI / 180.0f;
    float fov_down_rad = fov_down_deg * M_PI / 180.0f;
    
    projection_kernel<<<blocks_per_grid_proj, threads_per_block_proj>>>(d_points, num_points, d_range_buffer, d_index_buffer, image_height, image_width, fov_up_rad, fov_down_rad);
    
    auto result_py = py::array_t<int>(num_pixels);
    py::buffer_info result_buf = result_py.request();
    int *result_ptr = (int *)result_buf.ptr;
    cudaMemcpy(result_ptr, d_index_buffer, num_pixels * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_points);
    cudaFree(d_range_buffer);
    cudaFree(d_index_buffer);
    
    result_py.resize({image_height, image_width});
    return result_py;
}

PYBIND11_MODULE(projection_kernel, m) {
    m.def("spherical_projection", &spherical_projection_cuda, "CUDA-accelerated Spherical Projection");
}
