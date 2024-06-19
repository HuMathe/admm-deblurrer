#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>

namespace py = pybind11;

template <typename Tp>
static py::array_t<Tp> 
conv2d
( py::array_t<Tp> conv_src, py::array_t<Tp> conv_kernel )
{
    py::buffer_info src_info = conv_src.request(),\
                    kernel_info = conv_kernel.request();
    int nrows = src_info.shape[0], ncols = src_info.shape[1];
    int krows = kernel_info.shape[0], kcols = kernel_info.shape[1];
    int rows = nrows - krows + 1, cols = ncols - kcols + 1;
    int ksize = krows * kcols;

    py::array_t<Tp> conv_result({rows, cols});

    Tp* src_base_ptr = static_cast<Tp*>(src_info.ptr);
    Tp* kernel_base_ptr = static_cast<Tp*>(kernel_info.ptr);
    Tp* dst_base_ptr = static_cast<Tp*>(conv_result.request().ptr);

    std::vector<int> koffsets ( krows * kcols );
    std::vector<Tp> kcoeffs ( kernel_base_ptr, kernel_base_ptr + krows * kcols );
    

    for ( int y = 0; y < krows; y++ )
    {
        for ( int x = 0; x < kcols; x++ )
        {
            koffsets[y * kcols + x] = y * ncols + x;
        }
    }
    
    for( int y = 0; y < rows; y++ )
    {
        Tp* src_ptr = src_base_ptr + y * ncols;
        Tp * dst_ptr = dst_base_ptr + y * cols;

        for ( int x = 0; x < cols; x++ )
        {
            Tp s = 0; 
            for( int i = 0; i < ksize; i++ )
                s += src_ptr[x + koffsets[i]] * kcoeffs[i];
            dst_ptr[x] = s;
        }
    }

    return conv_result;

}

PYBIND11_MODULE(simple_conv2d, m) 
{
    m.def("conv2d_fp64", &conv2d<double>, "naive 2d convolusion (float 64)");
    m.def("conv2d_fp32", &conv2d<float>, "naive 2d convolusion (float 32)");
}