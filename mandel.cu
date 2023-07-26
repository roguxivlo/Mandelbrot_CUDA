#ifndef NO_FREETYPE
#define NO_FREETYPE
#endif

#include <cuda_runtime_api.h>
#include "Mandelbrot.h"
#include <math.h>
#include <chrono>
#include <iostream>
#include <pngwriter.h>
#include <string>
#include <fstream>

using namespace std;

const int min_iterations = 11;

// Kernel threads config:
const int _1D_config_size = 6;
int _1D_config[_1D_config_size];

const int _2D_config_256_size = 9;
std::pair<int, int> _2D_config_256[_2D_config_256_size];

const int _2D_config_1024_size = 11;
std::pair<int, int> _2D_config_1024[_2D_config_1024_size];

const int _2D_config_other_size = 7;
std::pair<int, int> _2D_config_other[_2D_config_other_size] =
    {
        {32, 32},
        {16, 16},
        {8, 8},
        {32, 16},
        {64, 8},
        {8, 64},
        {16, 32}
    };

void config_init() {
    for (int i = 0; i < _1D_config_size; ++i) {
        _1D_config[i] = 1 << (i + 5);
    }

    for (int i = 0; i < _2D_config_256_size; ++i) {
        _2D_config_256[i] = {1 << (_2D_config_256_size - i - 1), 1 << i};
    }

    for (int i = 0; i < _2D_config_1024_size; ++i) {
        _2D_config_1024[i] = {1 << (_2D_config_1024_size - i - 1), 1 << i};
    }
}


// Resolution config:
const int gpu_res = 10000;
const long long gpu_size = gpu_res * gpu_res;
const int cpu_res = 1000;
const long long cpu_size = cpu_res * cpu_res;
const real cpu_time_factor = 100;


// Picture coordinates:
const std::pair<const real, const real> LD = {-2.0, -1.25};
const std::pair<const real, const real> RT = {0.5, 1.25};

// Colors config:
const int iters = 256;


// Define Kernel:
__global__ void compute_mandel(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER, int *Mandel_gpu, int Mandel_size) {
    int i;
    real x, y, Zx = 0., Zy = 0., tZx = 0.;
    real dx = (X1 - X0) / (POZ - 1);
    real dy = (Y1 - Y0) / (PION - 1);

    // Check if block is 2D:
    if (blockDim.y > 1) {
        // 2D
        i = POZ * blockIdx.y * blockDim.y;
        i += POZ * threadIdx.y;
        i += blockIdx.x * blockDim.x + threadIdx.x;

        real x = X0 + (i % POZ) * dx;
        real y = Y1 - (i / POZ) * dy;
        real Zy = 0, Zx = 0, tZx = 0;
        if (i >= gpu_size) return;

        int j;
        for (j = 0; j < ITER && Zy * Zy + Zx * Zx < 4; ++j) {
            tZx = Zx;
            Zx = tZx * tZx - Zy * Zy + x;
            Zy = 2 * tZx * Zy + y;
        }
        Mandel_gpu[i] = j;
    }
    else {
        // 1D
        i = blockDim.x * blockIdx.x + threadIdx.x;
        x = X0 + (i % POZ) * dx;
        y = Y0 + (i / POZ) * dy;
        int j;

        if (i >= gpu_size) return;

        for (j = 0; j < ITER && Zy * Zy + Zx * Zx < 4; ++j) {
            tZx = Zx;
            Zx = tZx * tZx - Zy * Zy + x;
            Zy = 2 * tZx * Zy + y;
        }
        Mandel_gpu[i] = j;
    }
}

void mandel_cpu(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER,int *Mandel, int SIZE){
    double    dX=(X1-X0)/(POZ-1);
    double    dY=(Y1-Y0)/(PION-1);
    double x,y,Zx,Zy,tZx;
    int i;
    
    for (i = 0; i < SIZE; ++i) {
        x = X0 + dX * (i % POZ);
        y = Y0 + dY * (i / PION);
        Zx = Zy = tZx = 0;
        int j;
        for (j = 0; j < ITER && Zx * Zx + Zy * Zy < 4; ++j) {
            // compute z_n
            // z_{n+1} = z_n ^2 +(x,y);
            tZx = Zx;
            Zx = tZx * tZx - Zy * Zy + x;
            Zy = 2 * tZx * Zy + y;
        }
        Mandel[i] = j;
    }
}

std::chrono::duration<double> cpu_run(bool png) {
    int *m_test = (int *)malloc(sizeof(int) * cpu_res * cpu_res);
    auto start = std::chrono::steady_clock::now();
    mandel_cpu(LD.first, LD.second, RT.first, RT.second, cpu_res, cpu_res, iters, m_test, cpu_size);
    if (png) makePicturePNG(m_test, cpu_res, cpu_res, iters, 0);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    free(m_test);
    return diff;
}

std::chrono::duration<double> gpu_run(int blockDimX, int blockDimY, bool png, int png_id = -1) {
    dim3 threads(blockDimX, blockDimY);
    int gridDimX = (blockDimY == 1) ? gpu_size / blockDimX + 1 : gpu_res / blockDimX + 1;
    int gridDimY = (blockDimY == 1) ? 1 : gpu_res / blockDimY + 1;
    dim3 blocks(gridDimX, gridDimY);
    // std::cout << "run a kernel with " << blocks.x << " ";
    // std::cout << blocks.y << " blocks, " << threads.x << " ";
    // std::cout << threads.y << " threads per block\n";
    
    cudaError_t status;
    int *cuda_test;
    status = cudaMalloc((void**)&cuda_test, sizeof(int) * gpu_size);
    if (status != cudaSuccess) {
        std::cout << cudaGetErrorString(status);
    }

    auto start = chrono::steady_clock::now();

    compute_mandel<<<blocks, threads>>>(LD.first, LD.second, RT.first, RT.second, gpu_res, gpu_res, iters, cuda_test, gpu_size);
    
    if (png) {
        int *cuda_test_copy = (int *)malloc(sizeof(int) * gpu_size);
        status = cudaMemcpy(cuda_test_copy, cuda_test, sizeof(int) * gpu_size, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            std::cout << cudaGetErrorString(status);
        }
        makePicturePNG(cuda_test_copy, gpu_res, gpu_res, iters, png_id);
        free(cuda_test_copy);
    }
    
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status);
    }

    auto end = chrono::steady_clock::now();

    status = cudaFree(cuda_test);
    if (status != cudaSuccess) {
        std::cout << cudaGetErrorString(status);
    }

    return end - start;
}

void print_time(std::chrono::duration<double> time) {
    cout << chrono::duration <double, milli> (time).count() << " ms" << endl;
}

void run_all() {
    // cpu
    cout << "cpu;";
    for (int i = 0; i < min_iterations; ++i) {
        auto time = cpu_run(0);
        cout << chrono::duration <double, milli> (time).count() * cpu_time_factor;
        if (i + 1 < min_iterations) cout << "; ";
        else cout << "\n";
    }

    // 1D
    cout << "1D;";
    for (int i = 0; i < _1D_config_size; ++i) {
        cout << "config " << i << ";";
        for (int j = 0; j < min_iterations; ++j) {
            auto time = gpu_run(_1D_config[i], 1, 0);
            cout << chrono::duration <double, milli> (time).count();
            if (j + 1 < min_iterations) cout << "; ";
            else cout << "\n";
        }
    }

    // cpu
    cout << "cpu;";
    for (int i = 0; i < min_iterations; ++i) {
        auto time = cpu_run(0);
        cout << chrono::duration <double, milli> (time).count() * cpu_time_factor;
        if (i + 1 < min_iterations) cout << "; ";
        else cout << "\n";
    }

    // 2D_256
    cout << "2D_256;";
    for (int i = 0; i < _2D_config_256_size; ++i) {
        cout << "config " << i << ";";
        for (int j = 0; j < min_iterations; ++j) {
            auto time = gpu_run(_2D_config_256[i].first, _2D_config_256[i].second, 0);
            cout << chrono::duration <double, milli> (time).count();
            if (j + 1 < min_iterations) cout << "; ";
            else cout << "\n";
        }
    }

    // cpu
    cout << "cpu;";
    for (int i = 0; i < min_iterations; ++i) {
        auto time = cpu_run(0);
        cout << chrono::duration <double, milli> (time).count() * cpu_time_factor;
        if (i + 1 < min_iterations) cout << "; ";
        else cout << "\n";
    }

    // 2D_1024
    cout << "2D_1024;";
    for (int i = 0; i < _2D_config_1024_size; ++i) {
        cout << "config " << i << ";";
        for (int j = 0; j < min_iterations; ++j) {
            auto time = gpu_run(_2D_config_1024[i].first, _2D_config_1024[i].second, 0);
            cout << chrono::duration <double, milli> (time).count();
            if (j + 1 < min_iterations) cout << "; ";
            else cout << "\n";
        }
    }

    // cpu
    cout << "cpu;";
    for (int i = 0; i < min_iterations; ++i) {
        auto time = cpu_run(0);
        cout << chrono::duration <double, milli> (time).count() * cpu_time_factor;
        if (i + 1 < min_iterations) cout << "; ";
        else cout << "\n";
    }

    // 2D_other
    cout << "2D_other;";
    for (int i = 0; i < _2D_config_other_size; ++i) {
        cout << "config " << i << ";";
        for (int j = 0; j < min_iterations; ++j) {
            auto time = gpu_run(_2D_config_other[i].first, _2D_config_other[i].second, 0);
            cout << chrono::duration <double, milli> (time).count();
            if (j + 1 < min_iterations) cout << "; ";
            else cout << "\n";
        }
    }
}

int main() {
    config_init();
    run_all();
    // gpu_run(_2D_config_1024[4].first, _2D_config_1024[4].second, 1, 1);
}

void makePicturePNG(int *Mandel,int width, int height, int MAX, int id){
    double red_value, green_value, blue_value;
    float scale = 256.0/MAX;
    double MyPalette[41][3]={
        {1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0},// 0, 1, 2, 3, 
        {1.0,1.0,1.0},{1.0,0.7,1.0},{1.0,0.7,1.0},{1.0,0.7,1.0},// 4, 5, 6, 7,
        {0.97,0.5,0.94},{0.97,0.5,0.94},{0.94,0.25,0.88},{0.94,0.25,0.88},//8, 9, 10, 11,
        {0.91,0.12,0.81},{0.88,0.06,0.75},{0.85,0.03,0.69},{0.82,0.015,0.63},//12, 13, 14, 15, 
        {0.78,0.008,0.56},{0.75,0.004,0.50},{0.72,0.0,0.44},{0.69,0.0,0.37},//16, 17, 18, 19,
        {0.66,0.0,0.31},{0.63,0.0,0.25},{0.60,0.0,0.19},{0.56,0.0,0.13},//20, 21, 22, 23,
        {0.53,0.0,0.06},{0.5,0.0,0.0},{0.47,0.06,0.0},{0.44,0.12,0},//24, 25, 26, 27, 
        {0.41,0.18,0.0},{0.38,0.25,0.0},{0.35,0.31,0.0},{0.31,0.38,0.0},//28, 29, 30, 31,
        {0.28,0.44,0.0},{0.25,0.50,0.0},{0.22,0.56,0.0},{0.19,0.63,0.0},//32, 33, 34, 35,
        {0.16,0.69,0.0},{0.13,0.75,0.0},{0.06,0.88,0.0},{0.03,0.94,0.0},//36, 37, 38, 39,
        {0.0,0.0,0.0}//40
        };

    auto name = "Mandelbrot" + std::to_string(id);
    name.append(".png");
    pngwriter png(width,height,1.0,name.c_str());
    for (int j=height-1; j>=0; j--) {
        for (int i=0; i<width; i++) {
            // compute index to the palette
            int indx= (int) floor(5.0*scale*log2f(1.0f*Mandel[j*width+i]+1));
            red_value=MyPalette[indx][0];
            green_value=MyPalette[indx][2];
            blue_value=MyPalette[indx][1];
            png.plot(i,j, red_value, green_value, blue_value);            
        }
    }
    png.close();
}