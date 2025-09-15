#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;
using namespace std::chrono;


__global__ 
void gaussian_blur_shared(uchar* img_g, float* kernel_g, uchar* output, int img_rows, int img_cols, float kernelSum) 
{
    const int TILE_SIZE = 5;
    __shared__ uchar tile[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;


    // central pixel
    tile[ty+1][tx+1] = img_g[gy*img_cols + gx];

    // Only load halo if the global coordinate is valid
    if (gx-1 >= 0 && tx == 0) tile[ty+1][0] = img_g[gy*img_cols + gx-1];
    if (gx+1 < img_cols && tx == blockDim.x-1) tile[ty+1][TILE_SIZE-1] = img_g[gy*img_cols + gx+1];
    if (gy-1 >= 0 && ty == 0) tile[0][tx+1] = img_g[(gy-1)*img_cols + gx];
    if (gy+1 < img_rows && ty == blockDim.y-1) tile[TILE_SIZE-1][tx+1] = img_g[(gy+1)*img_cols + gx];

    __syncthreads();
    if (gx >= img_cols || gy >= img_rows) return;
    int t = gy * img_cols + gx;

    float sum = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int nx = ty + dy + 1, ny = tx + dx + 1;
            int pixel = tile[nx][ny];
            sum += pixel * kernel_g[(dy+1)*3 + (dx+1)];
        }
    }

    if (gx > 0 && gx < img_cols-1 && gy > 0 && gy < img_rows-1) {
        int t_out = (gy-1)*(img_cols-2) + (gx-1);
        output[t_out] = (uchar)(sum/kernelSum);
    }
}



int main() {
    // load image
    Mat img = imread("img.jpeg", IMREAD_GRAYSCALE);

    if (img.empty()) {
        cout << "Empty image!" << endl;
        return -1;
    }

    // convolutional kernel of size 3x3
    float kernel[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    float kernelSum = 16.0f;

    // make padded image to take care of edges
    Mat padded;
    copyMakeBorder(img, padded, 1, 1, 1, 1, BORDER_REPLICATE);
    int img_cols = padded.cols;

    // output image pointer initialized to 0
    Mat temp = Mat::zeros(img.size(), CV_8UC1);

    // pointer to padded image and output
    uchar* paddedPtr = padded.data;
    uchar* outputPtr = temp.data;
    
    // cuda pointers to image, kernel, and image output
    uchar* img_g;
    float* kernel_g;
    uchar* img_output;

    // getting size of image, padded image, and kernel
    size_t imgBytes = img.total()*img.elemSize();
    size_t paddedBytes = padded.total()*padded.elemSize();
    size_t kernelBytes = 9*sizeof(float);

    // allocating memory in vram for image, kernel and output
    cudaMalloc(&img_g, paddedBytes);
    cudaMalloc(&kernel_g, kernelBytes);
    cudaMalloc(&img_output, imgBytes);

    // copying padded image, kernel to vram and setting output to 0
    cudaMemcpy(img_g, paddedPtr, paddedBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_g, kernel, kernelBytes, cudaMemcpyHostToDevice);
    cudaMemset(img_output, 0, imgBytes);


    // calling kernel function
    dim3 block(3,3);
    dim3 grid((img_cols + block.x - 1) / block.x, (img_rows + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // start timing
    gaussian_blur_shared<<<grid, block>>>(img_g, kernel_g, img_output, img_rows, img_cols, kernelSum);

    cudaEventRecord(stop);  // stop timing
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Time taken: " << milliseconds << " ms" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(outputPtr, img_output, imgBytes, cudaMemcpyDeviceToHost);

    cudaFree(img_g);
    cudaFree(kernel_g);
    cudaFree(img_output);

    imwrite("g1.png", temp);

    cout << "saved image" << endl;

    return 0;

}