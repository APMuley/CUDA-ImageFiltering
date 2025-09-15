#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace cv;
using namespace std;
using namespace std::chrono;


__global__ 
void gaussian_blur_vram(uchar* img_g, float* kernel_g, uchar* output, int img_rows, int img_cols, float kernelSum) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int t = (y-1) * img_cols + (x-1);

    if (x > img_cols || y > img_rows) return;

    float sum = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int nx = x + dx, ny = y + dy;
            int pixel = img_g[ny*(img_cols+2) + nx];
            sum += pixel * kernel_g[(dy+1)*3 + (dx+1)];
        }
    }
    output[t] = (uchar) (sum / kernelSum);
}



int main() {
    // load image
    Mat img = imread("img.jpeg", IMREAD_GRAYSCALE);
    int img_rows = img.rows;
    int img_cols = img.cols;

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
    dim3 block(16,16);
    dim3 grid((img_cols + block.x - 1) / block.x, (img_rows + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // start timing
    gaussian_blur<<<grid, block>>>(img_g, kernel_g, img_output, img_rows, img_cols, kernelSum);

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