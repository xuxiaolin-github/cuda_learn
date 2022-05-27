#include<opencv2/opencv.hpp>
#include<iostream>
#include <thread>
// #include "circuqueue.hpp"
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include "utils.h"

static const int INPUT_H = 640;
static const int INPUT_W = 640;

static float data[3 * INPUT_H * INPUT_W];

__global__ void convert(uchar *mat ,float* data){

    
    uint ix=threadIdx.x+blockDim.x*blockIdx.x;
    uint iy=threadIdx.y+blockDim.y*blockIdx.y;

    int i=ix+iy*(gridDim.x*blockDim.x);

    data[i] = mat[i+2]/ 255.0;
    data[i + INPUT_H * INPUT_W] = mat[i+1] / 255.0;
    data[i + 2 * INPUT_H * INPUT_W] = mat[i+0] / 255.0;
    

}

void inline gpu_convert(uchar *d_src ,float*d_dst,cv::Mat &frame){
	cudaMemcpy(d_src,frame.data,3 * INPUT_H * INPUT_W,cudaMemcpyHostToDevice);
	dim3 grid1(20,20);
	dim3 block(32,32);
	convert<<<grid1,block>>>(d_src,d_dst);
	cudaDeviceSynchronize();
}


int main(int argc, char** argv) {
  
	// 打开文件
	cv::VideoCapture capture;
	capture.open("/home/xu/视频/01.avi");
	if (!capture.isOpened()) {
		printf("could not read this video file...\n");
		return -1;
	}
    cv::Mat frame;
	// cv::cuda::GpuMat dst,src;
	// src.upload(frame);
	// unsigned char *array=new unsigned char[3 * INPUT_H * INPUT_W];
	
	uchar* d_src = NULL;
    // uchar3* d_dst = NULL;
	float *d_dst=NULL;
    // cudaMalloc((void**)&d_src,3 * INPUT_H * INPUT_W);
	// cudaMalloc((void**)&d_dst,3 * INPUT_H * INPUT_W*sizeof(float));

    auto start = std::chrono::system_clock::now();
    while (true) {

		capture>>frame;
		if (frame.empty())
		{
			break;
		}


		auto cap_end = std::chrono::system_clock::now();
		

		

		// cv::imshow("01",frame);
		// cv::waitKey(1);
		gpu_convert(d_src ,d_dst,frame);
        


	


		

		auto preprocess = std::chrono::system_clock::now();
		std::cout <<"process: "<<std::setw(2)<< std::chrono::duration_cast<std::chrono::milliseconds>(preprocess-cap_end).count() << "ms  "<<std::endl ;

            
    }
	auto end = std::chrono::system_clock::now();
    std::cout <<"time: "<<std::setw(2)<< std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms  " ;
}