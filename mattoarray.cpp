#include<opencv2/opencv.hpp>
#include<iostream>
#include <thread>
#include <cuda_runtime.h>
#include "utils.h"

static const int INPUT_H = 640;
static const int INPUT_W = 640;

float data[3 * INPUT_H * INPUT_W];


void inline cpu_convert(cv::Mat &pr_img){

	for (int i = 0; i < INPUT_H * INPUT_W; i++) {
		data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
		data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
		data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
	}
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

	float *d_dst=NULL;
    
	cudaMalloc((void**)&d_dst,3 * INPUT_H * INPUT_W*sizeof(float));
	
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
		
        cpu_convert(frame);

		
		cudaMemcpy(d_dst,data,3 * INPUT_H * INPUT_W*sizeof(float),cudaMemcpyHostToDevice);

	


		

		auto preprocess = std::chrono::system_clock::now();
		std::cout <<"process: "<<std::setw(2)<< std::chrono::duration_cast<std::chrono::milliseconds>(preprocess-cap_end).count() << "ms  "<<std::endl ;

            
    }
	auto end = std::chrono::system_clock::now();
    std::cout <<"time: "<<std::setw(2)<< std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms  " ;
}