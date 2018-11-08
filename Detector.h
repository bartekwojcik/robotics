//
//Author: Bartosz Wójcik unsless stated otherwise
//
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <stdio.h> 
#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/dev/all.h>
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::dev;
using namespace cv;

class Detector
{
public:
	Detector();
	~Detector();
	void detectCircle(cv::Mat frame);
	void detectFace(Mat frame);
	Mat applyLinearFilter(ImageOf<PixelRgb>* yarp_img);
	Mat edgeDetectionFilter(ImageOf<PixelRgb>* yarp_img);
	Mat cannyEdgefilter(ImageOf<PixelRgb>* yarp_img);
};

