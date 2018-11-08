//
// Author: Bartosz Wójcik unless stated otherwise 
//
#include "Detector.h"
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

CascadeClassifier face_cascade;
String face_cascade_name = "haarcascade_frontalface_alt.xml";
RNG rng(12345);

Detector::Detector()
{
	printf("constructor");
	if (!face_cascade.load(face_cascade_name)) 
	{
		printf("--(!)Error loading\n");
	};
}


Detector::~Detector()
{
}

//function that detects face and draws an circle around it so we know what the face was actually detected
// heavily inspired by https://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
void Detector::detectFace(Mat frame)
{
	Mat src = frame.clone();
	std::vector<Rect> faces;
	Mat frame_gray;

	//convert colors to grayscale
	cvtColor(src, frame_gray, CV_BGR2GRAY);
	//this equalizes the histogram of a grayscale image (makes it easier to avoid false-positives)
	equalizeHist(frame_gray, frame_gray);

	//detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	//loop over detected faces
	for (size_t i = 0; i < faces.size(); i++)
	{
		//find a faces center
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		//draw an elipse
		ellipse(src, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 255, 0), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);
	}
	//save to file for our convenience 
	imwrite("face.jpg", frame);


}

//example circle detection from https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html
void Detector::detectCircle(Mat frame)
{

	Mat src = frame.clone();
	Mat  src_gray;

	/// Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

	vector<Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 100, 0, 0);

	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);		
		// circle outline
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}

	imwrite("circle.jpg", src);
}

//https://en.wikipedia.org/wiki/Kernel_(image_processing)
//applies edge detection filter on image
Mat Detector::edgeDetectionFilter(ImageOf<PixelRgb>* yarp_img)
{
	Mat dst;
	Point anchor = Point(-1, -1);
	int delta = 0;
	int ddepth = -1;
	//copy data from yarp image
	Mat copy = cv::cvarrToMat(static_cast<IplImage*>(yarp_img->getIplImage()));	
	Mat src = copy.src();
	//mask for edge detection
	float mask[9] = { -1,-1,-1, -1, 8, -1, -1,-1,-1 };
	Mat kernel(3, 3, CV_32FC1);
	//copy mask to kernel
	memcpy(kernel.data, mask, sizeof(float) * 9);
	
	//src - source image
	//dst - destination image
	//ddepth - depth of destination image, -1 means it will be the sam as sourc.depth()
	//depth means how many bits is used to indicate colour
	//anchor: relative position of a filtered point wihin the kernal, -1-1 means anchor is in the center
	//delta, some option value, i dont get this :/
	//pixel interpolation value
	filter2D(src, dst, ddepth , kernel, anchor , delta, BORDER_DEFAULT);
	return dst;

}
//https://en.wikipedia.org/wiki/Canny_edge_detector
//https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
Mat Detector::cannyEdgefilter(ImageOf<PixelRgb>* yarp_img)
{
	Mat gray, edge, dst, src;
	Mat copy = cv::cvarrToMat(static_cast<IplImage*>(yarp_img->getIplImage()));
	Mat src = copy.clone();
	//convert to grayscale
	cvtColor(src, gray, CV_BGR2GRAY);
	//use canny filter
	//50 and 150 are thresholds for hystesis procedure
	//3 - some size for sobel, dont know exactly 
	Canny(gray, edge, 40, 170, 3);


	edge.convertTo(dst, CV_8U);
	return dst;
}

//function that uses standard, example from https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/filter_2d/filter_2d.html#filter-2d
Mat Detector::applyLinearFilter(ImageOf<PixelRgb>* yarp_img)
{
	Mat src, dst, copy;

	Mat kernel;
	Point anchor;
	double delta;
	int ddepth;
	int kernel_size;

	int c;

	//convert from yarp image to open cv mat class
	copy = cv::cvarrToMat(static_cast<IplImage*>(yarp_img->getIplImage()));
	src = copy.clone();
	/// Initialize arguments for the filter
	anchor = Point(-1, -1);
	delta = 0;
	ddepth = -1;

	// Update kernel size for a normalized box filter
	//kernel of size 5 and with ones
	kernel_size = 5;
	//src - source image
	//dst - destination image
	//ddepth - depth of destination image, -1 means it will be the sam as sourc.depth()
	//depth means how many bits is used to indicate colour
	//anchor: relative position of a filtered point wihin the kernal, -1-1 means anchor is in the center
	//delta, some option value, i dont get this :/
	//pixel interpolation value
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);

	/// Apply filter
	filter2D(src, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT);
	//imwrite( "src.jpg", src );
	//imwrite( "dst.jpg", dst );
	return dst;
}

//we could not make it compile, there is problem with aruco installaytion i suppose
//void detect_markers(Mat inputImge)
//{
//	vector< int > markerIds;
//	vector< vector<Point2f> > markerCorners, rejectedCandidates;
//	vector< Vec3d > rvecs, tvecs;
//
//	cv::Ptr<aruco::DetectorParameters> parameters;
//	cv::Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
//
//	cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);	
//}
