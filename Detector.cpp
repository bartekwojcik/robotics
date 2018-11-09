//
// Author: Bartosz W�jcik unless stated otherwise 
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
	}
	//save to file for our convenience 
	imwrite("detectedFace.jpg", src);


}

//example circle detection from https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html
void Detector::detectCircle(Mat frame)
{
	//camera input is of low quality and it dones work unless circle is OBVIOUS
	//:(
	Mat src = imread("BlueCircle.png", 1);
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

	imwrite("detectedCircle.jpg", src);
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
	Mat src = copy.clone();
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
	imwrite("wikipediaEdgeDetectionWithKernel.jpg", dst);
	return dst;

}
//https://en.wikipedia.org/wiki/Canny_edge_detector
//https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
Mat Detector::cannyEdgefilter(ImageOf<PixelRgb>* yarp_img)
{
	Mat gray, edge, dst;
	Mat copy = cv::cvarrToMat(static_cast<IplImage*>(yarp_img->getIplImage()));
	Mat src = copy.clone();
	cvtColor(src, gray, CV_BGR2GRAY);
	/// Reduce noise with a kernel 3x3
	blur(gray, edge, Size(3, 3));

	/// Canny detector
	Canny(edge, edge, 50, 150, 3);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	src.copyTo(dst, edge);
	imwrite("cannyEdgeDetection.jpg", dst);
	return dst;
}

//function that uses simple filter, example from https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/filter_2d/filter_2d.html#filter-2d
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
	imwrite("exampleEdgeDetection.jpg", dst);
	return dst;
}

//sobel filter funcion
//sobel derivatives are used for edge detection
//using some scary maths they are convolving image with some small filter in both directions
//which, i believe, leads to emphasising edges.
//https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
Mat Detector::sobelFilter(ImageOf<PixelRgb>* yarp_img)
{
	int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
	Mat src, copy,src_gray, grad;
	copy = cv::cvarrToMat(static_cast<IplImage*>(yarp_img->getIplImage()));
	src = copy.clone();
	//gausian blur to reduce noise 
	GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// Convert it to gray
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  imwrite("sobelFilterEdgeDetection.jpg", grad);
  return grad;
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
