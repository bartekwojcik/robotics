//
//Author: Thomas Triffterer unless stated otherwise 
//

#include <stdio.h> 
#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/dev/all.h>
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::dev;
using namespace yarp::sig::draw;
#define RESULT_ERR -1
#define RESULT_OK 0
#include <stdio.h>
#include <stdio.h>
#include <unistd.h>
#include <yarp/os/Property.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "Detector.h"
using namespace cv;


//main function
int main(int argc, char *argv[]) {
	//
	// Author: Thomas Triffterer
	//
	printf("Start\n");
	Network yarp;
	//if there are no ports specified then stop
	if (argc <= 2) {
		printf("Please specifie the Input port\n");
		printf("Example: Imagereader /icubSim/cam /icubSim/cam/left\n");
		return RESULT_ERR;
	}
	//print arguments
	printf("ARGS=");
	printf(argv[1]);
	printf("\n");
	printf(argv[2]);
	printf("\n");

	//open screen port
	BufferedPort<ImageOf<PixelRgb> > screenPort;
	bool result = screenPort.open("/cw2/image/screenPort");
	if (result == false) {
		printf("Could not open Screen port\n");
		return RESULT_ERR;
	}

	//open image port
	BufferedPort<ImageOf<PixelRgb> >  imagePort;    //  make  a  port  for reading images
	printf("got here 1\n");
	result = imagePort.open("/cw2/image/rawIn");
	if (result == false) {
		printf("Could not open Image port\n");
		return RESULT_ERR;
	}

	//connect ports
	Network::connect(argv[1], "/icubSim/texture/screen");
	Network::connect(argv[2], imagePort.getName());
	printf("got here 2\n");

	//
	//Author: Bartosz Wójcik
	//
	Detector detector = Detector();
	BufferedPort<ImageOf<PixelRgb> > port;
	Property options;
	ConstString portName = options.check("name",Value("/worker")).asString();
	port.open(portName);
	//
	// End of Bartosz Wójcik's code
	//
	//infinte loop reading feed
	while (1) {

		// read an image
		ImageOf<PixelRgb> *image = imagePort.read();
		//some debug message to be sure that we got here and lack of content is because there is nothing to read
		printf("got here 3\n");

		// check we actually got something							
		if (image != NULL) {
			//check size of image
			printf("We got an image of size %dx%d\n", image->width(), image->height());
			//
			// Author: Bartosz Wójcik
			//
			//apply linear filter
			Mat filtered = detector.applyLinearFilter(image);
			Mat edgeFilter = detector.cannyEdgefilter(image);
			//print some details so we know that image is not empty
			printf("After appying filter:");
			printf("Is filter of ones empty: %d (0 is good, 1 is bad)\n", filtered.empty());
			printf("is empty: %d (0 is good, 1 is bad)\n", filtered.empty());
			
			ImageOf<PixelBgr> img& = port.prepare();
			img.setExternal(edgeFilter.data,edgeFilter.size[1],edgeFilter.size[0]); 
			port.write();
			
			//detect face
			detector.detectFace(filtered);

			////detect circle			
			detector.detectCircle(edgeFilter);
			//
			// End of Bartosz Wójcik's code
			//
		}
		else {
			printf("NULL IMAGE\n");
			return RESULT_ERR;
		}
	}


	return RESULT_OK;
}


