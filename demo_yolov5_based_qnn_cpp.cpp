#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "qnncontext.hpp"
#include "yolov5.hpp"


//-------------------------------------------------------------------------------------------------
// main
//-------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	if (!qnn::tools::helper::config(2, "None", 0)) {
		std::cerr << "Error: failed to config QNN context" << std::endl;
		return -1;
	}

	std::vector<std::string> labels;
	std::ifstream file(R"(.\models\coco.names)");
	if (file && file.is_open()) {
		std::string line;
		while (std::getline(file, line)) labels.push_back(line);
		file.close();
	}
	else {
		QNN_INF("failed to read labels from .\\models\\coco.names");
		return -1;
	}

	qnn::algo::yolov5::Yolov5 yolov5(R"(.\models\yolov5l_modified_quantized_8_8_8bit.bin)", 
		labels, 0.65f, 0.50f, R"(.\qnn)", "Htp", false);
	if (!yolov5.check()) {
		return -1;
	}

	cv::Mat image = cv::imread(R"(.\images\bus.jpg)");
	std::vector<qnn::algo::yolov5::Object> objects = yolov5.inference({ image }, "burst");
	qnn::algo::yolov5::draw_objects(image, objects);
	cv::imshow(R"(IMAGE: .\images\bus.jpg   )", image);

	image   = cv::imread(R"(.\images\zidane.jpg)");
	objects = yolov5.inference({ image }, "burst");
	qnn::algo::yolov5::draw_objects(image, objects);
	cv::imshow(R"(IMAGE: .\images\zidane.jpg)", image);

	QNN_WAR("press the 'q' key to exit, the 'n' key to capture usb camera");
	while (true) {
		int key = cv::waitKey(1);
		if (key == 'n') {
			QNN_INF("broken by user pressed the 'n' key");
			break;
		}
		else if (key == 'q') {
			QNN_INF("exited by user pressed the 'q' key");
			QNN_INF("bye bye");
			return 0;
		}
	}

	cv::destroyAllWindows();

	cv::VideoCapture capture(0);
	if (!capture.isOpened()) {
		QNN_ERR("could not open USB camera 0");
		return -1;
	}

	QNN_INF("video image resolution = %.0fx%.0f", 
		capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));


	QNN_INF("start to capture usb camera 0 ...");

	while (true) {
		capture >> image;
		if (image.empty()) {
			QNN_ERR("could not captrue an image");
			break;
		}

		objects = yolov5.inference({ image }, "burst");
		qnn::algo::yolov5::draw_objects(image, objects);
		cv::imshow(R"(USB CAMERA 0)", image);

		if (cv::waitKey(1) == 'q') {
			QNN_INF("exited by user pressed the 'q' key");
			break;
		}
	}

	capture.release();

	cv::destroyAllWindows();
	QNN_INF("bye bye");
	return 0;
}
