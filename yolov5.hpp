#pragma once

#include <opencv2/opencv.hpp>


namespace qnn {
namespace algo {
namespace yolov5 {


//-------------------------------------------------------------------------------------------------
// Object
//-------------------------------------------------------------------------------------------------
class Object {
public:
	Object(void) :
		m_box(cv::Rect(0, 0, 0, 0)), m_conf(0.0), m_class_id(0), m_label("") {
	}
	Object(cv::Rect box, float32_t conf, const std::string& label, int32_t classId = -1) :
		m_box(box), m_conf(conf), m_label(label), m_class_id(classId) {
	}

public:
	cv::Rect    m_box;
	float32_t   m_conf;
	std::string m_label;
	int32_t     m_class_id;
};


//-------------------------------------------------------------------------------------------------
// Yolov5
//-------------------------------------------------------------------------------------------------
class Yolov5
{
public:
	Yolov5(const std::string& modelPath, const std::vector<std::string>& labels,
		const float32_t confThreshold = 0.65f, const float32_t iouThreshold = 0.5f,
		const std::string& sdkPath = ".\\qnn", const std::string& runtime = "Htp",
		bool debug = false);

	std::vector<Object> inference(
		const std::vector<cv::Mat>& inputs, const std::string& perfProfile = "burst");

	bool check(void);

private:
	std::shared_ptr<void> m_impl;
};


//-------------------------------------------------------------------------------------------------
// draw_object
//-------------------------------------------------------------------------------------------------
extern "C" __declspec(dllexport) void draw_object (const cv::Mat&, const Object&);
extern "C" __declspec(dllexport) void draw_objects(const cv::Mat&, const std::vector<Object>&);


} // yolov5
} // algo
} // qnn
