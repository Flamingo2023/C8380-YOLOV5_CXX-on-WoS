#include "qnncontext.hpp"
#include "yolov5.hpp"


using namespace qnn::algo::yolov5;


//-------------------------------------------------------------------------------------------------
// MACROS
//-------------------------------------------------------------------------------------------------
#define CONV2INT(  x) static_cast<int32_t>(  x)
#define CONV2FLOAT(x) static_cast<float32_t>(x)


//-------------------------------------------------------------------------------------------------
// CONSTANTS
//-------------------------------------------------------------------------------------------------
const std::vector<size_t> YOLOV5_OUTPUT_SIZES = {
	19200 * 85, 4800 * 85, 1200 * 85
};

const std::vector<size_t> YOLOV5_STRIDES = {
	8, 16, 32
};

const std::vector< std::vector<size_t>> YOLOV5_SHAPES = {
	{1, 3, 80, 80, 85},
	{1, 3, 40, 40, 85},
	{1, 3, 20, 20, 85}
};

const std::vector< std::vector<size_t>> YOLOV5_ANCHORS = {
	{10,  13,  16,  30,  33,  23},
	{30,  61,  62,  45,  59, 119},
	{116, 90, 156, 198, 373, 326}
};


//-------------------------------------------------------------------------------------------------
// draw_object
//-------------------------------------------------------------------------------------------------
void qnn::algo::yolov5::draw_object(const cv::Mat& image, const Object& object)
{
	const cv::Scalar color(229, 160, 21);
	const int thickness = 2;
	cv::rectangle(image, object.m_box, color, thickness);

	std::ostringstream out;
	out << object.m_label << "," << std::fixed << std::setprecision(2) << object.m_conf;
	int baseline;
	cv::Size size = cv::getTextSize(out.str(), cv::FONT_ITALIC, 0.6, thickness, &baseline);

	const int pad = 4;
	int x = object.m_box.x < (thickness / 2) ? 0 : (object.m_box.x - thickness / 2),
		y = object.m_box.y < (size.height + pad) ? (object.m_box.y) :
		(object.m_box.y - size.height - pad);
	int width = (object.m_box.width + thickness / 2) < (size.width + pad) ?
		(size.width + pad) : (object.m_box.width + thickness / 2);
	cv::rectangle(image, cv::Point(x, y), cv::Point(x + width, y + size.height + pad),
		color, -1);

	const cv::Scalar white(255, 255, 255);
	cv::putText(image, out.str().c_str(), cv::Point(x + pad / 2, y + size.height + pad / 2),
		cv::FONT_ITALIC, 0.6, white, thickness);
}


//-------------------------------------------------------------------------------------------------
// draw_objects
//-------------------------------------------------------------------------------------------------
void qnn::algo::yolov5::draw_objects(const cv::Mat& image, const std::vector<Object>& objects)
{
	for (auto& object : objects ){
		draw_object(image, object);
	}
}


//-------------------------------------------------------------------------------------------------
// Yolov5Impl
//-------------------------------------------------------------------------------------------------
class Yolov5Impl : public qnn::tools::helper::QnnContext<std::vector<Object>>
{
public:
	Yolov5Impl(const std::string& modelPath, const std::vector<std::string>& labels,
		const float32_t confThreshold = 0.65f, const float32_t iouThreshold = 0.5f,
		const std::string& sdkPath = ".\\qnn", const std::string& runtime = "Htp",
		bool debug = false);

	std::vector<cv::Mat> preprocess(const std::vector<cv::Mat>& images);
	std::vector<Object>  postprocess(const std::vector<cv::Mat>& images,
		const std::vector<std::tuple<std::shared_ptr<float32_t>, size_t>>& outputs);

private:
	static std::string getModelName(void);

private:
	static int               m_instance_id;
	int                      m_id;
	std::vector<std::string> m_labels;
	float32_t                m_conf_threshold;
	float32_t                m_iou_threshold;
	bool                     m_debug;
	float32_t                m_scale;
	int32_t                  m_pad_x;
	int32_t                  m_pad_y;
};

int Yolov5Impl::m_instance_id = 0;

std::string Yolov5Impl::getModelName(void) {
	return std::string("Yolov5") + std::to_string(m_instance_id);
}

Yolov5Impl::Yolov5Impl(const std::string& modelPath, const std::vector<std::string>& labels,
	const float32_t confThreshold, const float32_t iouThreshold,
	const std::string& sdkPath, const std::string& runtime,	bool debug) : 
	QnnContext(getModelName(), modelPath, sdkPath, runtime), m_labels(labels), 
	m_conf_threshold(confThreshold), m_iou_threshold(iouThreshold), m_debug(debug) {
	m_id = m_instance_id++;
}

std::vector<cv::Mat> Yolov5Impl::preprocess(const std::vector<cv::Mat>& images) {
	if (images.size() != 1u) {
		QNN_ERR("invalid images, must be filled only one image");
		return {};
	}

	cv::Mat rgb_image;
	cv::cvtColor(images[0], rgb_image, cv::COLOR_BGR2RGB);

	const int32_t input_width = 640, input_height = 640;
	float32_t scale_width = CONV2FLOAT(input_width) / rgb_image.cols;
	float32_t scale_height = CONV2FLOAT(input_height) / rgb_image.rows;
	int32_t   width, height;
	if (scale_width < scale_height) {
		m_scale = scale_width;
		height = CONV2INT(rgb_image.rows * scale_width);
		width = input_width;
	}
	else {
		m_scale = scale_height;
		width = CONV2INT(rgb_image.cols * scale_height);
		height = input_height;
	}

	cv::Mat resized_image, float32_image;
	cv::resize(rgb_image, resized_image, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
	resized_image.convertTo(float32_image, CV_32FC3, 1.0 / 255.0, 0.0);

	m_pad_x = CONV2INT((input_width - width) / 2.0);
	m_pad_y = CONV2INT((input_height - height) / 2.0);
	cv::Mat padded_image = cv::Mat::zeros(input_height, input_width, CV_32FC3);
	float32_image.copyTo(padded_image(cv::Rect(m_pad_x, m_pad_y, width, height)));

	if (m_debug) {
		cv::imshow(std::string("raw image - ") + std::to_string(m_id), images[0]);
	}

	return { padded_image };
}

std::vector<Object> Yolov5Impl::postprocess(const std::vector<cv::Mat>& images,
	const std::vector<std::tuple<std::shared_ptr<float32_t>, size_t>>& outputs) {
	if (images.size() != 1u) {
		QNN_ERR("invalid images, must be filled only one image");
		return {};
	}
	if (outputs.size() != 3u) {
		QNN_ERR("invalid outputs, must be not empty");
		return {};
	}

	std::vector<cv::Rect>      objBoxes;
	std::vector<float32_t>     objConfs;
	std::vector<int32_t>       objClassIds;
	std::shared_ptr<float32_t> outputSharePtr;
	size_t                     outputSize;

	for (size_t index = 0; index < outputs.size(); index++) {
		std::tie(outputSharePtr, outputSize) = outputs[index];
		if (!outputSharePtr) {
			QNN_ERR("invalid output-%u buffer, must be not nullptr", index);
			break;
		}
		if (YOLOV5_OUTPUT_SIZES[index] != outputSize) {
			QNN_ERR("invalid output-%u size, must be equal %u", index, YOLOV5_OUTPUT_SIZES[index]);
			break;
		}

		float32_t* outputBuffer = outputSharePtr.get();
		for (size_t i = 0; i < outputSize; i += YOLOV5_SHAPES[index][4]) {
			float32_t objConf = outputBuffer[i + 4];
			if (objConf < m_conf_threshold) continue;
			auto& shape = YOLOV5_SHAPES[index];

			int32_t gridX = CONV2INT((i % (shape[3] * shape[4])) / (shape[4]));
			int32_t gridY = CONV2INT((i % (shape[2] * shape[3] * shape[4])) / (shape[3] * shape[4]));
			int32_t gridZ = CONV2INT((i / (shape[2] * shape[3] * shape[4])));
			float32_t x = outputBuffer[i + 0], y = outputBuffer[i + 1];
			float32_t w = outputBuffer[i + 2], h = outputBuffer[i + 3];

			x = (x * 2 - 0.5f + gridX) * YOLOV5_STRIDES[index];
			y = (y * 2 - 0.5f + gridY) * YOLOV5_STRIDES[index];
			w = (w * w * 4 * YOLOV5_ANCHORS[index][gridZ * 2 + 0]);
			h = (h * h * 4 * YOLOV5_ANCHORS[index][gridZ * 2 + 1]);
			outputBuffer[i + 0] = x; outputBuffer[i + 1] = y;
			outputBuffer[i + 2] = w; outputBuffer[i + 3] = h;

			int32_t centerX = CONV2INT(outputBuffer[i + 0]);
			int32_t centerY = CONV2INT(outputBuffer[i + 1]);
			int32_t width = CONV2INT(outputBuffer[i + 2]);
			int32_t height = CONV2INT(outputBuffer[i + 3]);
			int32_t left = CONV2INT(centerX - width / 2);
			int32_t top = CONV2INT(centerY - height / 2);

			float32_t classConf = 0.0;
			int32_t   classId = 0;
			auto get_best_class_info = [&classConf, &classId](const float32_t* classConfs, int n) {
				for (int index = 0; index < n; index++) {
					if (classConfs[index] > classConf) {
						classConf = classConfs[index];
						classId = index;
					}
				}
				};
			get_best_class_info(&outputBuffer[i + 5], (int)(shape[4] - 5));

			float32_t confidence = classConf * objConf;
			objBoxes.emplace_back(left, top, width, height);
			objConfs.emplace_back(confidence);
			objClassIds.emplace_back(classId);
		}
	}

	cv::Mat image;
	if (m_debug) {
		images[0].copyTo(image);
	}

	std::vector<int32_t> indices;
	cv::dnn::NMSBoxes(objBoxes, objConfs, m_conf_threshold, m_iou_threshold, indices);
	std::vector<Object> results;

	for (int index : indices) {
		cv::Rect rect(objBoxes[index]);
		rect.x = CONV2INT(std::round(CONV2FLOAT(rect.x - m_pad_x) / m_scale));
		rect.y = CONV2INT(std::round(CONV2FLOAT(rect.y - m_pad_y) / m_scale));
		rect.width = CONV2INT(std::round(CONV2FLOAT(rect.width) / m_scale));
		rect.height = CONV2INT(std::round(CONV2FLOAT(rect.height) / m_scale));
		Object object(rect, objConfs[index], m_labels[objClassIds[index]]);
		results.emplace_back(object);

		QNN_DBG("Objects[%d]: %dx%d@%d,%d %.4f %s", index, rect.width, rect.height,
			rect.x, rect.y, objConfs[index], m_labels[objClassIds[index]]);
		if (m_debug) {
			draw_object(image, object);
		}
	}

	if (m_debug) {
		cv::imshow(std::string("result image - ") + std::to_string(m_id), image);
	}

	return results;
}


//-------------------------------------------------------------------------------------------------
// Yolov5
//-------------------------------------------------------------------------------------------------
Yolov5::Yolov5(const std::string& modelPath, const std::vector<std::string>& labels,
	const float32_t confThreshold, const float32_t iouThreshold,
	const std::string& sdkPath, const std::string& runtime, bool debug) {
	auto impl = std::make_shared<Yolov5Impl>(modelPath, labels, confThreshold, iouThreshold,
		sdkPath, runtime, debug);
	if (impl->check()) m_impl = std::static_pointer_cast<void>(impl);
}

std::vector<Object> Yolov5::inference(
	const std::vector<cv::Mat>& inputs, const std::string& perfProfile) {
	if (m_impl) {
		auto impl = std::static_pointer_cast<Yolov5Impl>(m_impl);
		return impl->inference(inputs, perfProfile);
	} else {
		return {};
	}
}

bool Yolov5::check(void) {
	if (m_impl) {
		auto impl = std::static_pointer_cast<Yolov5Impl>(m_impl);
		return impl->check();
	}
	else {
		return false;
	}
}