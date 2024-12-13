#pragma once

#include <filesystem>

#include <opencv2/opencv.hpp>

#include "LibAppBuilder.hpp"


namespace qnn {
namespace tools {
namespace helper {


//-------------------------------------------------------------------------------------------------
// QnnContext
//-------------------------------------------------------------------------------------------------
template<typename T = nullptr_t>
class QnnContext
{
public:
	QnnContext(const std::string& modelName, const std::string& modelPath,
		const std::string& sdkPath = ".\\qnn", const std::string& runtime = "Htp") {
		if (!std::filesystem::exists(modelPath)) {
			QNN_ERR("invalid modelPath, must be exist in the filesystem");
			m_initialized = false;
			return;
		}

		if (!std::filesystem::exists(sdkPath)) {
			QNN_ERR("invalid sdkPath, must be exist in the filesystem");
			m_initialized = false;
			return;
		}

		if (runtime != "Htp" and runtime != "Cpu") {
			QNN_ERR("invalid runtime, must be 'Htp' or 'Cpu'");
			m_initialized = false;
			return;
		}
		m_runtime = runtime;

		if (!std::filesystem::exists(sdkPath + "\\Qnn" + m_runtime + ".dll")) {
			QNN_ERR("invalid sdkPath, must include the Qnn%s.dll", m_runtime.c_str());
			m_initialized = false;
			return;
		}
		m_backend_path = sdkPath + "\\Qnn" + m_runtime + ".dll";

		if (!std::filesystem::exists(sdkPath + "\\QnnSystem.dll")) {
			QNN_ERR("invalid sdkPath, must include the QnnSystem.dll");
			m_initialized = false;
			return;
		}
		m_system_path = sdkPath + "\\QnnSystem.dll";

		if (!m_app_builder.ModelInitialize(
			modelName, modelPath, m_backend_path, m_system_path)) {
			QNN_ERR("failed to load the model %s", modelPath.c_str());
			m_initialized = false;
			return;
		}

		m_model_name = modelName;
		m_initialized = true;
	}

	T inference(const std::vector<cv::Mat>& inputs, const std::string& perfProfile = "burst") {
		std::vector<cv::Mat> processed_inputs = preprocess(inputs);
		if (processed_inputs.size() == 0) {
			return {};
		}

		std::vector<uint8_t*> inputBuffers = {};
		for (size_t i = 0; i < processed_inputs.size(); i++) {
			inputBuffers.push_back(reinterpret_cast<uint8_t*>(processed_inputs[i].data));
		}

		std::vector<uint8_t*> outputBuffers = {};
		std::vector<size_t>   outputSize = {};
		std::string tmpPerfProfile = perfProfile;
		if (!m_app_builder.ModelInference(
			m_model_name, inputBuffers, outputBuffers, outputSize, tmpPerfProfile)) {
			return {};
		}

		std::vector<std::tuple<std::shared_ptr<float32_t>, size_t>> outputs = {};
		for (auto i = 0; i < outputBuffers.size(); i++) {
			outputs.emplace_back(
				std::shared_ptr<float32_t>(reinterpret_cast<float32_t*>(outputBuffers[i]), free),
				static_cast<size_t>(outputSize[i] / (sizeof(float32_t) / sizeof(uint8_t))));
		}

		return postprocess(inputs, outputs);
	}

	bool check(void) {
		return m_initialized;
	}

private:
	virtual std::vector<cv::Mat> preprocess(const std::vector<cv::Mat>&   ) = 0;
	virtual T postprocess(const std::vector<cv::Mat>&,
		const std::vector<std::tuple<std::shared_ptr<float32_t>, size_t>>&) = 0;

private:
	LibAppBuilder m_app_builder;
	std::string   m_runtime;
	std::string   m_backend_path;
	std::string   m_system_path;
	std::string   m_model_name;
	bool          m_initialized;
};


//-------------------------------------------------------------------------------------------------
// config
//-------------------------------------------------------------------------------------------------
extern "C" __declspec(dllexport) bool config(
	int32_t logLevel, const std::string& logPath, int32_t profileLevel);


} // helper
} // tools
} // qnn
