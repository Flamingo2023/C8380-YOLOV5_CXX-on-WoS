#include "qnncontext.hpp"


//-------------------------------------------------------------------------------------------------
// config
//-------------------------------------------------------------------------------------------------
bool qnn::tools::helper::config(
	int32_t logLevel = 1, const std::string& logPath = "None", int32_t profileLevel = 0) 
{
	/*
		QNN_LOG_LEVEL_ERROR   = 1,
		QNN_LOG_LEVEL_WARN    = 2,
		QNN_LOG_LEVEL_INFO    = 3,
		QNN_LOG_LEVEL_VERBOSE = 4,
		QNN_LOG_LEVEL_DEBUG   = 5,
	*/
	if (logLevel < 1 || logLevel > 5) {
		std::cerr << "Error: invalid logLevel, must be 1(ERROR)~5(DEBUG)" << std::endl;
		return false;
	}

	SetLogLevel(logLevel, logPath);

	/*
		OFF      = 0,
		BASIC    = 1,
		DETAILED = 2,
		INVALID  = 3,
	*/
	if (profileLevel < 0 || profileLevel > 2) {
		QNN_ERR("invalid profileLevel, must be 0(OFF)~2(DETAILED)");
		return false;
	}

	SetProfilingLevel(profileLevel);

	return true;
}
