#include "matx/matx_tm.h"

#include <chrono>

matx_int64_t matx_tm_now(matx_tm_unit unit)
{
	auto now = std::chrono::system_clock::now();
	auto ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
	uint64_t count = ns.time_since_epoch().count();

	switch (unit)
	{
	case matx_tm_unit::MATX_TM_SECOND:      return count / 1000000000ULL;
	case matx_tm_unit::MATX_TM_MILISECOND: return count / 1000000ULL;
	case matx_tm_unit::MATX_TM_MICROSECOND: return count / 1000ULL;
	case matx_tm_unit::MATX_TM_NANOSECOND:  return count;
	default: return count;
	}
}
