#pragma once

#include <cstdio>
#include <chrono>


namespace scw
{
	[[nodiscard]] inline std::chrono::time_point<std::chrono::steady_clock> start_timer() noexcept
	{
		return std::chrono::steady_clock::now();
	}


	[[nodiscard]] inline std::chrono::nanoseconds time_since(std::chrono::time_point<std::chrono::steady_clock> p_since) noexcept
	{
		return std::chrono::steady_clock::now() - p_since;
	}


	inline void print_ms(std::chrono::nanoseconds p_duration, const char* p_string = '\0') noexcept
	{
		printf("%s: %llu ms\n", p_string, static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::milliseconds>(p_duration).count()));
	}


	inline void print_micro(std::chrono::nanoseconds p_duration, const char* p_string = '\0') noexcept
	{
		printf("%s: %llu mic\n", p_string, static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::microseconds>(p_duration).count()));
	}


	inline void print(uint64_t p_dead_sum, const char* p_string = "Dead Sum") noexcept
	{
		printf("%s: %llu\n", p_string, p_dead_sum);
	}
}
