#include <thread>

#include "scw_bench.h"
#include "bitset_map.h"
#include "plf_rand.h"


constexpr static size_t ELEMENT_COUNT = 1'000'000;


struct payload
{
	uint64_t a;
	uint64_t b;
	uint64_t c;
	uint64_t d;
};


int main()
{
	std::this_thread::sleep_for(std::chrono::seconds(1LL));
	plf::srand(static_cast<unsigned int>(scw::start_timer().time_since_epoch().count()));


	uint64_t dead_sum = 0U;
	scw::bitset_map<payload, ELEMENT_COUNT / 10ULL> dead_map;

	for (uint64_t index = 0U; index < ELEMENT_COUNT / 10ULL; ++index)
	{
		dead_map.emplace_back_unchecked(plf::rand() % 100U, plf::rand() % 100U, plf::rand() % 100U, plf::rand() % 100U);
	}

	for (payload& element : dead_map)
	{
		dead_sum ^= element.a;
		dead_sum ^= element.b;
		dead_sum ^= element.c;
		dead_sum ^= element.c;
	}


	uint64_t iter_sum = plf::rand();
	uint64_t map_sparse_sum = iter_sum;
	uint64_t map_reinsert_iter_sum = iter_sum;

	auto map_insert = scw::start_timer();
	scw::bitset_map<payload, ELEMENT_COUNT> map(ELEMENT_COUNT);

	for (uint64_t index = 0U; index < ELEMENT_COUNT; ++index)
	{
		map.emplace(plf::rand() % 100U, plf::rand() % 100U, plf::rand() % 100U, plf::rand() % 100U);
	}
	auto map_insert_t = scw::time_since(map_insert);


	auto map_iter = scw::start_timer();
	for (payload& element : map)
	{
		iter_sum ^= element.a;
		iter_sum ^= element.b;
		iter_sum ^= element.c;
		iter_sum ^= element.d;
	}
	auto map_iter_t = scw::time_since(map_iter);


	uint64_t erased = 0U;
	auto map_erase = scw::start_timer();
	for (auto it = map.begin(); it != map.end();)
	{
		if (plf::rand() & 1U)
		{
			it = map.erase(it);

			++erased;
		}
		else
		{
			++it;
		}
	}
	auto map_erase_t = scw::time_since(map_erase);


	auto vec_sparse_iter = scw::start_timer();
	for (payload& element : map)
	{
		map_sparse_sum ^= element.a;
		map_sparse_sum ^= element.b;
		map_sparse_sum ^= element.c;
		map_sparse_sum ^= element.d;
	}
	auto map_sparse_iter_t = scw::time_since(vec_sparse_iter);


	auto map_reinsert = scw::start_timer();
	for (uint64_t index = 0U; index < erased; ++index)
	{
		map.emplace(plf::rand() % 100U, plf::rand() % 100U, plf::rand() % 100U, plf::rand() % 100U);
	}
	auto map_reinsert_t = scw::time_since(map_reinsert);


	auto map_reinsert_iter = scw::start_timer();
	for (payload& element : map)
	{
		map_reinsert_iter_sum ^= element.a;
		map_reinsert_iter_sum ^= element.b;
		map_reinsert_iter_sum ^= element.c;
		map_reinsert_iter_sum ^= element.d;
	}
	auto map_reinsert_iter_t = scw::time_since(map_reinsert_iter);


	scw::print_micro(map_insert_t, "Map Allocate");
	scw::print_micro(map_iter_t, "Map Dense Iterate");
	scw::print_micro(map_erase_t, "Map Erase");
	scw::print_micro(map_sparse_iter_t, "Map Sparse Iterate");
	scw::print_micro(map_reinsert_t, "Map Reinsert");
	scw::print_micro(map_reinsert_iter_t, "Map Reinsert Iterate");

	printf("\n");

	scw::print(iter_sum, "Map Sum");
	scw::print(map_sparse_sum, "Map Sparse Sum");
	scw::print(map_reinsert_iter_sum, "Map Reinsert Sum");
	scw::print(dead_sum);

	return 0;
}
