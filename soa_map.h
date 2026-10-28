#pragma once

#include <type_traits>
#include <algorithm>
#include <concepts>
#include <iterator>
#include <utility>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <memory>
#include <tuple>
#include <array>
#include <new>

#include <immintrin.h>


#if defined(__clang__)
#define SCW_FORCE_INLINE [[clang::always_inline]]
#define SCW_NO_INLINE [[clang::noinline]]
#define SCW_NO_UNIQUE_ADDRESS [[no_unique_address]]
#elif defined(__GNUC__)
#define SCW_FORCE_INLINE [[gnu::always_inline]]
#define SCW_NO_INLINE [[gnu::noinline]]
#define SCW_NO_UNIQUE_ADDRESS [[no_unique_address]]
#elif defined(_MSC_VER)
#define SCW_FORCE_INLINE [[msvc::forceinline]]
#define SCW_NO_INLINE [[msvc::noinline]]
#define SCW_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define SCW_FORCE_INLINE
#define SCW_NO_INLINE
#endif


namespace scw
{
	// CONCEPTS
	struct use_generations {};
	struct no_generations {};
	struct is_const {};
	struct not_const {};
	struct dense_iterate {};
	struct sparse_iterate {};
	struct return_table {};
	struct no_table {};

	struct for_each_transform {};
	struct erase_if_transform {};

	struct sse {};
	struct avx2 {};
	struct avx512 {};


	template<class T>
	concept is_trivial_concept = std::is_trivially_copyable_v<T> && std::is_trivially_copy_constructible_v<T> && std::is_trivially_copy_assignable_v<T> && std::is_trivially_destructible_v<T>;

	template<class T>
	concept use_generations_concept = std::same_as<T, use_generations> || std::same_as<T, no_generations>;

	template<class T>
	concept const_iterator_concept = std::same_as<T, is_const> || std::same_as<T, not_const>;

	template<class T>
	concept iteration_concept = std::same_as<T, dense_iterate> || std::same_as<T, sparse_iterate>;

	template<class T>
	concept return_remap_table_concept = std::same_as<T, return_table> || std::same_as<T, no_table>;

	template<class T>
	concept iteration_transform_concept = std::same_as<T, for_each_transform> || std::same_as<T, erase_if_transform>;

	template<class T>
	concept simd_concept = std::same_as<T, sse> || std::same_as<T, avx2> || std::same_as<T, avx512>;


	template<class, const_iterator_concept>
	class soa_map_iterator;

	template<class>
	class remap_table;


	namespace platform
	{
		inline size_t os_page_size;


		[[nodiscard]] size_t get_page_size() noexcept;
		[[nodiscard]] void* reserve(size_t) noexcept;
		[[nodiscard]] bool commit(void*, size_t) noexcept;
		[[nodiscard]] bool free(void*, size_t) noexcept;
		[[nodiscard]] bool decommit(void*, size_t) noexcept;


		[[nodiscard]] inline bool query_system_page_info()
		{
			os_page_size = get_page_size();

			if (os_page_size & os_page_size - 1ULL)
			{
				throw std::bad_alloc();
			}

			return false;
		}


		inline void initialize_system_page_data()
		{
			[[maybe_unused]] static const bool _ = query_system_page_info();
		}
	}


	[[nodiscard]] inline bool get_bit_(const uint64_t* p_bitset, uint32_t p_index) noexcept
	{
		return p_bitset[p_index >> 6U] & 1ULL << static_cast<size_t>(p_index & 63U);
	}


	[[nodiscard]] constexpr inline size_t align_(size_t p_value, size_t p_alignment) noexcept
	{
		return (p_value + p_alignment - 1ULL) & ~(p_alignment - 1ULL);
	}



	// BITSET MAP
	template<uint32_t t_vm_reserve_elements, use_generations_concept t_use_generations, class... t_types>
	class soa_map
	{
	private:
		static_assert((is_trivial_concept<t_types> && ...), "scw::soa_map requires all types to be trivial");
		static_assert(sizeof...(t_types) > 0, "scw::soa_map requires at least one type");
		static_assert(t_vm_reserve_elements&& t_vm_reserve_elements < UINT32_MAX, "scw::soa_map requires reserve size to be between 1 and uint32_t max - 1");

	public: // TYPES
		struct IndividualisticHandle
		{
			uint32_t index;
		};


		struct GenerationalHandle
		{
			uint32_t index;
			uint32_t generation;
		};

	private: // MEMBER ALIASES
		template<size_t t_index>
		using type_ = std::tuple_element_t<t_index, std::tuple<t_types...>>;


		constexpr static bool c_generational = std::same_as<t_use_generations, use_generations>;
		constexpr static size_t c_types_count = sizeof...(t_types);
		constexpr static size_t c_last_type_index = c_types_count - 1ULL;
		constexpr static auto c_indices = std::make_index_sequence<c_types_count>{};


		using types_array = std::array<size_t, c_types_count>;

	public:
		using handle = std::conditional_t<c_generational, GenerationalHandle, IndividualisticHandle>;

		template<class T>
		using iterator = soa_map_iterator<T, not_const>;

		template<class T>
		using const_iterator = soa_map_iterator<T, is_const>;

	public: // CONSTRUCTORS
		soa_map()
		{
			allocate_(1U);
		}


		explicit soa_map(uint32_t p_reserve_count)
		{
			allocate_(p_reserve_count);
		}


		soa_map(uint32_t p_element_count, const t_types&... p_values)
		{
			allocate_(p_element_count);

			m_high_water_mark = p_element_count;
			m_size = p_element_count;

			[] <size_t... t_indices>(std::index_sequence<t_indices...>, void* const* m_data, uint32_t p_element_count, const auto&... p_values) noexcept
			{
				([](void* const* m_data, uint32_t p_element_count, const auto& p_value) noexcept
					{
						for (uint32_t index = 0U; index < p_element_count; ++index)
						{
							::new(static_cast<type_<t_indices>*>(m_data[t_indices]) + index) type_<t_indices>(p_value);
						}
					}(m_data, p_element_count, p_values), ...);
			}(c_indices, m_data, p_element_count, p_values...);
		}


		soa_map(const soa_map& p_other)
		{
			copy_soa_map_(p_other);
		}


		soa_map& operator=(const soa_map& p_other)
		{
			if (this != &p_other)
			{
				soa_map temp(p_other);
				std::swap(*this, temp);
			}

			return *this;
		}


		soa_map(soa_map&& p_other) noexcept
		{
			steal_other_(std::move(p_other));
		}


		soa_map& operator=(soa_map&& p_other) noexcept
		{
			if (this != &p_other)
			{
				deallocate_();

				steal_other_(std::move(p_other));
			}

			return *this;
		}


		~soa_map() noexcept
		{
			deallocate_();
		}

	public: // MEMBER FUNCTIONS
		handle emplace(const t_types&... p_values)
		{
			const uint32_t slot = get_allocation_slot_();

			return construct_in_slot_(slot, p_values...);
		}


		handle insert(const t_types&... p_values)
		{
			return emplace(p_values...);
		}


		handle emplace_back(const t_types&... p_values)
		{
			const uint32_t slot = get_end_allocation_slot_();

			return construct_in_slot_(slot, p_values...);
		}


		handle push_back(const t_types&... p_values)
		{
			return emplace_back(p_values...);
		}


		handle emplace_back_unchecked(const t_types&... p_values) noexcept
		{
			const uint32_t slot = get_unchecked_allocation_slot_();

			return construct_in_slot_(slot, p_values...);
		}


		handle push_back_unchecked(const t_types&... p_values) noexcept
		{
			return emplace_back_unchecked(p_values...);
		}

		// see comment above for_each_while for performance tip regarding lambdas
		// this function is complicated for performance reasons
		// basically, it will pass the soa_map instance and slot index into your lambda, alongside the variadic args you pass into fill
		// 
		// call like this:
		// map.fill(count, [](auto& map, uint32_t slot, int& outside_local_var) noexcept // noexcept is mandatory
		//     {
		//         // do whatever you want here to prepare your data for insertion
		//         map.construct_in_slot(slot, 1, 2.0f, 3.0f, outside_local_var++); // call construct_in_slot with slot or nothing will be constructed
		//     }, local_var);
		template<class t_func, class... Args>
			requires (std::is_nothrow_invocable_v<t_func, soa_map&, uint32_t, Args...>)
		void fill(uint32_t p_count, t_func&& p_func, Args&&... p_args)
		{
			fill_slots_(p_count, std::forward<t_func>(p_func), std::forward<Args>(p_args)...);
		}

		// madina's version
		void fill(uint32_t p_count, const t_types&... p_values)
		{
			fill_slots_(p_count, p_values...);
		}


		SCW_FORCE_INLINE void construct_in_slot(uint32_t p_slot, const t_types&... p_values) noexcept
		{
			construct_in_slot_(p_slot, p_values...);
		}


		void erase(uint32_t p_index) noexcept
		{
			destroy_element_(p_index);
		}


		void erase(handle p_handle) noexcept
		{
			erase(p_handle.index);
		}


		template<size_t t_index>
		iterator<type_<t_index>> erase(const iterator<type_<t_index>>& p_iterator) noexcept
		{
			iterator<type_<t_index>> next_element = p_iterator;
			++next_element;

			destroy_element_(static_cast<uint32_t>(p_iterator.m_skip_offset + p_iterator.m_offset));

			return next_element;
		}


		template<size_t t_index>
		const_iterator<type_<t_index>> erase(const const_iterator<type_<t_index>>& p_iterator) noexcept
		{
			const_iterator<type_<t_index>> next_element = p_iterator;
			++next_element;

			destroy_element_(static_cast<uint32_t>(p_iterator.m_skip_offset + p_iterator.m_offset));

			return next_element;
		}


		void try_erase(uint32_t p_index, uint32_t p_generation) noexcept
			requires (c_generational)
		{
			if (is_generation(p_index, p_generation)) // invariants guarantee liveness
			{
				destroy_element_(p_index);
			}
		}


		void try_erase(uint32_t p_index) noexcept
			requires (!c_generational)
		{
			if (is_alive(p_index))
			{
				destroy_element_(p_index);
			}
		}


		void try_erase(handle p_handle) noexcept
			requires (c_generational)
		{
			try_erase(p_handle.index, p_handle.generation);
		}


		void try_erase(handle p_handle) noexcept
			requires (!c_generational)
		{
			try_erase(p_handle.index);
		}

		// expects user to check mask is non zero and to mask the mask using get_bitmask() because user may know this implicitly
		template<simd_concept t_simd = sse>
		void erase_chunk(uint32_t p_index, uint64_t p_mask) noexcept
		{
			destroy_chunk_<t_simd>(p_index, p_mask);
		}


		// HELPERS
		template<size_t t_index>
		[[nodiscard]] type_<t_index>& at(uint32_t p_index) noexcept
		{
			return static_cast<type_<t_index>*>(m_data[t_index])[p_index];
		}


		template<size_t t_index>
		[[nodiscard]] const type_<t_index>& at(uint32_t p_index) const noexcept
		{
			return static_cast<type_<t_index>*>(m_data[t_index])[p_index];
		}


		template<size_t t_index>
		[[nodiscard]] type_<t_index>& at(handle p_handle) noexcept
		{
			return at<t_index>(p_handle.index);
		}


		template<size_t t_index>
		[[nodiscard]] const type_<t_index>& at(handle p_handle) const noexcept
		{
			return at<t_index>(p_handle.index);
		}


		template<size_t t_index>
		[[nodiscard]] type_<t_index>* try_at(uint32_t p_index, uint32_t p_generation) noexcept
			requires (c_generational)
		{
			if (is_generation(p_index, p_generation))
			{
				return static_cast<type_<t_index>*>(m_data[t_index]) + p_index;
			}

			return nullptr;
		}


		template<size_t t_index>
		[[nodiscard]] type_<t_index>* try_at(uint32_t p_index) noexcept
			requires (!c_generational)
		{
			if (is_alive(p_index))
			{
				return static_cast<type_<t_index>*>(m_data[t_index]) + p_index;
			}

			return nullptr;
		}


		template<size_t t_index>
		[[nodiscard]] const type_<t_index>* try_at(uint32_t p_index, uint32_t p_generation) const noexcept
			requires (c_generational)
		{
			if (is_generation(p_index, p_generation))
			{
				return static_cast<type_<t_index>*>(m_data[t_index]) + p_index;
			}

			return nullptr;
		}


		template<size_t t_index>
		[[nodiscard]] const type_<t_index>* try_at(uint32_t p_index) const noexcept
			requires (!c_generational)
		{
			if (is_alive(p_index))
			{
				return static_cast<type_<t_index>*>(m_data[t_index]) + p_index;
			}

			return nullptr;
		}


		template<size_t t_index>
		[[nodiscard]] type_<t_index>* try_at(handle p_handle) noexcept
			requires (c_generational)
		{
			return try_at<t_index>(p_handle.index, p_handle.generation);
		}


		template<size_t t_index>
		[[nodiscard]] type_<t_index>* try_at(handle p_handle) noexcept
			requires (!c_generational)
		{
			return try_at<t_index>(p_handle.index);
		}


		template<size_t t_index>
		[[nodiscard]] const type_<t_index>* try_at(handle p_handle) const noexcept
			requires (c_generational)
		{
			return try_at<t_index>(p_handle.index, p_handle.generation);
		}


		template<size_t t_index>
		[[nodiscard]] const type_<t_index>* try_at(handle p_handle) const noexcept
			requires (!c_generational)
		{
			return try_at<t_index>(p_handle.index);
		}


		[[nodiscard]] bool is_alive(uint32_t p_index) const noexcept
		{
			return get_bit_(m_skip_data, p_index);
		}


		[[nodiscard]] bool is_alive(handle p_handle) const noexcept
		{
			return is_alive(p_handle.index);
		}


		[[nodiscard]] bool is_generation(uint32_t p_index, uint32_t p_generation) const noexcept
			requires (c_generational)
		{
			return m_generations[p_index] == p_generation;
		}


		[[nodiscard]] bool is_generation(handle p_handle) const noexcept
			requires (c_generational)
		{
			return is_generation(p_handle.index, p_handle.generation);
		}


		[[nodiscard]] uint32_t& get_generation(uint32_t p_index) noexcept
			requires (c_generational)
		{
			return m_generations[p_index];
		}


		[[nodiscard]] const uint32_t& get_generation(uint32_t p_index) const noexcept
			requires (c_generational)
		{
			return m_generations[p_index];
		}


		[[nodiscard]] uint32_t& get_generation(handle p_handle) noexcept
			requires (c_generational)
		{
			return get_generation(p_handle.index);
		}


		[[nodiscard]] const uint32_t& get_generation(handle p_handle) const noexcept
			requires (c_generational)
		{
			return get_generation(p_handle.index);
		}


		template<size_t t_index>
		[[nodiscard]] uint32_t& get_generation(const iterator<type_<t_index>>& p_iterator) noexcept
			requires (c_generational)
		{
			return get_generation(p_iterator.m_skip_offset + p_iterator.m_offset);
		}


		template<size_t t_index>
		[[nodiscard]] const uint32_t& get_generation(const iterator<type_<t_index>>& p_iterator) const noexcept
			requires (c_generational)
		{
			return get_generation(p_iterator.m_skip_offset + p_iterator.m_offset);
		}


		template<size_t t_index>
		[[nodiscard]] uint32_t& get_generation(const const_iterator<type_<t_index>>& p_iterator) noexcept
			requires (c_generational)
		{
			return get_generation(p_iterator.m_skip_offset + p_iterator.m_offset);
		}


		template<size_t t_index>
		[[nodiscard]] const uint32_t& get_generation(const const_iterator<type_<t_index>>& p_iterator) const noexcept
			requires (c_generational)
		{
			return get_generation(p_iterator.m_skip_offset + p_iterator.m_offset);
		}


		template<size_t t_index>
		[[nodiscard]] handle get_handle_from_iterator(const iterator<type_<t_index>>& p_iterator) const noexcept
		{
			const uint32_t index = p_iterator.m_skip_offset + p_iterator.m_offset;

			if constexpr (c_generational)
			{
				return { index, m_generations[index] };
			}
			else
			{
				return { index };
			}
		}


		template<size_t t_index>
		[[nodiscard]] handle get_handle_from_iterator(const const_iterator<type_<t_index>>& p_iterator) const noexcept
		{
			const uint32_t index = p_iterator.m_skip_offset + p_iterator.m_offset;

			if constexpr (c_generational)
			{
				return { index, m_generations[index] };
			}
			else
			{
				return { index };
			}
		}


		template<size_t t_index>
		[[nodiscard]] type_<t_index>* get_chunk(uint32_t p_index) noexcept
		{
			return static_cast<type_<t_index>*>(m_data[t_index]) + (p_index << 6U);
		}


		template<size_t t_index>
		[[nodiscard]] const type_<t_index>* get_chunk(uint32_t p_index) const noexcept
		{
			return static_cast<type_<t_index>*>(m_data[t_index]) + (p_index << 6U);
		}


		[[nodiscard]] bool is_empty() const noexcept
		{
			return !m_size;
		}


		[[nodiscard]] uint32_t size() const noexcept
		{
			return m_size;
		}


		[[nodiscard]] uint32_t end_size() const noexcept
		{
			return m_high_water_mark;
		}


		[[nodiscard]] uint32_t chunk_count() const noexcept
		{
			return m_high_water_mark >> 6U;
		}


		[[nodiscard]] uint64_t get_bitmask(uint32_t p_index) const noexcept
		{
			return m_skip_data[p_index];
		}

		// to push using unckecked insertion functions
		[[nodiscard]] uint32_t back_capacity() const noexcept
		{
			return m_capacity - m_high_water_mark;
		}

		// could be rapidly called in a loop
		[[nodiscard]] float density() const noexcept
		{
			return static_cast<float>(m_size) / static_cast<float>(m_high_water_mark);
		}


		[[nodiscard]] float try_density() const noexcept
		{
			if (m_high_water_mark)
			{
				return static_cast<float>(m_size) / static_cast<float>(m_high_water_mark);
			}
			else
			{
				return 1.0f;
			}
		}

		// checks done in grow_()
		void reserve(uint32_t p_reserve_count)
		{
			if (p_reserve_count > m_capacity)
			{
				grow_(p_reserve_count - m_capacity);
			}
		}

		// intended to break pointer stability
		template<return_remap_table_concept t_return_table = return_table, class Allocator = std::allocator<uint32_t>>
		std::conditional_t<std::same_as<t_return_table, return_table>, remap_table<Allocator>, no_table> compress(const Allocator& p_allocator = Allocator())
		{
			constexpr static bool c_return_table = std::same_as<t_return_table, return_table>;
			std::conditional_t<c_return_table, remap_table<Allocator>, no_table> table;

			if (m_size)
			{
				const uint32_t max_index = m_size - 1U;
				const uint32_t max_word_index = max_index >> 6U;

				uint32_t move_size = 0U;
				for (uint32_t current_index = 0U; current_index != max_word_index; ++current_index)
				{
					move_size += static_cast<uint32_t>(64ULL - _mm_popcnt_u64(m_skip_data[current_index]));
				}

				const uint64_t shift_amount = _andn_u64(static_cast<uint64_t>(max_index), 63ULL);
				move_size += static_cast<uint32_t>(64ULL - _mm_popcnt_u64(m_skip_data[max_word_index] & UINT64_MAX >> shift_amount) - shift_amount);

				if (move_size)
				{
					if constexpr (c_return_table)
					{
						table.allocate_(p_allocator, m_skip_data, static_cast<size_t>(m_size), static_cast<size_t>(m_high_water_mark));
					}

					uint64_t* hole_skip_data = m_skip_data;
					uint64_t current_holes_word = ~*hole_skip_data;
					size_t hole_start = 0ULL;
					uint64_t hole_offset = 0ULL;

					uint64_t* element_skip_data = m_skip_data + (m_size >> 6U);
					uint64_t current_elements_word = *element_skip_data & UINT64_MAX << static_cast<uint64_t>(m_size & 63U);
					size_t element_start = static_cast<size_t>(m_size & ~63U);
					uint64_t element_offset = 0ULL;

					while (true)
					{
						while (current_holes_word)
						{
							hole_offset = _tzcnt_u64(current_holes_word);
							current_holes_word = _blsr_u64(current_holes_word);

							while (!current_elements_word)
							{
								element_start += 64ULL;
								++element_skip_data;
								current_elements_word = *element_skip_data;
							}

							element_offset = _tzcnt_u64(current_elements_word);
							current_elements_word = _blsr_u64(current_elements_word);

							const uint32_t hole_slot = hole_start + hole_offset;
							const uint32_t element_slot = element_start + element_offset;

							[] <size_t... t_indices>(std::index_sequence<t_indices...>, void* const* m_data, size_t hole_slot, size_t element_slot) noexcept
							{
								(::new(static_cast<type_<t_indices>*>(m_data[t_indices]) + hole_slot) type_<t_indices>(static_cast<type_<t_indices>*>(m_data[t_indices])[element_slot]), ...);
							}(c_indices, m_data, hole_slot, element_slot);

							if constexpr (c_return_table)
							{
								table.insert_(element_slot, hole_slot);
							}

							if constexpr (c_generational)
							{
								m_generations[hole_slot] = m_generations[element_slot];
							}

							--move_size;
							if (!move_size)
							{
								goto COMPRESS_END;
							}
						}

						do
						{
							hole_start += 64ULL;
							++hole_skip_data;
							current_holes_word = ~*hole_skip_data;
						} while (!current_holes_word);
					}
				}
			}

		COMPRESS_END:
			m_free_list = UINT32_MAX;
			m_high_water_mark = m_size;

			memset(m_skip_data, 0xFF, get_skip_bytes_for_element_count_(m_capacity));

			return table;
		}


		// sets all m_skip_data bits past high water mark
		// we ignore generations here, that's fine
		void shrink_to_fit() noexcept
		{
			uint32_t index = 0U;

			if (m_size)
			{
				if (m_size == m_high_water_mark)
				{
					decommit_pages_(m_size - 1);

					return;
				}

				const uint64_t shift_amount = _andn_u64(static_cast<uint64_t>(m_high_water_mark), 63ULL);
				uint64_t word = m_skip_data[m_high_water_mark >> 6U] & UINT64_MAX >> shift_amount >> 1ULL;
				size_t skip_offset = static_cast<size_t>(m_high_water_mark & ~63U);

				while (!word) [[unlikely]]
				{
					skip_offset -= 64ULL;
					word = m_skip_data[skip_offset >> 6ULL];
				}

				index = static_cast<uint32_t>(skip_offset + 63ULL - _lzcnt_u64(word));
				decommit_pages_(index);

				m_high_water_mark = index + 1U;
				m_skip_data[m_high_water_mark >> 6U] |= UINT64_MAX << static_cast<uint64_t>(m_high_water_mark & 63U);
				const size_t bytes_to_reset = get_skip_bytes_for_element_count_(m_capacity) - (static_cast<size_t>(m_high_water_mark >> 6U) + 1ULL) * sizeof(uint64_t);
				memset(m_skip_data + (m_high_water_mark >> 6U) + 1ULL, 0xFF, bytes_to_reset);

				uint32_t new_free_list_index = UINT32_MAX;
				for (uint32_t current_index = index >> 6U; current_index != UINT32_MAX; --current_index)
				{
					if (m_skip_data[current_index] != UINT64_MAX)
					{
						m_free_table[current_index] = new_free_list_index;
						new_free_list_index = current_index;
					}
				}

				m_free_list = new_free_list_index;

				return;
			}

			decommit_pages_(index);

			memset(m_skip_data, 0xFF, get_skip_bytes_for_element_count_(m_capacity));

			m_free_list = UINT32_MAX;
			m_high_water_mark = 0U;
		}


		void clear() noexcept
		{
			if constexpr (c_generational)
			{
				for (uint32_t index = 0U; index < m_high_water_mark; ++index)
				{
					++m_generations[index];
				}
			}

			memset(m_skip_data, 0xFF, static_cast<size_t>((m_high_water_mark >> 6U) + 1U) * sizeof(uint64_t));

			m_free_list = UINT32_MAX;
			m_high_water_mark = 0U;
			m_size = 0U;
		}

		// ITERATORS
		// if there is no element at index 0, find the first element
		template<size_t t_index>
		[[nodiscard]] iterator<type_<t_index>> begin() noexcept
		{
			iterator<type_<t_index>> to_return(static_cast<type_<t_index>*>(m_data[t_index]), m_skip_data, *m_skip_data & UINT64_MAX << 1ULL, 0U, 0U);

			return *m_skip_data & 1ULL ? to_return : ++to_return;
		}


		template<size_t t_index>
		[[nodiscard]] iterator<type_<t_index>> end() noexcept
		{
			return iterator<type_<t_index>>(static_cast<type_<t_index>*>(m_data[t_index]) + (m_high_water_mark & ~63U), m_skip_data, 0ULL, m_high_water_mark & ~63U, m_high_water_mark & 63U);
		}


		template<size_t t_index>
		[[nodiscard]] const_iterator<type_<t_index>> begin() const noexcept
		{
			const_iterator<type_<t_index>> to_return(static_cast<type_<t_index>*>(m_data[t_index]), m_skip_data, *m_skip_data & UINT64_MAX << 1ULL, 0U, 0U);

			return *m_skip_data & 1ULL ? to_return : ++to_return;
		}


		template<size_t t_index>
		[[nodiscard]] const_iterator<type_<t_index>> end() const noexcept
		{
			return const_iterator<type_<t_index>>(static_cast<type_<t_index>*>(m_data[t_index]) + (m_high_water_mark & ~63U), m_skip_data, 0ULL, m_high_water_mark & ~63U, m_high_water_mark & 63U);
		}


		template<size_t t_index>
		[[nodiscard]] const_iterator<type_<t_index>> cbegin() const noexcept
		{
			return begin<t_index>();
		}


		template<size_t t_index>
		[[nodiscard]] const_iterator<type_<t_index>> cend() const noexcept
		{
			return end<t_index>();
		}


		template<size_t t_index>
		[[nodiscard]] iterator<type_<t_index>> get_iterator_from_chunk_index(uint32_t p_index) noexcept
		{
			iterator<type_<t_index>> to_return(static_cast<type_<t_index>*>(m_data[t_index]) + (p_index << 6U), m_skip_data, m_skip_data[p_index] & UINT64_MAX << 1ULL, p_index << 6U, 0U);

			return m_skip_data[p_index] & 1ULL ? to_return : ++to_return;
		}


		template<size_t t_index>
		[[nodiscard]] const_iterator<type_<t_index>> get_iterator_from_chunk_index(uint32_t p_index) const noexcept
		{
			const_iterator<type_<t_index>> to_return(static_cast<type_<t_index>*>(m_data[t_index]) + (p_index << 6U), m_skip_data, m_skip_data[p_index] & UINT64_MAX << 1ULL, p_index << 6U, 0U);

			return m_skip_data[p_index] & 1ULL ? to_return : ++to_return;
		}

		// as a performance tip, avoid capturing variables you intend to modify by reference [&]
		// observe that the function is variadic, pass references in after your lambda, make them parameters to your lambda
		// this should help the compiler vectorize and stop writing your variable to the stack, as lambda references seem to make it paranoid
		// about aliasing and obfuscates the code, reducing it's ability to vectorize
		//
		// call like this:
		// map.for_each_while([](uint32_t index, float& f, double& d, auto&, int& outside_local_var) noexcept // noexcept is mandatory, index is provided for deferred deletion pattern
		//     {
		//         // use auto& to ignore types you dont care about at the moment, the compiler will optimize them away
		//         // do whatever you want here
		//         f = 1.0f;
		//         d = 2.0;
		// 
		//         return true; // mandatory to continue, false will exit
		//     }, local_var);
		template<iteration_concept t_iteration = dense_iterate, class t_func, class... Args>
		SCW_FORCE_INLINE void for_each_while(t_func&& p_func, Args&&... p_args) noexcept
			requires (std::is_nothrow_invocable_r_v<bool, t_func, uint32_t, t_types&..., Args...>)
		{
			visit_all_caller_<t_iteration, for_each_transform>(std::forward<t_func>(p_func), 0U, UINT32_MAX, std::forward<Args>(p_args)...);
		}


		template<iteration_concept t_iteration = dense_iterate, class t_func, class... Args>
		SCW_FORCE_INLINE void for_each_while(t_func&& p_func, Args&&... p_args) const noexcept
			requires (std::is_nothrow_invocable_r_v<bool, t_func, uint32_t, const t_types&..., Args...>)
		{
			visit_all_caller_<t_iteration, for_each_transform>(std::forward<t_func>(p_func), 0U, UINT32_MAX, std::forward<Args>(p_args)...);
		}


		template<iteration_concept t_iteration = dense_iterate, class t_func, class... Args>
		SCW_FORCE_INLINE void for_each_chunk_while(t_func&& p_func, uint32_t p_start_chunk, uint32_t p_end_chunk = UINT32_MAX, Args&&... p_args) noexcept
			requires (std::is_nothrow_invocable_r_v<bool, t_func, uint32_t, t_types&..., Args...>)
		{
			visit_all_caller_<t_iteration, for_each_transform>(std::forward<t_func>(p_func), p_start_chunk, p_end_chunk, std::forward<Args>(p_args)...);
		}


		template<iteration_concept t_iteration = dense_iterate, class t_func, class... Args>
		SCW_FORCE_INLINE void for_each_chunk_while(t_func&& p_func, uint32_t p_start_chunk, uint32_t p_end_chunk = UINT32_MAX, Args&&... p_args) const noexcept
			requires (std::is_nothrow_invocable_r_v<bool, t_func, uint32_t, const t_types&..., Args...>)
		{
			visit_all_caller_<t_iteration, for_each_transform>(std::forward<t_func>(p_func), p_start_chunk, p_end_chunk, std::forward<Args>(p_args)...);
		}


		template<iteration_concept t_iteration = dense_iterate, class t_func, class... Args>
		SCW_FORCE_INLINE void erase_if(t_func&& p_func, Args&&... p_args) noexcept
			requires (std::is_nothrow_invocable_r_v<bool, t_func, uint32_t, t_types&..., Args...>)
		{
			visit_all_caller_<t_iteration, erase_if_transform>(std::forward<t_func>(p_func), 0U, UINT32_MAX, std::forward<Args>(p_args)...);
		}


		template<iteration_concept t_iteration = dense_iterate, class t_func, class... Args>
		SCW_FORCE_INLINE void erase_between_chunks_if(t_func&& p_func, uint32_t p_start_chunk, uint32_t p_end_chunk = UINT32_MAX, Args&&... p_args) noexcept
			requires (std::is_nothrow_invocable_r_v<bool, t_func, uint32_t, t_types&..., Args...>)
		{
			visit_all_caller_<t_iteration, erase_if_transform>(std::forward<t_func>(p_func), p_start_chunk, p_end_chunk, std::forward<Args>(p_args)...);
		}

	private: // IMPLEMENTATION
		// VM reservation split between many memory blocks here
		SCW_NO_INLINE void allocate_(uint32_t p_reserve_count)
		{
			p_reserve_count = std::clamp(p_reserve_count, 1U, t_vm_reserve_elements);

			constexpr static types_array c_aligned_data_bytes_array =
				[] <size_t... t_indices>(std::index_sequence<t_indices...>) noexcept -> types_array
			{
				types_array sizes;
				std::array<size_t, c_types_count + 1ULL> alignments;
				((sizes[t_indices] = get_bytes_for_element_count_(t_vm_reserve_elements, sizeof(type_<t_indices>))), ...);
				((alignments[t_indices] = alignof(type_<t_indices>)), ...);
				alignments[c_types_count] = alignof(uint64_t);

				size_t total_size = 0ULL;
				for (size_t index = 0ULL; index < c_types_count; ++index)
				{
					sizes[index] = align_(total_size + sizes[index], alignments[index + 1ULL]);
					sizes[index] -= total_size;
					total_size += sizes[index];
				}

				return sizes;
			}(c_indices);

			constexpr static size_t c_aligned_data_bytes = []() noexcept -> size_t
				{
					size_t aligned_data_bytes = 0ULL;
					for (size_t index = 0ULL; index < c_types_count; ++index)
					{
						aligned_data_bytes += c_aligned_data_bytes_array[index];
					}

					return aligned_data_bytes;
				}();

			const types_array reserve_sizes =
				[] <size_t... t_indices>(std::index_sequence<t_indices...>, uint32_t p_reserve_count) noexcept -> types_array
			{
				types_array sizes;
				((sizes[t_indices] = get_bytes_for_element_count_(p_reserve_count, sizeof(type_<t_indices>))), ...);

				return sizes;
			}(c_indices, p_reserve_count);

			constexpr static size_t c_aligned_skip_array_bytes = align_(get_skip_bytes_for_element_count_(t_vm_reserve_elements) + c_aligned_data_bytes, 64ULL) - c_aligned_data_bytes;
			const size_t skip_reserve_size = get_skip_bytes_for_element_count_(p_reserve_count);

			constexpr static size_t c_aligned_generation_bytes = align_(get_generation_bytes_for_element_count_(t_vm_reserve_elements), alignof(uint32_t));
			const size_t generation_reserve_size = get_generation_bytes_for_element_count_(p_reserve_count);

			constexpr static size_t c_aligned_free_table_bytes = get_free_table_bytes_for_element_count_(t_vm_reserve_elements);
			const size_t free_table_reserve_size = get_free_table_bytes_for_element_count_(p_reserve_count);

			[[maybe_unused]] static const bool _ = initialize_reserve_sizes_(c_aligned_data_bytes, c_aligned_skip_array_bytes, c_aligned_generation_bytes, c_aligned_free_table_bytes);

			m_data[0] = platform::reserve(sm_reserved_bytes);

			if (!m_data[0]) [[unlikely]]
			{
				allocate_fail_();
			}

			[] <size_t... t_indices>(std::index_sequence<t_indices...>, void** m_data) noexcept
			{
				((m_data[t_indices + 1ULL] = reinterpret_cast<char*>(m_data[t_indices]) + c_aligned_data_bytes_array[t_indices]), ...);
			}(std::make_index_sequence<c_last_type_index>{}, m_data);

			m_skip_data = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(m_data[0]) + c_aligned_data_bytes);
			if constexpr (c_generational)
			{
				m_generations = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(m_skip_data) + c_aligned_skip_array_bytes);
			}
			m_free_table = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(m_skip_data) + c_aligned_skip_array_bytes + c_aligned_generation_bytes);

			const bool commit_result =
				[] <size_t... t_indices>(std::index_sequence<t_indices...>, void* const* m_data, const types_array & reserve_sizes) noexcept
			{
				return (platform::commit(static_cast<type_<t_indices>*>(m_data[t_indices]), reserve_sizes[t_indices]) && ...);
			}(c_indices, m_data, reserve_sizes);

			if (!commit_result) [[unlikely]]
			{
				allocate_fail_();
			}
			if (!platform::commit(m_skip_data, skip_reserve_size)) [[unlikely]]
			{
				allocate_fail_();
			}
			if constexpr (c_generational)
			{
				if (!platform::commit(m_generations, generation_reserve_size)) [[unlikely]]
				{
					allocate_fail_();
				}
			}
			if (!platform::commit(m_free_table, free_table_reserve_size)) [[unlikely]]
			{
				allocate_fail_();
			}

			memset(m_skip_data, 0xFF, skip_reserve_size);

			m_capacity = p_reserve_count;
		}


		void allocate_fail_()
		{
			deallocate_();

			throw std::bad_alloc();
		}


		void copy_soa_map_(const soa_map& p_other)
		{
			m_free_list = p_other.m_free_list;
			m_high_water_mark = p_other.m_high_water_mark;
			m_size = p_other.m_size;

			allocate_(m_high_water_mark);

			[] <size_t... t_indices>(std::index_sequence<t_indices...>, void* const* m_data, uint32_t m_high_water_mark, const soa_map & p_other) noexcept
			{
				(memcpy(m_data[t_indices], p_other.m_data[t_indices], static_cast<size_t>(m_high_water_mark) * sizeof(type_<t_indices>)), ...);
			}(c_indices, m_data, m_high_water_mark, p_other);

			memcpy(m_skip_data, p_other.m_skip_data, get_skip_bytes_for_element_count_(m_high_water_mark));
			if constexpr (c_generational)
			{
				memcpy(m_generations, p_other.m_generations, get_generation_bytes_for_element_count_(m_high_water_mark));
			}
			memcpy(m_free_table, p_other.m_free_table, get_free_table_bytes_for_element_count_(m_high_water_mark));
		}


		void steal_other_(soa_map&& p_other) noexcept
		{
			memcpy(m_data, p_other.m_data, c_types_count * sizeof(void*));
			m_skip_data = p_other.m_skip_data;
			if constexpr (c_generational)
			{
				m_generations = p_other.m_generations;
			}
			m_free_table = p_other.m_free_table;
			m_free_list = p_other.m_free_list;
			m_capacity = p_other.m_capacity;
			m_high_water_mark = p_other.m_high_water_mark;
			m_size = p_other.m_size;

			memset(p_other.m_data, 0, c_types_count * sizeof(void*));
			p_other.m_skip_data = nullptr;
			if constexpr (c_generational)
			{
				p_other.m_generations = nullptr;
			}
			p_other.m_free_table = nullptr;
			p_other.m_free_list = UINT32_MAX;
			p_other.m_capacity = 0U;
			p_other.m_high_water_mark = 0U;
			p_other.m_size = 0U;
		}

		// this is okay, m_data is checked
		void deallocate_() noexcept
		{
			if (m_data[0])
			{
				if (!platform::free(m_data[0], sm_reserved_bytes)) [[unlikely]]
				{
					std::abort();
				}

				m_data[0] = nullptr;
			}
		}

		// grow intended to silently fail when full
		SCW_NO_INLINE void grow_(uint32_t p_elements_to_commit)
		{
			p_elements_to_commit = std::min(p_elements_to_commit, t_vm_reserve_elements - m_capacity);

			const types_array old_bytes_array =
				[] <size_t... t_indices>(std::index_sequence<t_indices...>, uint32_t m_capacity) noexcept -> types_array
			{
				types_array old_bytes_array;
				((old_bytes_array[t_indices] = get_bytes_for_element_count_(m_capacity, sizeof(type_<t_indices>))), ...);

				return old_bytes_array;
			}(c_indices, m_capacity);

			const types_array bytes_to_commit_array =
				[] <size_t... t_indices>(std::index_sequence<t_indices...>, uint32_t p_elements_to_commit) noexcept -> types_array
			{
				types_array bytes_to_commit_array;
				((bytes_to_commit_array[t_indices] = get_bytes_for_element_count_(p_elements_to_commit, sizeof(type_<t_indices>))), ...);

				return bytes_to_commit_array;
			}(c_indices, p_elements_to_commit);

			const size_t old_skip_bytes = get_skip_bytes_for_element_count_(m_capacity);
			const size_t skip_bytes_to_commit = get_skip_bytes_for_element_count_(m_capacity + p_elements_to_commit) - old_skip_bytes;

			size_t old_generation_bytes;
			size_t generation_bytes_to_commit;
			if constexpr (c_generational)
			{
				old_generation_bytes = get_generation_bytes_for_element_count_(m_capacity);
				generation_bytes_to_commit = get_generation_bytes_for_element_count_(m_capacity + p_elements_to_commit) - old_generation_bytes;
			}

			const size_t old_free_table_bytes = get_free_table_bytes_for_element_count_(m_capacity);
			const size_t free_table_bytes_to_commit = get_free_table_bytes_for_element_count_(m_capacity + p_elements_to_commit) - old_free_table_bytes;

			for (size_t index = 0ULL; index < c_types_count; ++index)
			{
				const size_t bytes_to_commit = bytes_to_commit_array[index];

				if (bytes_to_commit)
				{
					if (!platform::commit(reinterpret_cast<char*>(m_data[index]) + old_bytes_array[index], bytes_to_commit)) [[unlikely]]
					{
						throw std::bad_alloc();
					}
				}
			}
			if (skip_bytes_to_commit)
			{
				if (!platform::commit(reinterpret_cast<char*>(m_skip_data) + old_skip_bytes, skip_bytes_to_commit)) [[unlikely]]
				{
					throw std::bad_alloc();
				}
			}
			if constexpr (c_generational)
			{
				if (generation_bytes_to_commit)
				{
					if (!platform::commit(reinterpret_cast<char*>(m_generations) + old_generation_bytes, generation_bytes_to_commit)) [[unlikely]]
					{
						throw std::bad_alloc();
					}
				}
			}
			if (free_table_bytes_to_commit)
			{
				if (!platform::commit(reinterpret_cast<char*>(m_free_table) + old_free_table_bytes, free_table_bytes_to_commit)) [[unlikely]]
				{
					throw std::bad_alloc();
				}
			}

			memset(reinterpret_cast<char*>(m_skip_data) + old_skip_bytes, 0xFF, skip_bytes_to_commit);

			m_capacity = m_capacity + p_elements_to_commit;
		}


		// decommits physical memory, making sure not to decommit page when one memory block bleeds into the page of another
		SCW_NO_INLINE void decommit_pages_(uint32_t p_index) noexcept
		{
			const types_array page_offsets_array =
				[] <size_t... t_indices>(std::index_sequence<t_indices...>, void* const* m_data) noexcept -> types_array
			{
				types_array page_offsets_array;
				((page_offsets_array[t_indices] = reinterpret_cast<uintptr_t>(m_data[t_indices]) - _andn_u64(platform::os_page_size - 1ULL, reinterpret_cast<uintptr_t>(m_data[t_indices]))), ...);

				return page_offsets_array;
			}(c_indices, m_data);

			const types_array bytes_occupied_array =
				[] <size_t... t_indices>(std::index_sequence<t_indices...>, uint32_t p_index, const types_array & page_offsets_array) noexcept -> types_array
			{
				types_array bytes_occupied_array;
				((bytes_occupied_array[t_indices] = align_(page_offsets_array[t_indices] + get_bytes_for_element_count_(p_index + 1U, sizeof(type_<t_indices>)), platform::os_page_size)), ...);

				return bytes_occupied_array;
			}(c_indices, p_index, page_offsets_array);

			const types_array bytes_comitted_array =
				[] <size_t... t_indices>(std::index_sequence<t_indices...>, uint32_t m_capacity, const types_array & page_offsets_array) noexcept -> types_array
			{
				types_array bytes_comitted_array;
				((bytes_comitted_array[t_indices] = align_(page_offsets_array[t_indices] + get_bytes_for_element_count_(m_capacity, sizeof(type_<t_indices>)), platform::os_page_size)), ...);

				return bytes_comitted_array;
			}(c_indices, m_capacity, page_offsets_array);

			types_array bytes_to_decommit_array =
				[] <size_t... t_indices>(std::index_sequence<t_indices...>, const types_array & bytes_occupied_array, const types_array & bytes_comitted_array) noexcept -> types_array
			{
				types_array bytes_to_decommit_array;
				((bytes_to_decommit_array[t_indices] = bytes_comitted_array[t_indices] - bytes_occupied_array[t_indices]), ...);

				return bytes_to_decommit_array;
			}(c_indices, bytes_occupied_array, bytes_comitted_array);

			const size_t skip_page_offset = reinterpret_cast<uintptr_t>(m_skip_data) - _andn_u64(platform::os_page_size - 1ULL, reinterpret_cast<uintptr_t>(m_skip_data));
			const size_t skip_bytes_occupied = align_(skip_page_offset + get_skip_bytes_for_element_count_(static_cast<size_t>(p_index + 1U)), platform::os_page_size);
			const size_t skip_bytes_comitted = align_(skip_page_offset + get_skip_bytes_for_element_count_(m_capacity), platform::os_page_size);
			size_t skip_bytes_to_decommit = skip_bytes_comitted - skip_bytes_occupied;

			size_t generation_page_offset;
			size_t generation_bytes_occupied;
			size_t generation_bytes_comitted;
			size_t generation_bytes_to_decommit;
			if constexpr (c_generational)
			{
				generation_page_offset = reinterpret_cast<uintptr_t>(m_generations) - _andn_u64(platform::os_page_size - 1ULL, reinterpret_cast<uintptr_t>(m_generations));
				generation_bytes_occupied = align_(generation_page_offset + get_generation_bytes_for_element_count_(static_cast<size_t>(p_index + 1U)), platform::os_page_size);
				generation_bytes_comitted = align_(generation_page_offset + get_generation_bytes_for_element_count_(m_capacity), platform::os_page_size);
				generation_bytes_to_decommit = generation_bytes_comitted - generation_bytes_occupied;
			}

			const size_t free_table_page_offset = reinterpret_cast<uintptr_t>(m_free_table) - _andn_u64(platform::os_page_size - 1ULL, reinterpret_cast<uintptr_t>(m_free_table));
			const size_t free_table_bytes_occupied = align_(free_table_page_offset + get_free_table_bytes_for_element_count_(static_cast<size_t>(p_index + 1U)), platform::os_page_size);
			const size_t free_table_bytes_comitted = align_(free_table_page_offset + get_free_table_bytes_for_element_count_(m_capacity), platform::os_page_size);
			const size_t free_table_bytes_to_decommit = free_table_bytes_comitted - free_table_bytes_occupied;


			for (size_t index = 0ULL; index < c_last_type_index; ++index)
			{
				if (bytes_to_decommit_array[index])
				{
					if (reinterpret_cast<char*>(m_data[index]) - page_offsets_array[index] + bytes_comitted_array[index] > reinterpret_cast<char*>(m_data[index + 1ULL]))
					{
						bytes_to_decommit_array[index] -= platform::os_page_size;
					}
				}
			}
			if (bytes_to_decommit_array[c_last_type_index])
			{
				if (reinterpret_cast<char*>(m_data[c_last_type_index]) - page_offsets_array[c_last_type_index] + bytes_comitted_array[c_last_type_index] > reinterpret_cast<char*>(m_skip_data))
				{
					bytes_to_decommit_array[c_last_type_index] -= platform::os_page_size;
				}
			}
			if constexpr (c_generational)
			{
				if (skip_bytes_to_decommit)
				{
					if (reinterpret_cast<char*>(m_skip_data) - skip_page_offset + skip_bytes_comitted > reinterpret_cast<char*>(m_generations))
					{
						skip_bytes_to_decommit -= platform::os_page_size;
					}
				}
				if (generation_bytes_to_decommit)
				{
					if (reinterpret_cast<char*>(m_generations) - generation_page_offset + generation_bytes_comitted > reinterpret_cast<char*>(m_free_table))
					{
						generation_bytes_to_decommit -= platform::os_page_size;
					}
				}
			}
			else
			{
				if (skip_bytes_to_decommit)
				{
					if (reinterpret_cast<char*>(m_skip_data) - skip_page_offset + skip_bytes_comitted > reinterpret_cast<char*>(m_free_table))
					{
						skip_bytes_to_decommit -= platform::os_page_size;
					}
				}
			}

			for (size_t index = 0ULL; index < c_types_count; ++index)
			{
				const size_t bytes_to_decommit = bytes_to_decommit_array[index];

				if (bytes_to_decommit)
				{
					if (!platform::decommit(reinterpret_cast<char*>(m_data[index]) - page_offsets_array[index] + bytes_occupied_array[index], bytes_to_decommit)) [[unlikely]]
					{
						std::abort();
					}
				}
			}
			if (skip_bytes_to_decommit)
			{
				if (!platform::decommit(reinterpret_cast<char*>(m_skip_data) - skip_page_offset + skip_bytes_occupied, skip_bytes_to_decommit)) [[unlikely]]
				{
					std::abort();
				}
			}
			if constexpr (c_generational)
			{
				if (generation_bytes_to_decommit)
				{
					if (!platform::decommit(reinterpret_cast<char*>(m_generations) - generation_page_offset + generation_bytes_occupied, generation_bytes_to_decommit)) [[unlikely]]
					{
						std::abort();
					}
				}
			}
			if (free_table_bytes_to_decommit)
			{
				if (!platform::decommit(reinterpret_cast<char*>(m_free_table) - free_table_page_offset + free_table_bytes_occupied, free_table_bytes_to_decommit)) [[unlikely]]
				{
					std::abort();
				}
			}

			m_capacity = p_index + 1U;
		}

		// here, dependency of the not on word doesn't seem to matter, nor does the word load. Compiler seems to cache word here when insert is called in a loop
		// chunked free list here increases latency, but reduces memory accesses drastically and removes read before write dependency, making effective latency less compared to intrusive free list
		SCW_FORCE_INLINE [[nodiscard]] uint32_t get_allocation_slot_()
		{
			if (m_free_list != UINT32_MAX)
			{
				uint64_t word = m_skip_data[m_free_list];

				const uint64_t offset = _tzcnt_u64(~word);
				word |= word + 1ULL;
				const uint32_t slot = (m_free_list << 6U) + static_cast<uint32_t>(offset);

				m_skip_data[m_free_list] = word;

				if (word == UINT64_MAX) [[unlikely]]
				{
					m_free_list = m_free_table[m_free_list];
				}

				++m_size;

				return slot;
			}

			const uint32_t slot = m_high_water_mark;

			if (slot + 1U >= m_capacity)
			{
				grow_(m_capacity);
			}

			++m_high_water_mark;
			++m_size;

			return slot;
		}


		SCW_FORCE_INLINE [[nodiscard]] uint32_t get_end_allocation_slot_()
		{
			const uint32_t slot = m_high_water_mark;

			if (slot + 1U >= m_capacity)
			{
				grow_(m_capacity);
			}

			++m_high_water_mark;
			++m_size;

			return slot;
		}


		SCW_FORCE_INLINE [[nodiscard]] uint32_t get_unchecked_allocation_slot_() noexcept
		{
			const uint32_t slot = m_high_water_mark;

			++m_high_water_mark;
			++m_size;

			return slot;
		}


		template<class t_func, class... Args>
		SCW_FORCE_INLINE void fill_slots_(uint32_t p_count, t_func&& p_func, Args&&... p_args)
		{
			m_size += p_count;

			while (m_free_list != UINT32_MAX)
			{
				uint64_t word = ~m_skip_data[m_free_list];

				const uint64_t scaled_index = static_cast<uint64_t>(m_free_list) << 6ULL;
				const uint32_t free_slots = static_cast<uint32_t>(_mm_popcnt_u64(word));

				for (uint32_t count = 0U; count < free_slots && count < p_count; ++count)
				{
					const uint64_t offset = _tzcnt_u64(word);
					p_func(*this, static_cast<uint32_t>(scaled_index + offset), p_args...);
					word = _blsr_u64(word);
				}

				m_skip_data[m_free_list] = ~word;

				if (!word)
				{
					m_free_list = m_free_table[m_free_list];
				}
				else
				{
					return;
				}

				p_count -= free_slots;
			}

			m_size -= p_count;
			if (m_high_water_mark + p_count >= m_capacity)
			{
				grow_(m_high_water_mark + p_count - m_capacity);
			}
			m_size += p_count;

			for (uint32_t count = 0U; count < p_count; ++count)
			{
				p_func(*this, m_high_water_mark, std::forward<Args>(p_args)...);
				++m_high_water_mark;
			}
		}


		SCW_FORCE_INLINE void fill_slots_(uint32_t p_count, const t_types&... p_values)
		{
			m_size += p_count;

			while (m_free_list != UINT32_MAX)
			{
				uint64_t word = ~m_skip_data[m_free_list];

				const uint64_t scaled_index = static_cast<uint64_t>(m_free_list) << 6ULL;
				const uint32_t free_slots = static_cast<uint32_t>(_mm_popcnt_u64(word));

				for (uint32_t count = 0U; count < free_slots && count < p_count; ++count)
				{
					const uint64_t offset = _tzcnt_u64(word);
					construct_in_slot_(static_cast<uint32_t>(scaled_index + offset), p_values...);
					word = _blsr_u64(word);
				}

				m_skip_data[m_free_list] = ~word;

				if (!word)
				{
					m_free_list = m_free_table[m_free_list];
				}
				else
				{
					return;
				}

				p_count -= free_slots;
			}

			m_size -= p_count;
			if (m_high_water_mark + p_count >= m_capacity)
			{
				grow_(m_high_water_mark + p_count - m_capacity);
			}
			m_size += p_count;

			[] <size_t... t_indices>(std::index_sequence<t_indices...>, void* const* m_data, uint32_t m_high_water_mark, uint32_t p_count, const auto&... p_values) noexcept
			{
				([](void* const* m_data, uint32_t m_high_water_mark, uint32_t p_count, const auto& p_value) noexcept
					{
						for (uint32_t index = 0U; index < p_count; ++index)
						{
							::new(reinterpret_cast<type_<t_indices>*>(m_data[t_indices]) + m_high_water_mark) type_<t_indices>(p_value);
							++m_high_water_mark;
						}
					}(m_data, m_high_water_mark, p_count, p_values), ...);
			}(c_indices, m_data, m_high_water_mark, p_count, p_values...);

			m_high_water_mark += p_count;
		}


		SCW_FORCE_INLINE handle construct_in_slot_(uint32_t p_slot, const t_types&... p_values) noexcept
		{
			[] <size_t... t_indices>(std::index_sequence<t_indices...>, void* const* m_data, uint32_t p_slot, const auto&... p_values) noexcept
			{
				(::new(reinterpret_cast<type_<t_indices>*>(m_data[t_indices]) + p_slot) type_<t_indices>(p_values), ...);
			}(c_indices, m_data, p_slot, p_values...);

			if constexpr (c_generational)
			{
				return { p_slot, m_generations[p_slot] };
			}
			else
			{
				return { p_slot };
			}
		}

		// compared to a per element intrusive free list, this design is slower computationally but reduces memory accesses drastically
		// should be slightly slower in hot cache and much faster when not hot in cache
		SCW_FORCE_INLINE void destroy_element_(uint32_t p_index) noexcept
		{
			const uint32_t chunk_index = p_index >> 6U;
			const uint64_t word = m_skip_data[chunk_index];

			--m_size;

			m_skip_data[chunk_index] = _andn_u64(1ULL << (p_index & 63U), word);

			if constexpr (c_generational)
			{
				++m_generations[p_index];
			}

			if (word == UINT64_MAX) [[unlikely]]
			{
				::new(m_free_table + chunk_index) uint32_t(m_free_list);
				m_free_list = chunk_index;
			}
		}


		template<simd_concept t_simd>
		SCW_FORCE_INLINE void destroy_chunk_(uint32_t p_index, uint64_t p_mask) noexcept
		{
			if constexpr (!c_generational)
			{
				const uint64_t word = m_skip_data[p_index];

				m_size -= static_cast<uint32_t>(_mm_popcnt_u64(p_mask));

				m_skip_data[p_index] = _andn_u64(p_mask, word);

				if (word == UINT64_MAX) [[unlikely]]
				{
					if (p_mask)
					{
						::new(m_free_table + p_index) uint32_t(m_free_list);
						m_free_list = p_index;
					}
				}
			}
			else
			{
				if (p_mask)
				{
					const uint64_t word = m_skip_data[p_index];

					m_size -= static_cast<uint32_t>(_mm_popcnt_u64(p_mask));

					m_skip_data[p_index] = _andn_u64(p_mask, word);

					uint32_t* const base = m_generations + (p_index << 6U);

					if constexpr (std::same_as<t_simd, sse>)
					{
						uint64_t low_index = _tzcnt_u64(p_mask) & ~3ULL;
						const uint64_t high_index = 64ULL - _lzcnt_u64(p_mask);

						const __m128i INDICES = _mm_setr_epi32(1, 2, 4, 8);

						for (; low_index < high_index; low_index += 4ULL)
						{
							__m128i* const data = reinterpret_cast<__m128i*>(base + low_index);

							__m128i GENERATIONS = _mm_load_si128(data);

							const __m128i SUBTRACTION_MASK = _mm_cmpeq_epi32(_mm_and_si128(_mm_set1_epi32(static_cast<int>(p_mask >> low_index)), INDICES), INDICES);
							GENERATIONS = _mm_sub_epi32(GENERATIONS, SUBTRACTION_MASK);

							_mm_store_si128(data, GENERATIONS);
						}
					}
					else if constexpr (std::same_as<t_simd, avx2>)
					{
						uint64_t low_index = _tzcnt_u64(p_mask) & ~7ULL;
						const uint64_t high_index = 64ULL - _lzcnt_u64(p_mask);

						const __m256i INDICES = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

						for (; low_index < high_index; low_index += 8ULL)
						{
							__m256i* const data = reinterpret_cast<__m256i*>(base + low_index);

							__m256i GENERATIONS = _mm256_load_si256(data);

							const __m256i SUBTRACTION_MASK = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(static_cast<int>(p_mask >> low_index)), INDICES), INDICES);
							GENERATIONS = _mm256_sub_epi32(GENERATIONS, SUBTRACTION_MASK);

							_mm256_store_si256(data, GENERATIONS);
						}
					}
					else if constexpr (std::same_as<t_simd, avx512>)
					{
						uint64_t low_index = _tzcnt_u64(p_mask) & ~15ULL;
						const uint64_t high_index = 64ULL - _lzcnt_u64(p_mask);

						for (; low_index < high_index; low_index += 16ULL)
						{
							void* const data = base + low_index;

							__m512i GENERATIONS = _mm512_load_epi32(data);

							const __m512i SUBTRACTION_MASK = _mm512_movm_epi32(static_cast<unsigned short>(p_mask >> low_index));
							GENERATIONS = _mm512_sub_epi32(GENERATIONS, SUBTRACTION_MASK);

							_mm512_store_epi32(data, GENERATIONS);
						}
					}

					if (word == UINT64_MAX) [[unlikely]]
					{
						::new(m_free_table + p_index) uint32_t(m_free_list);
						m_free_list = p_index;
					}
				}
			}
		}


		template<iteration_concept t_iteration, iteration_transform_concept t_iteration_transform, class t_func, class... Args>
		SCW_FORCE_INLINE void visit_all_caller_(t_func&& p_func, uint32_t p_start_chunk, uint32_t p_end_chunk, Args&&... p_args) noexcept
		{
			visit_all_<t_iteration, t_iteration_transform>(*this, std::forward<t_func>(p_func), p_start_chunk, p_end_chunk, std::forward<Args>(p_args)...);
		}


		template<iteration_concept t_iteration, iteration_transform_concept t_iteration_transform, class t_func, class... Args>
		SCW_FORCE_INLINE void visit_all_caller_(t_func&& p_func, uint32_t p_start_chunk, uint32_t p_end_chunk, Args&&... p_args) const noexcept
		{
			visit_all_<t_iteration, t_iteration_transform>(*this, std::forward<t_func>(p_func), p_start_chunk, p_end_chunk, std::forward<Args>(p_args)...);
		}


		template<iteration_concept t_iteration, iteration_transform_concept t_iteration_transform, class self, class t_func, class... Args>
		SCW_FORCE_INLINE static void visit_all_(self& p_self, t_func&& p_func, uint32_t p_start_chunk, uint32_t p_end_chunk, Args&&... p_args) noexcept
		{
			constexpr static bool c_for_each_transform = std::same_as<t_iteration_transform, for_each_transform>;
			constexpr static bool c_erase_if_transform = std::same_as<t_iteration_transform, erase_if_transform>;

			size_t start = static_cast<size_t>(p_start_chunk << 6U);
			const size_t end = p_end_chunk == UINT32_MAX ? static_cast<size_t>(p_self.m_high_water_mark & ~63U) : static_cast<size_t>(p_end_chunk << 6U);
			uint64_t* word_pointer = p_self.m_skip_data + p_start_chunk;
			uint64_t word = *word_pointer;
			uint64_t offset = 0ULL;

			while (start < end) [[likely]]
			{
				if constexpr (std::same_as<t_iteration, dense_iterate>)
				{
					while (word == UINT64_MAX)
					{
						for (offset = 0ULL; offset != 64ULL; ++offset)
						{
							const bool result = p_self.invoke_user_callback_(c_indices, p_self.m_data, p_func, start, offset, p_args...);

							if constexpr (c_for_each_transform)
							{
								if (!result)
								{
									return;
								}
							}
							else if constexpr (c_erase_if_transform)
							{
								if (result)
								{
									p_self.erase(static_cast<uint32_t>(start + offset));
								}
							}
						}

						start += 64ULL;
						++word_pointer;
						word = *word_pointer;

						if (start == end)
						{
							goto VISIT_ALL_TAIL;
						}
					}
				}

#ifdef __GNUC__
				if (word)
				{
					do
					{
						offset = _tzcnt_u64(word);
						word = _blsr_u64(word);

						const bool result = p_self.invoke_user_callback_(c_indices, p_self.m_data, p_func, start, offset, p_args...);

						if constexpr (c_for_each_transform)
						{
							if (!result)
							{
								return;
							}
						}
						else if constexpr (c_erase_if_transform)
						{
							if (result)
							{
								p_self.erase(static_cast<uint32_t>(start + offset));
							}
						}

						bool test;
						__asm__("test %1,%1" : "=@ccz"(test) : "r"(word));

						if (test)
						{
							break;
						}
					} while (true);
				}
#else
				while (word)
				{
					offset = _tzcnt_u64(word);
					word = _blsr_u64(word);

					const bool result = p_self.invoke_user_callback_(c_indices, p_self.m_data, p_func, start, offset, p_args...);

					if constexpr (c_for_each_transform)
					{
						if (!result)
						{
							return;
						}
					}
					else if constexpr (c_erase_if_transform)
					{
						if (result)
						{
							p_self.erase(static_cast<uint32_t>(start + offset));
						}
					}
				}
#endif

				do
				{
					start += 64ULL;
					++word_pointer;
					word = *word_pointer;
				} while (!word);
			}

		VISIT_ALL_TAIL:
			if (p_end_chunk != UINT32_MAX)
			{
				return;
			}

			offset = _tzcnt_u64(word);
			const uint64_t end_offset = static_cast<uint64_t>(p_self.m_high_water_mark & 63U);

			while (offset != end_offset)
			{
				const bool result = p_self.invoke_user_callback_(c_indices, p_self.m_data, p_func, start, offset, p_args...);

				if constexpr (c_for_each_transform)
				{
					if (!result)
					{
						return;
					}
				}
				else if constexpr (c_erase_if_transform)
				{
					if (result)
					{
						p_self.erase(static_cast<uint32_t>(start + offset));
					}
				}

				word = _blsr_u64(word);
				offset = _tzcnt_u64(word);
			}
		}


		template<size_t... t_indices, class t_func, class... Args>
		SCW_FORCE_INLINE [[nodiscard]] bool invoke_user_callback_(std::index_sequence<t_indices...>, void* const* m_data, t_func& p_func, size_t start, uint64_t offset, Args&... p_args) noexcept
		{
			return p_func(static_cast<uint32_t>(start + offset), static_cast<type_<t_indices>*>(m_data[t_indices])[start + offset]..., p_args...);
		}


		template<size_t... t_indices, class t_func, class... Args>
		SCW_FORCE_INLINE [[nodiscard]] bool invoke_user_callback_(std::index_sequence<t_indices...>, const void* const* m_data, t_func& p_func, size_t start, uint64_t offset, Args&... p_args) const noexcept
		{
			return p_func(static_cast<uint32_t>(start + offset), static_cast<const type_<t_indices>*>(m_data[t_indices])[start + offset]..., p_args...);
		}


		[[nodiscard]] constexpr static size_t get_bytes_for_element_count_(uint32_t p_count, size_t p_size) noexcept
		{
			return static_cast<size_t>(p_count) * p_size;
		}


		[[nodiscard]] constexpr static size_t get_skip_bytes_for_element_count_(uint32_t p_count) noexcept
		{
			return align_(static_cast<size_t>(p_count) + 1ULL, 64ULL) >> 3ULL;
		}


		[[nodiscard]] constexpr static size_t get_generation_bytes_for_element_count_(uint32_t p_count) noexcept
		{
			if constexpr (c_generational)
			{
				return align_(static_cast<size_t>(p_count), 64ULL) * sizeof(uint32_t);
			}
			else
			{
				return 0ULL;
			}
		}


		[[nodiscard]] constexpr static size_t get_free_table_bytes_for_element_count_(uint32_t p_count) noexcept
		{
			return (align_(static_cast<size_t>(p_count) + 1ULL, 64ULL) >> 6ULL) * sizeof(uint32_t);
		}

		// gets page size and shift amount once in the program's run time
		[[nodiscard]] static bool initialize_reserve_sizes_(size_t p_aligned_data_bytes, size_t p_skip_array_bytes, size_t p_generation_bytes, size_t p_free_table_bytes)
		{
			platform::initialize_system_page_data();

			sm_reserved_bytes = align_(p_aligned_data_bytes + p_skip_array_bytes + p_generation_bytes + p_free_table_bytes, platform::os_page_size);

			return false;
		}

	private: // MEMBERS
		inline static size_t sm_reserved_bytes;

		void* m_data[c_types_count] = {};
		uint64_t* m_skip_data = nullptr;
		SCW_NO_UNIQUE_ADDRESS std::conditional_t<c_generational, uint32_t*, no_generations> m_generations = {};
		uint32_t* m_free_table = nullptr;
		uint32_t m_free_list = UINT32_MAX;
		uint32_t m_capacity = 0U;
		uint32_t m_high_water_mark = 0U;
		uint32_t m_size = 0U;
	};



	// ITERATOR
	template<class T, const_iterator_concept t_is_const>
	class soa_map_iterator
	{
	private:
		template<uint32_t, use_generations_concept, class...>
		friend class soa_map;


		constexpr static bool c_constant = std::same_as<t_is_const, is_const>;


		using return_value = std::conditional_t<c_constant, const T, T>;
		using data_value_type = return_value*;
		using skip_value_type = std::conditional_t<c_constant, const uint64_t*, uint64_t*>;

	public:
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using iterator_category = std::forward_iterator_tag;
	public:
		soa_map_iterator() noexcept = default;

		soa_map_iterator(data_value_type p_data, skip_value_type p_skip_pointer_base, uint64_t p_word, uint32_t p_skip_offset, uint32_t p_offset) noexcept :
			m_data(p_data), m_skip_pointer_base(p_skip_pointer_base), m_word(p_word), m_skip_offset(p_skip_offset), m_offset(p_offset) {}

	public:
		soa_map_iterator& operator++() noexcept
		{
			while (!m_word) [[unlikely]]
			{
				m_data += 64ULL;
				m_skip_offset += 64U;
				m_word = m_skip_pointer_base[m_skip_offset >> 6U];
			}

			m_offset = static_cast<uint32_t>(_tzcnt_u64(m_word));
			m_word = _blsr_u64(m_word);

			return *this;
		}


		soa_map_iterator operator++(int) noexcept
		{
			const soa_map_iterator other{ *this };
			++*this;

			return other;
		}


		[[nodiscard]] return_value& operator*() const noexcept
		{
			return m_data[m_offset];
		}


		[[nodiscard]] return_value* operator->() const noexcept
		{
			return &m_data[m_offset];
		}


		bool operator==(const soa_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset == p_other.m_skip_offset + p_other.m_offset; }
		bool operator!=(const soa_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset != p_other.m_skip_offset + p_other.m_offset; }
		bool operator>(const soa_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset > p_other.m_skip_offset + p_other.m_offset; }
		bool operator<(const soa_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset < p_other.m_skip_offset + p_other.m_offset; }
		bool operator>=(const soa_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset >= p_other.m_skip_offset + p_other.m_offset; }
		bool operator<=(const soa_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset <= p_other.m_skip_offset + p_other.m_offset; }

	private:
		data_value_type m_data;
		skip_value_type m_skip_pointer_base;
		uint64_t m_word;
		uint32_t m_skip_offset;
		uint32_t m_offset;
	};



	// REMAP TABLE, simplified, never needs to grow or make checks
	template<class Allocator>
	class remap_table
	{
	private:
		remap_table() noexcept = default;

	public:
		remap_table(const remap_table& p_other) : m_state(p_other.m_state)
		{
			if (m_state.data)
			{
				m_state.data = m_state.allocate(m_state.size + m_state.bitset_size + 1ULL);
				m_state.offset_data = m_state.data - p_other.m_state.offset;
				m_state.deletion_data = reinterpret_cast<uint64_t*>(align_(reinterpret_cast<size_t>(m_state.data + m_state.size), 8ULL));
				memcpy(m_state.data, p_other.m_state.data, (m_state.size + m_state.bitset_size + 1ULL) * sizeof(uint32_t));
				memcpy(m_state.deletion_data, p_other.m_state.deletion_data, m_state.bitset_size * sizeof(uint32_t));
			}
		}


		remap_table& operator=(const remap_table& p_other)
		{
			if (this != &p_other)
			{
				CompressedState new_state = p_other.m_state;

				if (new_state.data)
				{
					new_state.data = new_state.allocate(new_state.size + new_state.bitset_size + 1ULL);
					new_state.deletion_data = reinterpret_cast<uint64_t*>(align_(reinterpret_cast<size_t>(new_state.data + new_state.size), 8ULL));
					new_state.offset_data = new_state.data - p_other.m_state.offset;

					deallocate_();
					memcpy(new_state.data, p_other.m_state.data, (new_state.size + new_state.bitset_size + 1ULL) * sizeof(uint32_t));
					memcpy(new_state.deletion_data, p_other.m_state.deletion_data, new_state.bitset_size * sizeof(uint32_t));
				}

				m_state = std::move(new_state);
			}

			return *this;
		}


		remap_table(remap_table&& p_other) noexcept : m_state(std::move(p_other.m_state))
		{
			p_other.m_state.data = nullptr;
			p_other.m_state.deletion_data = nullptr;
			p_other.m_state.offset_data = nullptr;
			p_other.m_state.offset = 0ULL;
			p_other.m_state.size = 0ULL;
			p_other.m_state.bitset_size = 0ULL;
		}


		remap_table& operator=(remap_table&& p_other) noexcept
		{
			if (this != &p_other)
			{
				deallocate_();

				m_state = std::move(p_other.m_state);

				p_other.m_state.data = nullptr;
				p_other.m_state.deletion_data = nullptr;
				p_other.m_state.offset_data = nullptr;
				p_other.m_state.offset = 0ULL;
				p_other.m_state.size = 0ULL;
				p_other.m_state.bitset_size = 0ULL;
			}

			return *this;
		}


		~remap_table() noexcept
		{
			deallocate_();
		}

	public:
		bool remap(uint32_t& p_old_handle) const noexcept
		{
			const bool was_alive = get_bit_(m_state.deletion_data, p_old_handle);

			if (p_old_handle >= m_state.offset)
			{
				p_old_handle = m_state.offset_data[p_old_handle];
			}

			return was_alive;
		}


		bool branchless_remap(uint32_t& p_old_handle) const noexcept
		{
			const bool was_alive = get_bit_(m_state.deletion_data, p_old_handle);

			p_old_handle = branchless_remap_(p_old_handle);

			return was_alive;
		}


		[[nodiscard]] bool is_empty() const noexcept
		{
			return !m_state.size;
		}

	private:
		void allocate_(const Allocator& p_allocator, const uint64_t* p_deletion_data, size_t p_element_count, size_t p_high_water_mark)
		{
			m_state = CompressedState{ p_allocator };

			m_state.size = p_high_water_mark - p_element_count;
			m_state.bitset_size = (((p_high_water_mark - 1ULL >> 6ULL) + 1ULL) * 2ULL);
			m_state.data = m_state.allocate(m_state.size + m_state.bitset_size + 1ULL);
			m_state.offset_data = m_state.data - p_element_count;
			m_state.deletion_data = reinterpret_cast<uint64_t*>(align_(reinterpret_cast<size_t>(m_state.data + m_state.size), 8ULL));
			m_state.offset = p_element_count;

			memcpy(m_state.deletion_data, p_deletion_data, m_state.bitset_size * sizeof(uint32_t));
		}


		void deallocate_() noexcept
		{
			if (m_state.data)
			{
				m_state.deallocate(m_state.data, m_state.size + m_state.bitset_size + 1ULL);
			}
		}


		void insert_(uint32_t p_old_handle, uint32_t p_new_handle) noexcept
		{
			m_state.offset_data[p_old_handle] = p_new_handle;
		}


		SCW_FORCE_INLINE [[nodiscard]] uint32_t branchless_remap_(uint32_t p_old_handle) const noexcept
		{
			const uint32_t in_bounds = p_old_handle >= m_state.offset;
			const uint32_t mask = 0U - in_bounds;
			const uint32_t remap_handle = m_state.data[(p_old_handle - m_state.offset) & mask];

			return in_bounds ? remap_handle : p_old_handle;
		}

	private:
		template<uint32_t, use_generations_concept, class...>
		friend class soa_map;


		struct CompressedState : public Allocator
		{
			uint32_t* data = nullptr;
			uint32_t* offset_data = nullptr;
			uint64_t* deletion_data = nullptr;
			size_t offset = 0ULL;
			size_t size = 0ULL;
			size_t bitset_size = 0ULL;
		};

		CompressedState m_state;
	};
}


#ifdef SCW_MAP_PLATFORM
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif


namespace scw
{
	namespace platform
	{
#ifdef _WIN32
		[[nodiscard]] size_t get_page_size() noexcept
		{
			SYSTEM_INFO system_info;
			GetSystemInfo(&system_info);

			return static_cast<size_t>(system_info.dwPageSize);
		}


		[[nodiscard]] void* reserve(size_t p_size) noexcept
		{
			return VirtualAlloc(NULL, p_size, MEM_RESERVE, PAGE_READWRITE);
		}


		[[nodiscard]] bool commit(void* p_address, size_t p_size) noexcept
		{
			return VirtualAlloc(p_address, p_size, MEM_COMMIT, PAGE_READWRITE);
		}


		[[nodiscard]] bool free(void* p_address, size_t p_size) noexcept
		{
			return VirtualFree(p_address, 0ULL, MEM_RELEASE);
		}


		[[nodiscard]] bool decommit(void* p_address, size_t p_size) noexcept
		{
			return VirtualFree(p_address, p_size, MEM_DECOMMIT);
		}

#else
		[[nodiscard]] size_t get_page_size() noexcept
		{
			return static_cast<size_t>(getpagesize());
		}


		[[nodiscard]] void* reserve(size_t p_size) noexcept
		{
			void* reserve_region = mmap(nullptr, p_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);

			return reserve_region == MAP_FAILED ? nullptr : reserve_region;
		}


		[[nodiscard]] bool commit(void* p_address, size_t p_size) noexcept
		{
			return true;
		}


		[[nodiscard]] bool free(void* p_address, size_t p_size) noexcept
		{
			return !munmap(p_address, p_size);
		}


		[[nodiscard]] bool decommit(void* p_address, size_t p_size) noexcept
		{
			return !madvise(p_address, p_size, MADV_DONTNEED);
		}
#endif
	}
}
#endif
