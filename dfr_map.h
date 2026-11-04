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
#elif defined(__GNUC__)
#define SCW_FORCE_INLINE [[gnu::always_inline]]
#define SCW_NO_INLINE [[gnu::noinline]]
#elif defined(_MSC_VER)
#define SCW_FORCE_INLINE [[msvc::forceinline]]
#define SCW_NO_INLINE [[msvc::noinline]]
#else
#define SCW_FORCE_INLINE
#define SCW_NO_INLINE
#endif


namespace scw
{
	// CONCEPTS
	struct is_const {};
	struct not_const {};
	struct dense_iterate {};
	struct sparse_iterate {};
	struct blind_iterate {};
	struct return_table {};
	struct no_table {};

	struct for_each_transform {};
	struct erase_if_transform {};


	template<class T>
	concept is_trivial_concept = std::is_trivially_copyable_v<T> && std::is_trivially_copy_constructible_v<T> && std::is_trivially_copy_assignable_v<T> && std::is_trivially_destructible_v<T>;

	template<class T>
	concept const_iterator_concept = std::same_as<T, is_const> || std::same_as<T, not_const>;

	template<class T>
	concept iteration_concept = std::same_as<T, dense_iterate> || std::same_as<T, sparse_iterate> || std::same_as<T, blind_iterate>;

	template<class T>
	concept return_remap_table_concept = std::same_as<T, return_table> || std::same_as<T, no_table>;

	template<class T>
	concept iteration_transform_concept = std::same_as<T, for_each_transform> || std::same_as<T, erase_if_transform>;


	template<class, const_iterator_concept>
	class dfr_map_iterator;

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
	template<uint32_t t_vm_reserve_elements, class... t_types>
	class dfr_map
	{
	private:
		static_assert((is_trivial_concept<t_types> && ...), "scw::dfr_map requires all types to be trivial");
		static_assert(sizeof...(t_types) > 0, "scw:dfr_map requires at least one type");
		static_assert(t_vm_reserve_elements&& t_vm_reserve_elements < UINT32_MAX, "scw::dfr_map requires reserve size to be between 1 and uint32_t max - 1");

	private: // MEMBER ALIASES
		template<size_t t_index>
		using type_ = std::tuple_element_t<t_index, std::tuple<t_types...>>;


		constexpr static size_t c_types_count = sizeof...(t_types);
		constexpr static size_t c_last_type_index = c_types_count - 1ULL;
		constexpr static auto c_indices = std::make_index_sequence<c_types_count>{};


		using types_array = std::array<size_t, c_types_count>;

	public:
		using handle = uint32_t;

		template<class T>
		using iterator = dfr_map_iterator<T, not_const>;

		template<class T>
		using const_iterator = dfr_map_iterator<T, is_const>;

	public: // CONSTRUCTORS
		dfr_map()
		{
			allocate_(1U);
		}


		explicit dfr_map(uint32_t p_reserve_count)
		{
			allocate_(p_reserve_count);
		}


		dfr_map(uint32_t p_element_count, const t_types&... p_values)
		{
			allocate_(p_element_count);

			m_high_water_mark = p_element_count;

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


		dfr_map(const dfr_map& p_other)
		{
			copy_dfr_map_(p_other);
		}


		dfr_map& operator=(const dfr_map& p_other)
		{
			if (this != &p_other)
			{
				dfr_map temp(p_other);
				std::swap(*this, temp);
			}

			return *this;
		}


		dfr_map(dfr_map&& p_other) noexcept
		{
			steal_other_(std::move(p_other));
		}


		dfr_map& operator=(dfr_map&& p_other) noexcept
		{
			if (this != &p_other)
			{
				deallocate_();

				steal_other_(std::move(p_other));
			}

			return *this;
		}


		~dfr_map() noexcept
		{
			deallocate_();
		}

	public: // MEMBER FUNCTIONS
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


		void erase(uint32_t p_index) noexcept
		{
			destroy_element_(p_index);
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


		void try_erase(uint32_t p_index) noexcept
		{
			if (is_alive(p_index))
			{
				destroy_element_(p_index);
			}
		}


		void erase_chunk(uint32_t p_index, uint64_t p_mask) noexcept
		{
			destroy_chunk_(p_index, p_mask);
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
		[[nodiscard]] type_<t_index>* try_at(uint32_t p_index) noexcept
		{
			if (is_alive(p_index))
			{
				return static_cast<type_<t_index>*>(m_data[t_index]) + p_index;
			}

			return nullptr;
		}


		template<size_t t_index>
		[[nodiscard]] const type_<t_index>* try_at(uint32_t p_index) const noexcept
		{
			if (is_alive(p_index))
			{
				return static_cast<type_<t_index>*>(m_data[t_index]) + p_index;
			}

			return nullptr;
		}


		[[nodiscard]] bool is_alive(uint32_t p_index) const noexcept
		{
			return get_bit_(m_skip_data, p_index);
		}


		template<size_t t_index>
		[[nodiscard]] handle get_handle_from_iterator(const iterator<type_<t_index>>& p_iterator) const noexcept
		{
			return p_iterator.m_skip_offset + p_iterator.m_offset;
		}


		template<size_t t_index>
		[[nodiscard]] handle get_handle_from_iterator(const const_iterator<type_<t_index>>& p_iterator) const noexcept
		{
			return p_iterator.m_skip_offset + p_iterator.m_offset;
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
			return !m_high_water_mark;
		}


		[[nodiscard]] uint32_t size() const noexcept
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

			if (m_high_water_mark)
			{
				const uint32_t high_index = m_high_water_mark - 1U;
				const uint32_t high_word_index = high_index >> 6U;

				uint32_t element_count = 0U;
				for (uint32_t current_index = 0U; current_index != high_word_index; ++current_index)
				{
					element_count += static_cast<uint32_t>(_mm_popcnt_u64(m_skip_data[current_index]));
				}

				const uint64_t high_word_shift_amount = _andn_u64(static_cast<uint64_t>(high_index), 63ULL);
				element_count += static_cast<uint32_t>(_mm_popcnt_u64(m_skip_data[high_word_index] & UINT64_MAX >> high_word_shift_amount));

				if (element_count)
				{
					const uint32_t max_index = element_count - 1U;
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
							table.allocate_(p_allocator, m_skip_data, static_cast<size_t>(element_count), static_cast<size_t>(m_high_water_mark));
						}

						uint64_t* hole_skip_data = m_skip_data;
						uint64_t current_holes_word = ~*hole_skip_data;
						size_t hole_start = 0ULL;
						uint64_t hole_offset = 0ULL;

						uint64_t* element_skip_data = m_skip_data + (element_count >> 6U);
						uint64_t current_elements_word = *element_skip_data & UINT64_MAX << static_cast<uint64_t>(element_count & 63U);
						size_t element_start = static_cast<size_t>(element_count & ~63U);
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
				m_high_water_mark = element_count;
			}

			memset(m_skip_data, 0xFF, get_skip_bytes_for_element_count_(m_capacity));

			return table;
		}

		// only call after compress
		void shrink_to_fit() noexcept
		{
			decommit_pages_(m_high_water_mark);
		}


		void clear() noexcept
		{
			memset(m_skip_data, 0xFF, static_cast<size_t>((m_high_water_mark >> 6U) + 1U) * sizeof(uint64_t));

			m_high_water_mark = 0U;
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

			constexpr static size_t c_aligned_skip_array_bytes = get_skip_bytes_for_element_count_(t_vm_reserve_elements);
			const size_t skip_reserve_size = get_skip_bytes_for_element_count_(p_reserve_count);

			[[maybe_unused]] static const bool _ = initialize_reserve_sizes_(c_aligned_data_bytes, c_aligned_skip_array_bytes);

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

			memset(m_skip_data, 0xFF, skip_reserve_size);

			m_capacity = p_reserve_count;
		}


		void allocate_fail_()
		{
			deallocate_();

			throw std::bad_alloc();
		}


		void copy_dfr_map_(const dfr_map& p_other)
		{
			m_high_water_mark = p_other.m_high_water_mark;

			allocate_(m_high_water_mark);

			[] <size_t... t_indices>(std::index_sequence<t_indices...>, void* const* m_data, uint32_t m_high_water_mark, const dfr_map & p_other) noexcept
			{
				(memcpy(m_data[t_indices], p_other.m_data[t_indices], static_cast<size_t>(m_high_water_mark) * sizeof(type_<t_indices>)), ...);
			}(c_indices, m_data, m_high_water_mark, p_other);

			memcpy(m_skip_data, p_other.m_skip_data, get_skip_bytes_for_element_count_(m_high_water_mark));
		}


		void steal_other_(dfr_map&& p_other) noexcept
		{
			memcpy(m_data, p_other.m_data, c_types_count * sizeof(void*));
			m_skip_data = p_other.m_skip_data;
			m_capacity = p_other.m_capacity;
			m_high_water_mark = p_other.m_high_water_mark;

			memset(p_other.m_data, 0, c_types_count * sizeof(void*));
			p_other.m_skip_data = nullptr;
			p_other.m_capacity = 0U;
			p_other.m_high_water_mark = 0U;
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

			m_capacity = p_index + 1U;
		}


		SCW_FORCE_INLINE [[nodiscard]] uint32_t get_end_allocation_slot_()
		{
			const uint32_t slot = m_high_water_mark;

			if (slot + 1U >= m_capacity)
			{
				grow_(m_capacity);
			}

			++m_high_water_mark;

			return slot;
		}


		SCW_FORCE_INLINE [[nodiscard]] uint32_t get_unchecked_allocation_slot_() noexcept
		{
			const uint32_t slot = m_high_water_mark;

			++m_high_water_mark;

			return slot;
		}


		SCW_FORCE_INLINE handle construct_in_slot_(uint32_t p_slot, const t_types&... p_values) noexcept
		{
			[] <size_t... t_indices>(std::index_sequence<t_indices...>, void* const* m_data, uint32_t p_slot, const auto&... p_values) noexcept
			{
				(::new(reinterpret_cast<type_<t_indices>*>(m_data[t_indices]) + p_slot) type_<t_indices>(p_values), ...);
			}(c_indices, m_data, p_slot, p_values...);

			return p_slot;
		}

		// compared to a per element intrusive free list, this design is slower computationally but reduces memory accesses drastically
		// should be slightly slower in hot cache and much faster when not hot in cache
		SCW_FORCE_INLINE void destroy_element_(uint32_t p_index) noexcept
		{
			const uint32_t chunk_index = p_index >> 6U;
			const uint64_t word = m_skip_data[chunk_index];

			m_skip_data[chunk_index] = _andn_u64(1ULL << (p_index & 63U), word);
		}


		SCW_FORCE_INLINE void destroy_chunk_(uint32_t p_index, uint64_t p_mask) noexcept
		{
			const uint64_t word = m_skip_data[p_index];

			m_skip_data[p_index] = _andn_u64(p_mask, word);
		}


		template<iteration_concept t_iteration, iteration_transform_concept t_iteration_transform, class t_func, class... Args>
		SCW_FORCE_INLINE void visit_all_caller_(t_func&& p_func, uint32_t p_start_chunk, uint32_t p_end_chunk, Args&&... p_args) noexcept
		{
			if constexpr (std::same_as<t_iteration, blind_iterate>)
			{
				blind_iterate_<t_iteration, t_iteration_transform>(*this, std::forward<t_func>(p_func), p_start_chunk, p_end_chunk, std::forward<Args>(p_args)...);
			}
			else
			{
				visit_all_<t_iteration, t_iteration_transform>(*this, std::forward<t_func>(p_func), p_start_chunk, p_end_chunk, std::forward<Args>(p_args)...);
			}
		}


		template<iteration_concept t_iteration, iteration_transform_concept t_iteration_transform, class t_func, class... Args>
		SCW_FORCE_INLINE void visit_all_caller_(t_func&& p_func, uint32_t p_start_chunk, uint32_t p_end_chunk, Args&&... p_args) const noexcept
		{
			if constexpr (std::same_as<t_iteration, blind_iterate>)
			{
				blind_iterate_<t_iteration, t_iteration_transform>(*this, std::forward<t_func>(p_func), p_start_chunk, p_end_chunk, std::forward<Args>(p_args)...);
			}
			else
			{
				visit_all_<t_iteration, t_iteration_transform>(*this, std::forward<t_func>(p_func), p_start_chunk, p_end_chunk, std::forward<Args>(p_args)...);
			}
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
			uint64_t erase_word = 0ULL;

			while (start < end) [[likely]]
			{
				if constexpr (std::same_as<t_iteration, dense_iterate>)
				{
					while (word == UINT64_MAX)
					{
						for (offset = 0ULL; offset != 64ULL; ++offset)
						{
							const bool result = p_self.invoke_user_callback_(c_indices, p_self.m_data, p_func, start + offset, p_args...);

							if constexpr (c_for_each_transform)
							{
								if (!result)
								{
									return;
								}
							}
							else if constexpr (c_erase_if_transform)
							{
								erase_word |= static_cast<uint64_t>(result) << offset;
							}
						}

						if constexpr (c_erase_if_transform)
						{
							p_self.destroy_chunk_(static_cast<uint32_t>(start >> 6ULL), erase_word);
							erase_word = 0ULL;
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

						const bool result = p_self.invoke_user_callback_(c_indices, p_self.m_data, p_func, start + offset, p_args...);

						if constexpr (c_for_each_transform)
						{
							if (!result)
							{
								return;
							}
						}
						else if constexpr (c_erase_if_transform)
						{
							erase_word |= static_cast<uint64_t>(result) << offset;
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

					const bool result = p_self.invoke_user_callback_(c_indices, p_self.m_data, p_func, start + offset, p_args...);

					if constexpr (c_for_each_transform)
					{
						if (!result)
						{
							return;
						}
					}
					else if constexpr (c_erase_if_transform)
					{
						erase_word |= static_cast<uint64_t>(result) << offset;
					}
				}
#endif
				if constexpr (c_erase_if_transform)
				{
					p_self.destroy_chunk_(static_cast<uint32_t>(start >> 6ULL), erase_word);
					erase_word = 0ULL;
				}

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
				const bool result = p_self.invoke_user_callback_(c_indices, p_self.m_data, p_func, start + offset, p_args...);

				if constexpr (c_for_each_transform)
				{
					if (!result)
					{
						return;
					}
				}
				else if constexpr (c_erase_if_transform)
				{
					erase_word |= static_cast<uint64_t>(result) << offset;
				}

				word = _blsr_u64(word);
				offset = _tzcnt_u64(word);
			}

			if constexpr (c_erase_if_transform)
			{
				p_self.destroy_chunk_(static_cast<uint32_t>(start >> 6ULL), erase_word);
			}
		}


		template<iteration_concept t_iteration, iteration_transform_concept t_iteration_transform, class self, class t_func, class... Args>
		SCW_FORCE_INLINE static void blind_iterate_(self& p_self, t_func&& p_func, uint32_t p_start_chunk, uint32_t p_end_chunk, Args&&... p_args) noexcept
		{
			size_t start = static_cast<size_t>(p_start_chunk << 6U);
			const size_t end = p_end_chunk == UINT32_MAX ? p_self.m_high_water_mark : static_cast<size_t>(p_end_chunk << 6U);

			if constexpr (std::same_as<t_iteration_transform, for_each_transform>)
			{
				for (; start != end; ++start)
				{
					if (!p_self.invoke_user_callback_(c_indices, p_self.m_data, p_func, start, p_args...))
					{
						return;
					}
				}
			}
			else if constexpr (std::same_as<t_iteration_transform, erase_if_transform>)
			{
				const size_t chunks_end = p_end_chunk == UINT32_MAX ? static_cast<size_t>(p_self.m_high_water_mark & ~63U) : static_cast<size_t>(p_end_chunk << 6U);
				uint64_t erase_word = 0ULL;

				for (; start != chunks_end; start += 64ULL)
				{
					for (uint32_t offset = 0ULL; offset != 64ULL; ++offset)
					{
						erase_word |= static_cast<uint64_t>(p_self.invoke_user_callback_(c_indices, p_self.m_data, p_func, start + offset, p_args...)) << offset;
					}

					const uint32_t chunk_index = static_cast<uint32_t>(start >> 6ULL);
					p_self.destroy_chunk_(chunk_index, erase_word & p_self.get_bitmask(chunk_index));
					erase_word = 0ULL;
				}

				for (; start != end; ++start)
				{
					erase_word |= static_cast<uint64_t>(p_self.invoke_user_callback_(c_indices, p_self.m_data, p_func, start, p_args...)) << (start & 63U);
				}

				const uint32_t chunk_index = static_cast<uint32_t>(start >> 6ULL);
				p_self.destroy_chunk_(chunk_index, erase_word & p_self.get_bitmask(chunk_index));
			}
		}


		template<size_t... t_indices, class t_func, class... Args>
		SCW_FORCE_INLINE [[nodiscard]] bool invoke_user_callback_(std::index_sequence<t_indices...>, void* const* m_data, t_func& p_func, size_t p_index, Args&... p_args) noexcept
		{
			return p_func(static_cast<uint32_t>(p_index), static_cast<type_<t_indices>*>(m_data[t_indices])[p_index]..., p_args...);
		}


		template<size_t... t_indices, class t_func, class... Args>
		SCW_FORCE_INLINE [[nodiscard]] bool invoke_user_callback_(std::index_sequence<t_indices...>, const void* const* m_data, t_func& p_func, size_t p_index, Args&... p_args) const noexcept
		{
			return p_func(static_cast<uint32_t>(p_index), static_cast<const type_<t_indices>*>(m_data[t_indices])[p_index]..., p_args...);
		}


		[[nodiscard]] constexpr static size_t get_bytes_for_element_count_(uint32_t p_count, size_t p_size) noexcept
		{
			return static_cast<size_t>(p_count) * p_size;
		}


		[[nodiscard]] constexpr static size_t get_skip_bytes_for_element_count_(uint32_t p_count) noexcept
		{
			return align_(static_cast<size_t>(p_count) + 1ULL, 64ULL) >> 3ULL;
		}

		// gets page size and shift amount once in the program's run time
		[[nodiscard]] static bool initialize_reserve_sizes_(size_t p_aligned_data_bytes, size_t p_skip_array_bytes)
		{
			platform::initialize_system_page_data();

			sm_reserved_bytes = align_(p_aligned_data_bytes + p_skip_array_bytes, platform::os_page_size);

			return false;
		}

	private: // MEMBERS
		inline static size_t sm_reserved_bytes;

		void* m_data[c_types_count] = {};
		uint64_t* m_skip_data = nullptr;
		uint32_t m_capacity = 0U;
		uint32_t m_high_water_mark = 0U;
	};



	// ITERATOR
	template<class T, const_iterator_concept t_is_const>
	class dfr_map_iterator
	{
	private:
		template<uint32_t, class...>
		friend class dfr_map;


		constexpr static bool c_constant = std::same_as<t_is_const, is_const>;


		using return_value = std::conditional_t<c_constant, const T, T>;
		using data_value_type = return_value*;
		using skip_value_type = std::conditional_t<c_constant, const uint64_t*, uint64_t*>;

	public:
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using iterator_category = std::forward_iterator_tag;
	public:
		dfr_map_iterator() noexcept = default;

		dfr_map_iterator(data_value_type p_data, skip_value_type p_skip_pointer_base, uint64_t p_word, uint32_t p_skip_offset, uint32_t p_offset) noexcept :
			m_data(p_data), m_skip_pointer_base(p_skip_pointer_base), m_word(p_word), m_skip_offset(p_skip_offset), m_offset(p_offset) {}

	public:
		dfr_map_iterator& operator++() noexcept
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


		dfr_map_iterator operator++(int) noexcept
		{
			const dfr_map_iterator other{ *this };
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


		bool operator==(const dfr_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset == p_other.m_skip_offset + p_other.m_offset; }
		bool operator!=(const dfr_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset != p_other.m_skip_offset + p_other.m_offset; }
		bool operator>(const dfr_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset > p_other.m_skip_offset + p_other.m_offset; }
		bool operator<(const dfr_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset < p_other.m_skip_offset + p_other.m_offset; }
		bool operator>=(const dfr_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset >= p_other.m_skip_offset + p_other.m_offset; }
		bool operator<=(const dfr_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset <= p_other.m_skip_offset + p_other.m_offset; }

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

					memcpy(new_state.data, p_other.m_state.data, (new_state.size + new_state.bitset_size + 1ULL) * sizeof(uint32_t));
					memcpy(new_state.deletion_data, p_other.m_state.deletion_data, new_state.bitset_size * sizeof(uint32_t));
				}

				deallocate_();

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
		template<uint32_t, class...>
		friend class dfr_map;


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
