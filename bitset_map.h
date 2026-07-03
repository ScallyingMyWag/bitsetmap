#pragma once

#include <initializer_list>
#include <type_traits>
#include <algorithm>
#include <exception>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <ranges>
#include <bit>
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
	struct use_generations {};
	struct no_generations {};
	struct is_const {};
	struct not_const {};
	struct return_table {};
	struct no_table {};


	template<class T>
	concept const_iterator_concept = std::same_as<T, is_const> || std::same_as<T, not_const>;

	template<class T>
	concept use_generations_concept = std::same_as<T, use_generations> || std::same_as<T, no_generations>;

	template<class T>
	concept return_remap_table_concept = std::same_as<T, return_table> || std::same_as<T, no_table>;


	template<class, uint32_t, use_generations_concept, const_iterator_concept>
	class bitset_map_iterator;

	template<class>
	class remap_table;


	namespace platform
	{
		inline size_t os_page_size;


		size_t get_page_size() noexcept;
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



	// BITSET MAP
	template<class T, uint32_t t_vm_reserve_elements, use_generations_concept t_use_generations = no_generations>
	class bitset_map
	{
	private:
		static_assert(std::is_nothrow_destructible_v<T>, "scw::bitset_map requires T to be nothrow destructible");
		static_assert(t_vm_reserve_elements&& t_vm_reserve_elements < UINT32_MAX, "scw::bitset_map requires reserve size to be between 1 and uint32_t max - 1");

	private: // TYPES
		struct IndividualisticNode
		{
			T value;
		};

		// generation in front to improve handle validation cache hits
		struct GenerationalNode
		{
			uint32_t generation;
			T value;
		};

	public:
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
		template<class, uint32_t, use_generations_concept, const_iterator_concept>
		friend class bitset_map_iterator;


		constexpr static bool c_generational = std::same_as<t_use_generations, use_generations>;


		using node = std::conditional_t<c_generational, GenerationalNode, IndividualisticNode>;

	public:
		using handle = std::conditional_t<c_generational, GenerationalHandle, IndividualisticHandle>;
		using iterator = bitset_map_iterator<T, t_vm_reserve_elements, t_use_generations, not_const>;
		using const_iterator = bitset_map_iterator<T, t_vm_reserve_elements, t_use_generations, is_const>;

	public: // CONSTRUCTORS
		bitset_map()
		{
			allocate_(1U);
		}


		explicit bitset_map(uint32_t p_reserve_count)
		{
			allocate_(p_reserve_count);
		}


		bitset_map(uint32_t p_element_count, const T& p_value)
		{
			constexpr static bool c_nothrow_constructible = std::is_nothrow_constructible_v<T, const T&>;

			allocate_(p_element_count);

			if constexpr (c_nothrow_constructible)
			{
				m_high_water_mark = p_element_count;
				m_size = p_element_count;

				for (uint32_t index = 0U; index < p_element_count; ++index)
				{
					::new(&m_data[index].value) T(p_value);
				}
			}
			else
			{
				uint32_t index = 0U;

				try
				{
					for (; index < p_element_count; ++index)
					{
						emplace_back_unchecked(p_value);
					}
				}
				catch (...)
				{
					deallocate_<true>(index);

					throw;
				}
			}
		}


		template<std::ranges::input_range t_range>
		explicit bitset_map(t_range&& p_range)
			requires (!std::derived_from<std::remove_cvref_t<t_range>, bitset_map>)
		{
			constexpr static bool c_nothrow_constructible = std::is_nothrow_constructible_v<T, std::ranges::range_reference_t<t_range>>;

			if constexpr (std::ranges::sized_range<std::remove_reference_t<t_range>>)
			{
				allocate_(std::ranges::size(p_range));

				if constexpr (c_nothrow_constructible)
				{
					for (auto&& element : p_range)
					{
						emplace_back_unchecked(element);
					}
				}
				else
				{
					uint32_t index = 0U;

					try
					{
						for (auto&& element : p_range)
						{
							emplace_back_unchecked(element);
							++index;
						}
					}
					catch (...)
					{
						deallocate_<true>(index);

						throw;
					}
				}
			}
			else
			{
				allocate_(1U);

				if constexpr (c_nothrow_constructible)
				{
					for (auto&& element : p_range)
					{
						emplace_back(element);
					}
				}
				else
				{
					uint32_t index = 0U;

					try
					{
						for (auto&& element : p_range)
						{
							emplace_back(element);
							++index;
						}
					}
					catch (...)
					{
						deallocate_<true>(index);

						throw;
					}
				}
			}
		}


		template<std::input_iterator t_iterator>
		bitset_map(t_iterator p_first, t_iterator p_last)
		{
			constexpr static bool c_nothrow_constructible = std::is_nothrow_constructible_v<T, std::iter_reference_t<t_iterator>>;

			if constexpr (std::random_access_iterator<t_iterator>)
			{
				const uint32_t element_count = static_cast<uint32_t>(p_last - p_first);
				allocate_(element_count);

				if constexpr (c_nothrow_constructible)
				{
					for (; p_first != p_last; ++p_first)
					{
						emplace_back_unchecked(*p_first);
					}
				}
				else
				{
					uint32_t index = 0U;

					try
					{
						for (; p_first != p_last; ++p_first)
						{
							emplace_back_unchecked(*p_first);
							++index;
						}
					}
					catch (...)
					{
						deallocate_<true>(index);

						throw;
					}
				}
			}
			else
			{
				allocate_(1U);

				if constexpr (c_nothrow_constructible)
				{
					for (; p_first != p_last; ++p_first)
					{
						emplace_back(*p_first);
					}
				}
				else
				{
					uint32_t index = 0U;

					try
					{
						for (; p_first != p_last; ++p_first)
						{
							emplace_back(*p_first);
							++index;
						}
					}
					catch (...)
					{
						deallocate_<true>(index);

						throw;
					}
				}
			}
		}


		bitset_map(std::initializer_list<T> p_list) : bitset_map(p_list.begin(), p_list.end()) {}


		bitset_map(const bitset_map& p_other)
		{
			copy_bitset_map_(p_other);
		}


		bitset_map& operator=(const bitset_map& p_other)
		{
			if (this != &p_other)
			{
				bitset_map temp(p_other);
				std::swap(*this, temp);
			}

			return *this;
		}


		bitset_map(bitset_map&& p_other) noexcept
		{
			steal_other_(std::move(p_other));
		}


		bitset_map& operator=(bitset_map&& p_other) noexcept
		{
			if (this != &p_other)
			{
				deallocate_();

				steal_other_(std::move(p_other));
			}

			return *this;
		}


		~bitset_map() noexcept
		{
			deallocate_();
		}

	public: // MEMBER FUNCTIONS
		template<class... Args>
		handle emplace(Args&&... p_args)
		{
			if constexpr (!std::is_nothrow_constructible_v<T, Args...>)
			{
				const uint32_t rollback_high_water_mark = m_high_water_mark;

				const uint32_t slot = get_allocation_slot_();

				return construct_in_slot_(slot, rollback_high_water_mark, std::forward<Args>(p_args)...);
			}
			else
			{
				const uint32_t slot = get_allocation_slot_();

				return construct_in_slot_(slot, std::forward<Args>(p_args)...);
			}
		}


		handle insert(const T& p_value)
		{
			return emplace(p_value);
		}


		handle insert(T&& p_value)
		{
			return emplace(std::move(p_value));
		}


		template<class... Args>
		handle emplace_back(Args&&... p_args)
		{
			if constexpr (!std::is_nothrow_constructible_v<T, Args...>)
			{
				const uint32_t rollback_high_water_mark = m_high_water_mark;

				const uint32_t slot = get_end_allocation_slot_();

				return construct_in_slot_(slot, rollback_high_water_mark, std::forward<Args>(p_args)...);
			}
			else
			{
				const uint32_t slot = get_end_allocation_slot_();

				return construct_in_slot_(slot, std::forward<Args>(p_args)...);
			}
		}


		handle push_back(const T& p_value)
		{
			return emplace_back(p_value);
		}


		handle push_back(T&& p_value)
		{
			return emplace_back(std::move(p_value));
		}


		template<class... Args>
		handle emplace_back_unchecked(Args&&... p_args) noexcept(std::is_nothrow_constructible_v<T, Args...>)
		{
			if constexpr (!std::is_nothrow_constructible_v<T, Args...>)
			{
				const uint32_t rollback_high_water_mark = m_high_water_mark;

				const uint32_t slot = get_unchecked_allocation_slot_();

				return construct_in_slot_(slot, rollback_high_water_mark, std::forward<Args>(p_args)...);
			}
			else
			{
				const uint32_t slot = get_unchecked_allocation_slot_();

				return construct_in_slot_(slot, std::forward<Args>(p_args)...);
			}
		}


		handle push_back_unchecked(const T& p_value) noexcept(std::is_nothrow_constructible_v<T, const T&>)
		{
			return emplace_back_unchecked(p_value);
		}


		handle push_back_unchecked(T&& p_value) noexcept(std::is_nothrow_constructible_v<T, T&&>)
		{
			return emplace_back_unchecked(std::move(p_value));
		}


		void fill(uint32_t p_count, const T& p_value)
			requires (std::is_nothrow_constructible_v<T, const T&>)
		{
			fill_slots_(p_count, p_value);
		}


		void erase(uint32_t p_index) noexcept
		{
			destroy_element_(p_index);
		}


		void erase(handle p_handle) noexcept
		{
			erase(p_handle.index);
		}


		template<std::same_as<T*> t_pointer>
		void erase(t_pointer p_element) noexcept
		{
			erase(index_of_(p_element));
		}


		iterator erase(const iterator& p_iterator) noexcept
		{
			iterator next_element = p_iterator;
			++next_element;

			destroy_element_(static_cast<uint32_t>(p_iterator.m_skip_offset + p_iterator.m_offset));

			return next_element;
		}


		const_iterator erase(const const_iterator& p_iterator) noexcept
		{
			const_iterator next_element = p_iterator;
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


		template<std::same_as<T*> t_pointer>
		void try_erase(t_pointer p_element, uint32_t p_generation) noexcept
			requires (c_generational)
		{
			try_erase(index_of_(p_element), p_generation);
		}


		template<std::same_as<T*> t_pointer>
		void try_erase(t_pointer p_element) noexcept
			requires (!c_generational)
		{
			try_erase(index_of_(p_element));
		}


		void erase_chunk(uint32_t p_index, uint64_t p_mask) noexcept
			requires (!c_generational && std::is_trivially_copyable_v<T>)
		{
			destroy_chunk_(p_index, p_mask);
		}


		// HELPERS
		[[nodiscard]] T& at(uint32_t p_index) noexcept
		{
			return m_data[p_index].value;
		}


		[[nodiscard]] const T& at(uint32_t p_index) const noexcept
		{
			return m_data[p_index].value;
		}


		[[nodiscard]] T& at(handle p_handle) noexcept
		{
			return at(p_handle.index);
		}


		[[nodiscard]] const T& at(handle p_handle) const noexcept
		{
			return at(p_handle.index);
		}


		[[nodiscard]] T* try_at(uint32_t p_index, uint32_t p_generation) noexcept
			requires (c_generational)
		{
			if (is_generation(p_index, p_generation))
			{
				return &m_data[p_index].value;
			}

			return nullptr;
		}


		[[nodiscard]] T* try_at(uint32_t p_index) noexcept
			requires (!c_generational)
		{
			if (is_alive(p_index))
			{
				return &m_data[p_index].value;
			}

			return nullptr;
		}


		[[nodiscard]] const T* try_at(uint32_t p_index, uint32_t p_generation) const noexcept
			requires (c_generational)
		{
			if (is_generation(p_index, p_generation))
			{
				return &m_data[p_index].value;
			}

			return nullptr;
		}


		[[nodiscard]] const T* try_at(uint32_t p_index) const noexcept
			requires (!c_generational)
		{
			if (is_alive(p_index))
			{
				return &m_data[p_index].value;
			}

			return nullptr;
		}


		[[nodiscard]] T* try_at(handle p_handle) noexcept
			requires (c_generational)
		{
			return try_at(p_handle.index, p_handle.generation);
		}


		[[nodiscard]] T* try_at(handle p_handle) noexcept
			requires (!c_generational)
		{
			return try_at(p_handle.index);
		}


		[[nodiscard]] const T* try_at(handle p_handle) const noexcept
			requires (c_generational)
		{
			return try_at(p_handle.index, p_handle.generation);
		}


		[[nodiscard]] const T* try_at(handle p_handle) const noexcept
			requires (!c_generational)
		{
			return try_at(p_handle.index);
		}


		template<std::same_as<T*> t_pointer>
		[[nodiscard]] T* try_at(t_pointer p_element, uint32_t p_generation) noexcept
			requires (c_generational)
		{
			if (is_generation(p_element, p_generation))
			{
				return p_element;
			}

			return nullptr;
		}


		template<std::same_as<T*> t_pointer>
		[[nodiscard]] T* try_at(t_pointer p_element) noexcept
			requires (!c_generational)
		{
			if (is_alive(p_element))
			{
				return p_element;
			}

			return nullptr;
		}


		template<std::same_as<const T*> t_pointer>
		[[nodiscard]] const T* try_at(t_pointer p_element, uint32_t p_generation) const noexcept
			requires (c_generational)
		{
			if (is_generation(p_element, p_generation))
			{
				return p_element;
			}

			return nullptr;
		}


		template<std::same_as<const T*> t_pointer>
		[[nodiscard]] const T* try_at(t_pointer p_element) const noexcept
			requires (!c_generational)
		{
			if (is_alive(p_element))
			{
				return p_element;
			}

			return nullptr;
		}


		[[nodiscard]] bool is_alive(uint32_t p_index) const noexcept
		{
			return get_bit_(p_index);
		}


		[[nodiscard]] bool is_alive(handle p_handle) const noexcept
		{
			return is_alive(p_handle.index);
		}


		template<std::same_as<T*> t_pointer>
		[[nodiscard]] bool is_alive(t_pointer p_element) const noexcept
		{
			return is_alive(index_of_(p_element));
		}


		[[nodiscard]] bool is_generation(uint32_t p_index, uint32_t p_generation) const noexcept
			requires (c_generational)
		{
			return m_data[p_index].generation == p_generation;
		}


		[[nodiscard]] bool is_generation(handle p_handle) const noexcept
			requires (c_generational)
		{
			return is_generation(p_handle.index, p_handle.generation);
		}


		template<std::same_as<T*> t_pointer>
		[[nodiscard]] bool is_generation(t_pointer p_element, uint32_t p_generation) const noexcept
			requires (c_generational)
		{
			return reinterpret_cast<node*>(reinterpret_cast<char*>(p_element) - offsetof(node, value))->generation == p_generation;
		}


		[[nodiscard]] uint32_t& get_generation(uint32_t p_index) noexcept
			requires (c_generational)
		{
			return m_data[p_index].generation;
		}


		[[nodiscard]] const uint32_t& get_generation(uint32_t p_index) const noexcept
			requires (c_generational)
		{
			return m_data[p_index].generation;
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


		template<std::same_as<T*> t_pointer>
		[[nodiscard]] uint32_t& get_generation(t_pointer p_element) noexcept
			requires (c_generational)
		{
			return get_generation(index_of_(p_element));
		}


		template<std::same_as<const T*> t_pointer>
		[[nodiscard]] const uint32_t& get_generation(t_pointer p_element) const noexcept
			requires (c_generational)
		{
			return get_generation(index_of_(p_element));
		}


		template<class t_iterator>
		[[nodiscard]] uint32_t& get_generation(const t_iterator& p_iterator) noexcept
			requires (c_generational)
		{
			return p_iterator.m_data[p_iterator.m_skip_offset + p_iterator.m_offset].generation;
		}


		template<class t_iterator>
		[[nodiscard]] const uint32_t& get_generation(const t_iterator& p_iterator) const noexcept
			requires (c_generational)
		{
			return p_iterator.m_data[p_iterator.m_skip_offset + p_iterator.m_offset].generation;
		}


		[[nodiscard]] bool is_empty() const noexcept
		{
			return !m_size;
		}


		[[nodiscard]] uint32_t size() const noexcept
		{
			return m_size;
		}


		[[nodiscard]] uint32_t chunk_count() const noexcept
		{
			return m_high_water_mark >> 6U;
		}


		[[nodiscard]] T* get_chunk(uint32_t p_index) noexcept
			requires (!c_generational && std::is_trivially_copyable_v<T>)
		{
			return reinterpret_cast<T*>(m_data + p_index);
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

		// checks done in grow()
		void reserve(uint32_t p_reserve_count)
		{
			if (p_reserve_count > m_capacity)
			{
				grow_(p_reserve_count - m_capacity);
			}
		}


		// Intended to break pointer stability
		template<return_remap_table_concept t_return_table = return_table, class Allocator = std::allocator<uint32_t>>
		std::conditional_t<std::same_as<t_return_table, return_table>, remap_table<Allocator>, no_table> compress()
			requires (std::is_nothrow_move_constructible_v<T>)
		{
			constexpr static bool c_return_table = std::same_as<t_return_table, return_table>;
			std::conditional_t<c_return_table, remap_table<Allocator>, no_table> table;

			if (m_size)
			{
				const uint32_t max_index = m_size - 1U;
				const uint32_t max_word_index = m_size - 1U >> 6U;

				uint32_t elements_to_move = 0U;

				for (uint32_t current_index = 0U; current_index != max_word_index; ++current_index)
				{
					elements_to_move += static_cast<uint32_t>(64ULL - _mm_popcnt_u64(m_skip_data[current_index]));
				}

				const uint64_t shift_amount = _andn_u64(static_cast<uint64_t>(max_index), 63ULL);
				elements_to_move += static_cast<uint32_t>(64ULL - _mm_popcnt_u64(m_skip_data[max_word_index] & UINT64_MAX >> shift_amount) - shift_amount);

				if (elements_to_move)
				{
					if constexpr (c_return_table)
					{
						table.allocate_(m_high_water_mark - m_size, m_size);
					}

					node* hole_data = m_data;
					uint64_t* hole_skip_data = m_skip_data;
					uint64_t current_holes_word = ~*hole_skip_data;
					uint32_t hole_index = 0U;
					uint32_t hole_offset = 0U;

					node* element_data = m_data + (m_size & ~63U);
					uint64_t* element_skip_data = m_skip_data + (m_size >> 6U);
					uint64_t current_elements_word = *element_skip_data & UINT64_MAX << static_cast<uint64_t>(m_size & 63U);
					uint32_t element_offset = 0U;
					uint32_t element_index = m_size & ~63U;

					while (elements_to_move)
					{
						while (!current_holes_word)
						{
							hole_data += 64ULL;
							hole_index += 64U;
							++hole_skip_data;
							current_holes_word = ~*hole_skip_data;
						}

						hole_offset = static_cast<uint32_t>(_tzcnt_u64(current_holes_word));
						current_holes_word = _blsr_u64(current_holes_word);

						while (!current_elements_word)
						{
							element_data += 64ULL;
							element_index += 64U;
							++element_skip_data;
							current_elements_word = *element_skip_data;
						}

						element_offset = static_cast<uint32_t>(_tzcnt_u64(current_elements_word));
						current_elements_word = _blsr_u64(current_elements_word);

						if constexpr (c_generational)
						{
							hole_data[hole_offset].generation = element_data[element_offset].generation;
						}

						::new(&hole_data[hole_offset].value) T(std::move(element_data[element_offset].value));

						if constexpr (!std::is_trivially_destructible_v<T>)
						{
							element_data[element_offset].value.~T();
						}

						if constexpr (c_return_table)
						{
							table.insert_(element_index + element_offset, hole_index + hole_offset);
						}

						--elements_to_move;
					}
				}

				decommit_pages_(max_index);
			}
			else
			{
				decommit_pages_(0U);
			}

			m_free_list = UINT32_MAX;
			m_high_water_mark = m_size;

			if constexpr (c_generational)
			{
				memset(m_data + m_high_water_mark, 0, static_cast<size_t>(m_capacity - m_high_water_mark) * sizeof(node));
			}

			memset(m_skip_data, 0xFFFFFFFF, get_skip_bytes_for_element_count_(m_capacity));

			return table;
		}

		// sets all m_skip_data bits past high water mark
		void shrink_to_fit() noexcept
		{
			uint32_t index = 0U;

			if (m_size)
			{
				const const_iterator last_iterator = clast();
				index = last_iterator.m_skip_offset + last_iterator.m_offset;

				m_high_water_mark = index + 1U;

				decommit_pages_(index);

				if constexpr (c_generational)
				{
					memset(m_data + m_high_water_mark, 0, static_cast<size_t>(m_capacity - m_high_water_mark) * sizeof(node));
				}

				m_skip_data[m_high_water_mark >> 6U] |= UINT64_MAX << static_cast<uint64_t>(m_high_water_mark & 63U);
				const size_t bytes_to_reset = get_skip_bytes_for_element_count_(m_capacity) - (static_cast<size_t>(m_high_water_mark >> 6U) + 1ULL) * sizeof(uint64_t);
				memset(m_skip_data + (m_high_water_mark >> 6U) + 1ULL, 0xFFFFFFFF, bytes_to_reset);

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

			memset(m_skip_data, 0xFFFFFFFF, get_skip_bytes_for_element_count_(m_capacity));

			m_free_list = UINT32_MAX;
			m_high_water_mark = 0U;
		}


		void clear() noexcept
		{
			if constexpr (c_generational || !std::is_trivially_destructible_v<T>)
			{
				node* data = m_data;
				node* const end_data = data + (m_high_water_mark & ~63U);
				uint64_t* word_pointer = m_skip_data;
				uint64_t word = *word_pointer;
				uint64_t offset = 0ULL;
				const uint64_t end_offset = static_cast<uint64_t>(m_high_water_mark & 63U);

				while (data != end_data) [[likely]]
				{
#ifdef __GNUC__
					if (word)
					{
						do
						{
							offset = _tzcnt_u64(word);
							word = _blsr_u64(word);

							if constexpr (c_generational)
							{
								++data[offset].generation;
							}

							if constexpr (!std::is_trivially_destructible_v<T>)
							{
								data[offset].value.~T();
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

						if constexpr (c_generational)
						{
							++data[offset].generation;
						}

						if constexpr (!std::is_trivially_destructible_v<T>)
						{
							data[offset].value.~T();
						}
					}
#endif
					do
					{
						data += 64ULL;
						++word_pointer;
						word = *word_pointer;
					} while (!word);
				}

				offset = _tzcnt_u64(word);

				while (offset != end_offset)
				{
					if constexpr (c_generational)
					{
						++data[offset].generation;
					}

					if constexpr (!std::is_trivially_destructible_v<T>)
					{
						data[offset].value.~T();
					}

					word = _blsr_u64(word);
					offset = _tzcnt_u64(word);
				}
			}

			memset(m_skip_data, 0xFFFFFFFF, static_cast<size_t>((m_high_water_mark >> 6U) + 1U) * sizeof(uint64_t));

			m_free_list = UINT32_MAX;
			m_high_water_mark = 0U;
			m_size = 0U;
		}

		// ITERATORS
		// if there is no element at index 0, find the first element
		[[nodiscard]] iterator begin() noexcept
		{
			iterator to_return(m_data, m_skip_data, *m_skip_data & UINT64_MAX << 1ULL, 0U, 0U);

			return *m_skip_data & 1ULL ? to_return : ++to_return;
		}


		[[nodiscard]] iterator end() noexcept
		{
			return iterator(m_data + (m_high_water_mark & ~63U), m_skip_data, 0ULL, m_high_water_mark & ~63U, m_high_water_mark & 63U);
		}

		// to be used in reverse, double shift instead of add to avoid UB
		[[nodiscard]] iterator last() noexcept
		{
			if (!m_size)
			{
				return begin();
			}

			const uint64_t shift_amount = _andn_u64(static_cast<uint64_t>(m_high_water_mark), 63ULL);
			const uint64_t word = m_skip_data[m_high_water_mark >> 6U] & UINT64_MAX >> shift_amount >> 1ULL;
			iterator to_return(m_data + (m_high_water_mark & ~63U), m_skip_data, word, m_high_water_mark & ~63U, m_high_water_mark & 63U);

			return --to_return;
		}


		[[nodiscard]] const_iterator begin() const noexcept
		{
			const_iterator to_return(m_data, m_skip_data, *m_skip_data & UINT64_MAX << 1ULL, 0U, 0U);

			return *m_skip_data & 1ULL ? to_return : ++to_return;
		}


		[[nodiscard]] const_iterator end() const noexcept
		{
			return const_iterator(m_data + (m_high_water_mark & ~63U), m_skip_data, 0ULL, m_high_water_mark & ~63U, m_high_water_mark & 63U);
		}


		[[nodiscard]] const_iterator last() const noexcept
		{
			if (!m_size)
			{
				return begin();
			}

			const uint64_t shift_amount = _andn_u64(static_cast<uint64_t>(m_high_water_mark), 63ULL);
			const uint64_t word = m_skip_data[m_high_water_mark >> 6U] & UINT64_MAX >> shift_amount >> 1ULL;
			const_iterator to_return(m_data + (m_high_water_mark & ~63U), m_skip_data, word, m_high_water_mark & ~63U, m_high_water_mark & 63U);

			return --to_return;
		}


		[[nodiscard]] const_iterator cbegin() const noexcept
		{
			return begin();
		}


		[[nodiscard]] const_iterator cend() const noexcept
		{
			return end();
		}


		[[nodiscard]] const_iterator clast() const noexcept
		{
			return last();
		}


		[[nodiscard]] iterator get_iterator_from_chunk_index(uint32_t p_index) noexcept
		{
			iterator to_return(m_data, m_skip_data, m_skip_data[p_index] & UINT64_MAX << 1ULL, p_index << 6U, 0U);

			return m_skip_data[p_index] & 1ULL ? to_return : ++to_return;
		}


		[[nodiscard]] const_iterator get_iterator_from_chunk_index(uint32_t p_index) const noexcept
		{
			const_iterator to_return(m_data, m_skip_data, m_skip_data[p_index] & UINT64_MAX << 1ULL, p_index << 6U, 0U);

			return m_skip_data[p_index] & 1ULL ? to_return : ++to_return;
		}


		template<class t_func>
		void for_each_while(t_func p_func) noexcept
		{
			node* data = m_data;
			node* const end_data = data + (m_high_water_mark & ~63U);
			uint64_t* word_pointer = m_skip_data;
			uint64_t word = *word_pointer;
			uint64_t offset = 0ULL;
			const uint64_t end_offset = static_cast<uint64_t>(m_high_water_mark & 63U);

			while (data != end_data) [[likely]]
			{
#ifdef __GNUC__ // test + jump can be up to ~23% faster than jumping on zero flag for whatever reason. MSVC is not smart enough to jump on zero flag. GCC uses the zf set by bslr which is slower than test
				if (word)
				{
					do
					{
						offset = _tzcnt_u64(word);
						word = _blsr_u64(word);

						if (!p_func(data[offset].value))
						{
							return;
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

					if (!p_func(data[offset].value))
					{
						return;
					}
				}
#endif
				do
				{
					data += 64ULL;
					++word_pointer;
					word = *word_pointer;
				} while (!word);
			}

			offset = _tzcnt_u64(word);

			while (offset != end_offset)
			{
				if (!p_func(data[offset].value))
				{
					return;
				}

				word = _blsr_u64(word);
				offset = _tzcnt_u64(word);
			}
		}


		template<class t_func>
		void for_each(t_func p_func) noexcept
		{
			for_each_while([&p_func](T& p_element) -> bool
				{
					p_func(p_element);

					return true;
				});
		}


		template<class t_func>
		void erase_if(t_func p_func) noexcept
		{
			node* data = m_data;
			node* const end_data = data + (m_high_water_mark & ~63U);
			uint64_t* word_pointer = m_skip_data;
			uint64_t word = *word_pointer;
			uint64_t offset = 0ULL;
			const uint64_t end_offset = static_cast<uint64_t>(m_high_water_mark & 63U);

			while (data != end_data) [[likely]]
			{
#ifdef __GNUC__
				if (word)
				{
					do
					{
						offset = _tzcnt_u64(word);
						word = _blsr_u64(word);

						if (p_func(data[offset].value))
						{
							erase(static_cast<uint32_t>(data - m_data + offset));
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

					if (p_func(data[offset].value))
					{
						erase(static_cast<uint32_t>(data - m_data + offset));
					}
				}
#endif
				do
				{
					data += 64ULL;
					++word_pointer;
					word = *word_pointer;
				} while (!word);
			}

			offset = _tzcnt_u64(word);

			while (offset != end_offset)
			{
				if (p_func(data[offset].value))
				{
					erase(static_cast<uint32_t>(data - m_data + offset));
				}

				word = _blsr_u64(word);
				offset = _tzcnt_u64(word);
			}
		}

	private: // IMPLEMENTATION
		// VM reservation split between three memory blocks here
		SCW_NO_INLINE void allocate_(uint32_t p_reserve_count)
		{
			constexpr static size_t c_aligned_data_bytes = align_(get_bytes_for_element_count_(t_vm_reserve_elements), alignof(uint64_t));
			constexpr static size_t c_aligned_skip_array_bytes = align_(get_skip_bytes_for_element_count_(t_vm_reserve_elements), alignof(uint64_t*));
			constexpr static size_t c_aligned_free_table_bytes = get_free_table_bytes_for_element_count_(t_vm_reserve_elements);

			[[maybe_unused]] static const bool _ = initialize_reserve_sizes_(c_aligned_data_bytes, c_aligned_skip_array_bytes, c_aligned_free_table_bytes);

			const uint32_t elements_to_reserve = std::clamp(p_reserve_count, 1U, t_vm_reserve_elements);

			const size_t reserve_size = get_bytes_for_element_count_(elements_to_reserve);
			const size_t skip_reserve_size = get_skip_bytes_for_element_count_(elements_to_reserve);
			const size_t free_table_reserve_size = get_free_table_bytes_for_element_count_(elements_to_reserve);

			m_data = static_cast<node*>(platform::reserve(sm_reserved_bytes));

			if (!m_data) [[unlikely]]
			{
				allocate_fail_();
			}

			m_skip_data = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(m_data) + c_aligned_data_bytes);
			m_free_table = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(m_data) + c_aligned_data_bytes + c_aligned_skip_array_bytes);

			m_capacity = elements_to_reserve;

			if (!platform::commit(m_data, reserve_size) ||
				!platform::commit(m_skip_data, skip_reserve_size) ||
				!platform::commit(m_free_table, free_table_reserve_size)) [[unlikely]]
			{
				allocate_fail_();
			}

			memset(m_skip_data, 0xFFFFFFFF, skip_reserve_size);
		}


		void allocate_fail_()
		{
			if (m_data)
			{
				if (!platform::free(m_data, sm_reserved_bytes)) [[unlikely]]
				{
					std::abort();
				}

				m_data = nullptr;
			}

			throw std::bad_alloc();
		}


		void copy_bitset_map_(const bitset_map& p_other)
		{
			m_free_list = p_other.m_free_list;
			m_high_water_mark = p_other.m_high_water_mark;
			m_size = p_other.m_size;

			allocate_(m_high_water_mark);

			memcpy(m_skip_data, p_other.m_skip_data, get_skip_bytes_for_element_count_(m_high_water_mark));
			memcpy(m_free_table, p_other.m_free_table, get_free_table_bytes_for_element_count_(m_high_water_mark));

			if constexpr (std::is_trivially_copyable_v<T>)
			{
				memcpy(m_data, p_other.m_data, static_cast<size_t>(m_high_water_mark) * sizeof(node));
			}
			else
			{
				uint32_t index = 0U;

				try
				{
					for (; index != m_high_water_mark; ++index)
					{
						if constexpr (c_generational)
						{
							m_data[index].generation = p_other.m_data[index].generation;
						}

						if (is_alive(index))
						{
							::new(&m_data[index].value) T(p_other.m_data[index].value);
						}
					}
				}
				catch (...)
				{
					deallocate_<true>(index);

					throw;
				}
			}
		}


		void steal_other_(bitset_map&& p_other) noexcept
		{
			m_data = p_other.m_data;
			m_skip_data = p_other.m_skip_data;
			m_free_table = p_other.m_free_table;
			m_free_list = p_other.m_free_list;
			m_capacity = p_other.m_capacity;
			m_high_water_mark = p_other.m_high_water_mark;
			m_size = p_other.m_size;

			p_other.m_data = nullptr;
			p_other.m_skip_data = nullptr;
			p_other.m_free_table = nullptr;
			p_other.m_free_list = UINT32_MAX;
			p_other.m_capacity = 0U;
			p_other.m_high_water_mark = 0U;
			p_other.m_size = 0U;
		}

		// this is okay, m_data is checked
		template<bool t_enable_last_index = false>
		void deallocate_(uint32_t p_last_index = 0U) noexcept
		{
			if constexpr (!std::is_trivially_destructible_v<T>)
			{
				if (m_data)
				{
					for (T& element : *this)
					{
						if constexpr (t_enable_last_index)
						{
							if (index_of_(&element) >= p_last_index)
							{
								break;
							}
						}

						element.~T();
					}
				}
			}

			if (m_data)
			{
				if (!platform::free(m_data, sm_reserved_bytes)) [[unlikely]]
				{
					std::abort();
				}

				m_data = nullptr;
			}
		}

		// grow intended to silently fail when full
		SCW_NO_INLINE void grow_(uint32_t p_elements_to_commit)
		{
			const size_t old_bytes = get_bytes_for_element_count_(m_capacity);
			const size_t old_skip_bytes = get_skip_bytes_for_element_count_(m_capacity);
			const size_t old_free_table_bytes = get_free_table_bytes_for_element_count_(m_capacity);

			p_elements_to_commit = std::min(p_elements_to_commit, t_vm_reserve_elements - m_capacity);

			const size_t bytes_to_commit = get_bytes_for_element_count_(p_elements_to_commit);
			const size_t skip_bytes_to_commit = get_skip_bytes_for_element_count_(m_capacity + p_elements_to_commit) - old_skip_bytes;
			const size_t free_table_bytes_to_commit = get_free_table_bytes_for_element_count_(m_capacity + p_elements_to_commit) - old_free_table_bytes;

			if (bytes_to_commit)
			{
				if (!platform::commit(reinterpret_cast<char*>(m_data) + old_bytes, bytes_to_commit)) [[unlikely]]
				{
					throw std::bad_alloc();
				}
			}

			if (skip_bytes_to_commit)
			{
				if (!platform::commit(reinterpret_cast<char*>(m_skip_data) + old_skip_bytes, skip_bytes_to_commit)) [[unlikely]]
				{
					throw std::bad_alloc();
				}
			}

			if (free_table_bytes_to_commit)
			{
				if (!platform::commit(reinterpret_cast<char*>(m_free_table) + old_free_table_bytes, free_table_bytes_to_commit)) [[unlikely]]
				{
					throw std::bad_alloc();
				}
			}

			memset(reinterpret_cast<char*>(m_skip_data) + old_skip_bytes, 0xFFFFFFFF, skip_bytes_to_commit);

			m_capacity = m_capacity + p_elements_to_commit;
		}

		// here, dependency of the not on word doesn't seem to matter, nor does the word load. Compiler seems to cache word here when insert is called in a loop
		// chunked free list here increases latency, but reduces memory accesses drastically and removes read before write dependency, making effective latency less compared to intrusive free list
		SCW_FORCE_INLINE [[nodiscard]] uint32_t get_allocation_slot_()
		{
			if (m_free_list != UINT32_MAX)
			{
				uint64_t word = m_skip_data[m_free_list];

				const uint64_t offset = _tzcnt_u64(~word);
				word |= 1ULL << offset;
				const uint32_t slot = static_cast<uint32_t>((static_cast<uint64_t>(m_free_list) << 6ULL) + offset);

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


		SCW_FORCE_INLINE void fill_slots_(uint32_t p_count, const T& p_value)
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
					construct_in_slot_(static_cast<uint32_t>(scaled_index + offset), p_value);
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

			if (m_high_water_mark + p_count >= m_capacity)
			{
				grow_(m_high_water_mark + p_count - m_capacity);
			}

			for (uint32_t count = 0U; count < p_count; ++count)
			{
				construct_in_slot_(m_high_water_mark, p_value);
				++m_high_water_mark;
			}
		}


		template<class... Args>
		SCW_FORCE_INLINE handle construct_in_slot_(uint32_t p_slot, Args&&... p_args) noexcept
			requires (std::is_nothrow_constructible_v<T, Args...>)
		{
			::new(&m_data[p_slot].value) T(std::forward<Args>(p_args)...);

			if constexpr (c_generational)
			{
				return { p_slot, m_data[p_slot].generation };
			}
			else
			{
				return { p_slot };
			}
		}

		// rollback can be calculated from p_slot here, free table is not overwritten so that's fine
		template<class... Args>
		SCW_FORCE_INLINE [[nodiscard]] handle construct_in_slot_(uint32_t p_slot, uint32_t p_high_water_mark, Args&&... p_args)
		{
			try
			{
				::new(&m_data[p_slot].value) T(std::forward<Args>(p_args)...);
			}
			catch (...)
			{
				if (m_high_water_mark == p_high_water_mark)
				{
					m_free_list = p_slot >> 6U;
					--m_size;
					m_skip_data[m_free_list] = _andn_u64(1ULL << (p_slot & 63U), m_skip_data[m_free_list]);
				}
				else
				{
					--m_high_water_mark;
					--m_size;
				}

				throw;
			}

			if constexpr (c_generational)
			{
				return { p_slot, m_data[p_slot].generation };
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
				++m_data[p_index].generation;
			}

			if constexpr (!std::is_trivially_destructible_v<T>)
			{
				m_data[p_index].value.~T();
			}

			if (word == UINT64_MAX) [[unlikely]]
			{
				::new(m_free_table + chunk_index) uint32_t(m_free_list);
				m_free_list = chunk_index;
			}
		}


		SCW_FORCE_INLINE void destroy_chunk_(uint32_t p_index, uint64_t p_mask) noexcept
		{
			const uint64_t word = m_skip_data[p_index];

			m_size -= _mm_popcnt_u64(p_mask);

			m_skip_data[p_index] = _andn_u64(p_mask, word);

			if (word == UINT64_MAX) [[unlikely]]
			{
				::new(m_free_table + p_index) uint32_t(m_free_list);
				m_free_list = p_index;
			}
		}

		// decommits physical memory, making sure not to decommit page when one memory block bleeds into the page of another
		SCW_NO_INLINE void decommit_pages_(uint32_t p_index) noexcept
		{
			const size_t bytes_occupied = align_(static_cast<size_t>(p_index + 1U) * sizeof(node), platform::os_page_size);
			const size_t bytes_comitted = align_(m_capacity * sizeof(node), platform::os_page_size);
			size_t bytes_to_decommit = bytes_comitted - bytes_occupied;

			if (reinterpret_cast<char*>(m_data) + bytes_comitted > reinterpret_cast<char*>(m_skip_data))
			{
				bytes_to_decommit -= std::min(bytes_to_decommit, platform::os_page_size);
			}

			const size_t skip_page_offset = reinterpret_cast<uintptr_t>(m_skip_data) - _andn_u64(platform::os_page_size - 1ULL, reinterpret_cast<uintptr_t>(m_skip_data));
			const size_t skip_bytes_occupied = align_(skip_page_offset + get_skip_bytes_for_element_count_(static_cast<size_t>(p_index + 1U)), platform::os_page_size);
			const size_t skip_bytes_comitted = align_(skip_page_offset + get_skip_bytes_for_element_count_(m_capacity), platform::os_page_size);
			size_t skip_bytes_to_decommit = skip_bytes_comitted - skip_bytes_occupied;

			if (reinterpret_cast<char*>(m_skip_data) - skip_page_offset + skip_bytes_comitted > reinterpret_cast<char*>(m_free_table))
			{
				skip_bytes_to_decommit -= std::min(skip_bytes_to_decommit, platform::os_page_size);
			}

			const size_t free_table_page_offset = reinterpret_cast<uintptr_t>(m_free_table) - _andn_u64(platform::os_page_size - 1ULL, reinterpret_cast<uintptr_t>(m_free_table));
			const size_t free_table_bytes_occupied = align_(free_table_page_offset + get_free_table_bytes_for_element_count_(static_cast<size_t>(p_index + 1U)), platform::os_page_size);
			const size_t free_table_bytes_comitted = align_(free_table_page_offset + get_free_table_bytes_for_element_count_(m_capacity), platform::os_page_size);
			const size_t free_table_bytes_to_decommit = free_table_bytes_comitted - free_table_bytes_occupied;

			if (bytes_to_decommit)
			{
				if (!platform::decommit(reinterpret_cast<char*>(m_data) + bytes_occupied, bytes_to_decommit)) [[unlikely]]
				{
					std::abort();
				}
			}

			if (skip_bytes_to_decommit)
			{
				if (!platform::decommit(reinterpret_cast<char*>(m_skip_data) - skip_page_offset + skip_bytes_occupied, skip_bytes_to_decommit)) [[unlikely]]
				{
					std::abort();
				}
			}

			if (free_table_bytes_to_decommit)
			{
				if (!platform::decommit(reinterpret_cast<char*>(m_free_table) - free_table_page_offset + free_table_bytes_occupied, free_table_bytes_to_decommit)) [[unlikely]]
				{
					std::abort();
				}
			}

			m_capacity = std::min(t_vm_reserve_elements, static_cast<uint32_t>(bytes_occupied / sizeof(node)));
		}

		// helpers
		[[nodiscard]] uint32_t index_of_(T* p_element) noexcept
		{
			return reinterpret_cast<node*>(reinterpret_cast<char*>(p_element) - offsetof(node, value)) - m_data;
		}


		[[nodiscard]] uint32_t index_of_(const T* p_element) const noexcept
		{
			return reinterpret_cast<const node*>(reinterpret_cast<const char*>(p_element) - offsetof(node, value)) - m_data;
		}


		[[nodiscard]] bool get_bit_(uint32_t p_index) const noexcept
		{
			return m_skip_data[p_index >> 6U] & 1ULL << static_cast<uint64_t>(p_index & 63U);
		}

		// calculates bytes for the three memory blocks, adding sentinel to the end for high water mark to reside when full
		[[nodiscard]] constexpr static size_t get_bytes_for_element_count_(uint32_t p_count) noexcept
		{
			return static_cast<size_t>(p_count) * sizeof(node);
		}


		[[nodiscard]] constexpr static size_t get_skip_bytes_for_element_count_(uint32_t p_count) noexcept
		{
			return align_(static_cast<size_t>(p_count) + 1ULL, 64ULL) >> 3ULL;
		}


		[[nodiscard]] constexpr static size_t get_free_table_bytes_for_element_count_(uint32_t p_count) noexcept
		{
			return (align_(static_cast<size_t>(p_count) + 1ULL, 64ULL) >> 6ULL) * sizeof(uint32_t);
		}

		// gets page size and shift amount once in the program's run time
		[[nodiscard]] static bool initialize_reserve_sizes_(size_t p_aligned_data_bytes, size_t p_skip_array_bytes, size_t p_free_table_bytes)
		{
			platform::initialize_system_page_data();

			sm_reserved_bytes = align_(p_aligned_data_bytes + p_skip_array_bytes + p_free_table_bytes, platform::os_page_size);

			return false;
		}

		// aligns to next boundary
		[[nodiscard]] constexpr static size_t align_(size_t p_value, size_t p_alignment) noexcept
		{
			return (p_value + p_alignment - 1ULL) & ~(p_alignment - 1ULL);
		}

	private: // MEMBERS
		inline static size_t sm_reserved_bytes;

		node* m_data = nullptr;
		uint64_t* m_skip_data = nullptr;
		uint32_t* m_free_table = nullptr;
		uint32_t m_free_list = UINT32_MAX;
		uint32_t m_capacity = 0U;
		uint32_t m_high_water_mark = 0U;
		uint32_t m_size = 0U;
	};



	// ITERATOR
	template<class T, uint32_t t_elements, use_generations_concept t_use_generations, const_iterator_concept t_is_const>
	class bitset_map_iterator
	{
	private:
		template<class, uint32_t, use_generations_concept>
		friend class bitset_map;


		constexpr static bool c_constant = std::same_as<t_is_const, is_const>;


		using return_value = std::conditional_t<c_constant, const T, T>;
		using data_value_type = std::conditional_t<c_constant, const typename bitset_map<T, t_elements, t_use_generations>::node*, typename bitset_map<T, t_elements, t_use_generations>::node*>;
		using skip_value_type = std::conditional_t<c_constant, const uint64_t*, uint64_t*>;

	public:
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using iterator_category = std::bidirectional_iterator_tag;

	public:
		bitset_map_iterator() noexcept = default;

		bitset_map_iterator(data_value_type p_data, skip_value_type p_skip_pointer_base, uint64_t p_word, uint32_t p_skip_offset, uint32_t p_offset) noexcept :
			m_data(p_data), m_skip_pointer_base(p_skip_pointer_base), m_word(p_word), m_skip_offset(p_skip_offset), m_offset(p_offset) {}

	public:
		// this is extremely fast, highly predictable branch. ttd is essentially just base + tzcnt(word), while blsr is computed alongside payload. 4 cycles ttd in most cases
		bitset_map_iterator& operator++() noexcept
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


		bitset_map_iterator operator++(int) noexcept
		{
			const bitset_map_iterator other{ *this };
			++*this;

			return other;
		}

		// 5 cycle ttd here
		bitset_map_iterator& operator--() noexcept
		{
			while (!m_word) [[unlikely]]
			{
				m_data -= 64ULL;
				m_skip_offset -= 64U;
				m_word = m_skip_pointer_base[m_skip_offset >> 6U];
			}

			m_offset = 63U - static_cast<uint32_t>(_lzcnt_u64(m_word));
			m_word = _bzhi_u64(m_word, m_offset);

			return *this;
		}


		bitset_map_iterator operator--(int) noexcept
		{
			const bitset_map_iterator other{ *this };
			--*this;

			return other;
		}


		[[nodiscard]] return_value& operator*() const noexcept
		{
			return m_data[m_offset].value;
		}


		[[nodiscard]] return_value* operator->() const noexcept
		{
			return reinterpret_cast<return_value*>(&m_data[m_offset].value);
		}


		bool operator==(const bitset_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset == p_other.m_skip_offset + p_other.m_offset; }
		bool operator!=(const bitset_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset != p_other.m_skip_offset + p_other.m_offset; }
		bool operator>(const bitset_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset > p_other.m_skip_offset + p_other.m_offset; }
		bool operator<(const bitset_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset < p_other.m_skip_offset + p_other.m_offset; }
		bool operator>=(const bitset_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset >= p_other.m_skip_offset + p_other.m_offset; }
		bool operator<=(const bitset_map_iterator& p_other) const noexcept { return m_skip_offset + m_offset <= p_other.m_skip_offset + p_other.m_offset; }

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

		remap_table(const Allocator& p_alloc) noexcept : m_state(p_alloc) {}

	public:
		remap_table(const remap_table& p_other) : m_state(p_other.m_state)
		{
			m_state.data = m_state.allocate(m_state.size);
			m_state.offset_data = m_state.data - (p_other.m_state.data - p_other.m_state.offset_data);
			memcpy(m_state.data, p_other.m_state.data, m_state.size * sizeof(uint32_t));
		}


		remap_table& operator=(const remap_table& p_other)
		{
			if (this != &p_other)
			{
				CompressedState new_state = p_other.m_state;
				new_state.data = new_state.allocate(new_state.size);
				new_state.offset_data = new_state.data - (p_other.m_state.data - p_other.m_state.offset_data);

				m_state.deallocate(m_state.data, m_state.size);
				memcpy(new_state.data, p_other.m_state.data, new_state.size * sizeof(uint32_t));

				m_state = std::move(new_state);
			}

			return *this;
		}


		remap_table(remap_table&& p_other) noexcept : m_state(std::move(p_other.m_state))
		{
			p_other.m_state.data = nullptr;
			p_other.m_state.offset_data = nullptr;
			p_other.m_state.size = 0ULL;
		}


		remap_table& operator=(remap_table&& p_other) noexcept
		{
			if (this != &p_other)
			{
				m_state.deallocate(m_state.data, m_state.size);

				m_state = std::move(p_other.m_state);

				p_other.m_state.data = nullptr;
				p_other.m_state.offset_data = nullptr;
				p_other.m_state.size = 0ULL;
			}

			return *this;
		}


		~remap_table() noexcept
		{
			if (m_state.data)
			{
				m_state.deallocate(m_state.data, m_state.size);
			}
		}

	public:
		[[nodiscard]] uint32_t find(uint32_t p_old_handle) const noexcept
		{
			return m_state.offset_data[p_old_handle];
		}


		[[nodiscard]] bool is_empty() const noexcept
		{
			return !m_state.size;
		}

	private:
		void allocate_(size_t p_element_count, size_t p_offset)
		{
			m_state.size = p_element_count;

			m_state.data = m_state.allocate(m_state.size);
			m_state.offset_data = m_state.data - p_offset;
		}


		void insert_(uint32_t p_old_handle, uint32_t p_new_handle) noexcept
		{
			m_state.offset_data[p_old_handle] = p_new_handle;
		}

	private:
		template<class, uint32_t, use_generations_concept>
		friend class bitset_map;


		struct CompressedState : public Allocator
		{
			uint32_t* data = nullptr;
			uint32_t* offset_data = nullptr;
			size_t size = 0ULL;
		};

		CompressedState m_state;
	};
}


#ifdef SCW_MAP_PLATFORM
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/mman.h>
#endif


namespace scw
{
	namespace platform
	{
#ifdef _WIN32
		size_t get_page_size() noexcept
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
		size_t get_page_size() noexcept
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
			return !madvise(p_address, p_size, MADV_POPULATE_WRITE);
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
