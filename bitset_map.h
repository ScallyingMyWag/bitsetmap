#pragma once

#include <type_traits>
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <ranges>
#include <bit>
#include <new>

#include <immintrin.h>


namespace scw
{
	// CONCEPTS
	struct use_generations {};
	struct no_generations {};
	struct is_const {};
	struct not_const {};
	struct bitscan {};
	struct branched {};
	struct return_map {};
	struct no_map {};


	template<class T>
	concept const_iterator_concept = std::same_as<T, is_const> || std::same_as<T, not_const>;

	template<class T>
	concept use_generations_concept = std::same_as<T, use_generations> || std::same_as<T, no_generations>;

	template<class T>
	concept iterator_branch_concept = std::same_as<T, bitscan> || std::same_as<T, branched>;

	template<class T>
	concept return_remap_map_concept = std::same_as<T, return_map> || std::same_as<T, no_map>;


	template<class, uint32_t, use_generations_concept, const_iterator_concept, iterator_branch_concept>
	class bitset_map_iterator;

	template<class>
	class remap_map;


	namespace platform
	{
		inline size_t OS_PAGE_SIZE;


		size_t get_page_size() noexcept;
		[[nodiscard]] void* reserve(size_t) noexcept;
		[[nodiscard]] bool commit(void*, size_t) noexcept;
		[[nodiscard]] bool free(void*, size_t) noexcept;
		[[nodiscard]] bool decommit(void*, size_t) noexcept;


		[[nodiscard]] inline bool query_system_page_info() noexcept
		{
			OS_PAGE_SIZE = platform::get_page_size();

			return false;
		}


		inline void initialize_system_page_data() noexcept
		{
			[[maybe_unused]] static const bool _ = query_system_page_info();
		}
	}


	namespace implementation
	{
		template<class, uint32_t, use_generations_concept, const_iterator_concept, iterator_branch_concept>
		class bitset_map_iterator_base;
	}



	// BITSET MAP
	template<class T, uint32_t t_VM_reserve_elements, use_generations_concept t_use_generations = no_generations>
	class bitset_map
	{
	private:
		static_assert(std::is_nothrow_destructible_v<T>, "scw::bitset_map requires T to be nothrow destructible");

	private: // TYPES
		struct IndividualisticNode
		{
			union
			{
				T value;
				uint32_t free_list_index;
			};
		};


		struct GenerationalNode
		{
			uint32_t generation;

			union
			{
				T value;
				uint32_t free_list_index;
			};
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
		template<class, uint32_t, use_generations_concept, const_iterator_concept, iterator_branch_concept>
		friend class implementation::bitset_map_iterator_base;


		constexpr inline static bool c_generational = std::same_as<t_use_generations, use_generations>;


		using Node = std::conditional_t<c_generational, GenerationalNode, IndividualisticNode>;

	public:
		using handle = std::conditional_t<c_generational, GenerationalHandle, IndividualisticHandle>;
		using iterator = bitset_map_iterator<T, t_VM_reserve_elements, t_use_generations, not_const, bitscan>;
		using const_iterator = bitset_map_iterator<T, t_VM_reserve_elements, t_use_generations, is_const, bitscan>;
		using branched_iterator = bitset_map_iterator<T, t_VM_reserve_elements, t_use_generations, not_const, branched>;
		using const_branched_iterator = bitset_map_iterator<T, t_VM_reserve_elements, t_use_generations, is_const, branched>;

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
				for (uint32_t index = 0U; index < p_element_count; ++index)
				{
					emplace_back_unchecked(p_value);
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


		template <std::ranges::input_range t_range>
		explicit bitset_map(t_range&& p_range)
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


		template <class t_iterator>
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

			const uint32_t slot = get_allocation_slot_();

			return construct_in_slot_(slot, std::forward<Args>(p_args)...);
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

			const uint32_t slot = get_end_allocation_slot_();

			return construct_in_slot_(slot, std::forward<Args>(p_args)...);
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

			const uint32_t slot = get_unchecked_allocation_slot_();

			return construct_in_slot_(slot, std::forward<Args>(p_args)...);
		}


		handle push_back_unchecked(const T& p_value) noexcept(std::is_nothrow_constructible_v<T, const T&>)
		{
			return emplace_back_unchecked(p_value);
		}


		handle push_back_unchecked(T&& p_value) noexcept(std::is_nothrow_constructible_v<T, T&&>)
		{
			return emplace_back_unchecked(std::move(p_value));
		}


		void erase(uint32_t p_index) noexcept
		{
			destroy_element_(p_index);
		}


		void erase(handle p_handle) noexcept
		{
			erase(p_handle.index);
		}


		void erase(T* p_element) noexcept
		{
			erase(index_of_(p_element));
		}


		template<class t_iterator>
		t_iterator erase(const t_iterator& p_iterator) noexcept
		{
			t_iterator next_element = p_iterator;
			++next_element;

			destroy_element_(static_cast<uint32_t>(p_iterator.m_offset));

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


		void try_erase(T* p_element, uint32_t p_generation) noexcept
			requires (c_generational)
		{
			try_erase(index_of_(p_element), p_generation);
		}


		void try_erase(T* p_element) noexcept
			requires (!c_generational)
		{
			try_erase(index_of_(p_element));
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
			requires c_generational
		{
			return try_at(p_handle.index, p_handle.generation);
		}


		[[nodiscard]] T* try_at(handle p_handle) noexcept
			requires (!c_generational)
		{
			return try_at(p_handle.index);
		}


		[[nodiscard]] const T* try_at(handle p_handle) const noexcept
			requires c_generational
		{
			return try_at(p_handle.index, p_handle.generation);
		}


		[[nodiscard]] const T* try_at(handle p_handle) const noexcept
			requires (!c_generational)
		{
			return try_at(p_handle.index);
		}


		[[nodiscard]] T* try_at(T* p_element, uint32_t p_generation) noexcept
			requires (c_generational)
		{
			if (is_generation(p_element, p_generation))
			{
				return p_element;
			}

			return nullptr;
		}


		[[nodiscard]] T* try_at(T* p_element) noexcept
			requires (!c_generational)
		{
			if (is_alive(p_element))
			{
				return p_element;
			}

			return nullptr;
		}


		[[nodiscard]] const T* try_at(const T* p_element, uint32_t p_generation) const noexcept
			requires (c_generational)
		{
			if (is_generation(p_element, p_generation))
			{
				return p_element;
			}

			return nullptr;
		}


		[[nodiscard]] const T* try_at(const T* p_element) const noexcept
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


		[[nodiscard]] bool is_alive(T* p_element) const noexcept
		{
			return is_alive(index_of_(p_element));
		}


		[[nodiscard]] bool is_generation(uint32_t p_index, uint32_t p_generation) const noexcept
			requires c_generational
		{
			return m_data[p_index].generation == p_generation;
		}


		[[nodiscard]] bool is_generation(handle p_handle) const noexcept
			requires c_generational
		{
			return is_generation(p_handle.index, p_handle.generation);
		}


		[[nodiscard]] bool is_generation(T* p_element, uint32_t p_generation) const noexcept
			requires c_generational
		{
			return reinterpret_cast<Node*>(reinterpret_cast<char*>(p_element) - offsetof(Node, value))->generation == p_generation;
		}


		[[nodiscard]] uint32_t& get_generation(uint32_t p_index) noexcept
			requires c_generational
		{
			return m_data[p_index].generation;
		}


		[[nodiscard]] const uint32_t& get_generation(const uint32_t p_index) const noexcept
			requires c_generational
		{
			return m_data[p_index].generation;
		}


		[[nodiscard]] uint32_t& get_generation(handle p_handle) noexcept
			requires c_generational
		{
			return get_generation(p_handle.index);
		}


		[[nodiscard]] const uint32_t& get_generation(const handle p_handle) const noexcept
			requires c_generational
		{
			return get_generation(p_handle.index);
		}


		[[nodiscard]] uint32_t& get_generation(T* p_element) noexcept
			requires c_generational
		{
			return get_generation(index_of_(p_element));
		}


		[[nodiscard]] const uint32_t& get_generation(const T* p_element) const noexcept
			requires c_generational
		{
			return get_generation(index_of_(p_element));
		}


		template<class t_iterator>
		[[nodiscard]] uint32_t& get_generation(const t_iterator& p_iterator) noexcept
			requires c_generational
		{
			return p_iterator.m_data[p_iterator.m_offset].generation;
		}


		template<class t_iterator>
		[[nodiscard]] const uint32_t& get_generation(const t_iterator& p_iterator) const noexcept
			requires c_generational
		{
			return p_iterator.m_data[p_iterator.m_offset].generation;
		}


		[[nodiscard]] bool is_empty() const noexcept
		{
			return !m_size;
		}


		[[nodiscard]] uint32_t size() const noexcept
		{
			return m_size;
		}


		[[nodiscard]] uint32_t back_capacity() const noexcept
		{
			return static_cast<uint32_t>(m_end_data - m_data - m_high_water_mark);
		}


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


		void reserve(uint32_t p_reserve_count)
		{
			const uint32_t new_page_count = align(static_cast<size_t>(p_reserve_count) * sizeof(Node), platform::OS_PAGE_SIZE) / platform::OS_PAGE_SIZE;

			if (new_page_count > m_page_count)
			{
				grow_(new_page_count - m_page_count);
			}
		}

		// Intended to break pointer stability
		template<return_remap_map_concept t_return_map = return_map, class Allocator = std::allocator<uint32_t>>
		std::conditional_t<std::same_as<t_return_map, return_map>, remap_map<Allocator>, no_map> compress()
			requires std::is_nothrow_move_constructible_v<T>
		{
			static constexpr bool c_return_map = std::same_as<t_return_map, return_map>;
			std::conditional_t<c_return_map, remap_map<Allocator>, no_map> map;

			uint32_t last_index = 0U;

			if (m_size)
			{
				last_index = clast().m_offset;
				uint32_t elements_to_move = 0U;

				if constexpr (c_return_map)
				{
					uint32_t counted_elements = 0U;
					uint32_t free_slots = 0U;

					for (uint32_t current_index = 0U; free_slots <= m_size - counted_elements; ++current_index)
					{
						if (current_index == last_index >> 6U)
						{
							const uint64_t shift_amount = static_cast<uint64_t>(~last_index & 63U);
							const uint64_t pop_count = _mm_popcnt_u64(m_skip_data[current_index] & UINT64_MAX >> shift_amount);
							counted_elements += pop_count;
							free_slots += 64ULL - pop_count - shift_amount;
						}
						else
						{
							const uint64_t pop_count = _mm_popcnt_u64(m_skip_data[current_index]);
							counted_elements += pop_count;
							free_slots += 64ULL - pop_count;
						}
					}

					elements_to_move = m_size - counted_elements;
				}
				else
				{
					elements_to_move = (last_index + 1U) - m_size;
				}

				if (elements_to_move)
				{
					if constexpr (c_return_map)
					{
						size_t pow_2_elements_to_move = std::bit_ceil(elements_to_move);

						if (static_cast<float>(elements_to_move) / static_cast<float>(pow_2_elements_to_move) > 0.6f)
						{
							pow_2_elements_to_move *= 2ULL;
						}

						map.allocate(pow_2_elements_to_move);
					}

					for (uint32_t current_index = 0U; current_index < last_index; ++current_index)
					{
						if (!is_alive(current_index))
						{
							if constexpr (c_generational)
							{
								m_data[current_index].generation = m_data[last_index].generation;
							}

							::new(&m_data[current_index].value) T(std::move(m_data[last_index].value));

							if constexpr (!std::is_trivially_destructible_v<T>)
							{
								m_data[last_index].value.~T();
							}

							if constexpr (c_return_map)
							{
								map.insert(last_index, current_index);
							}

							while (!is_alive(--last_index) && last_index > current_index) {}
						}
					}
				}

				decommit_pages_(m_size);
			}
			else
			{
				decommit_pages_(last_index);
			}

			m_high_water_mark = m_size;
			m_free_list_index = UINT32_MAX;

			if constexpr (c_generational)
			{
				memset(m_data + m_high_water_mark, 0, static_cast<size_t>(m_end_data - m_data - m_high_water_mark) * sizeof(Node));
			}

			memset(m_skip_data, ~0, get_skip_bytes_for_page_count_(m_page_count));

			return map;
		}


		void shrink_to_fit() noexcept
		{
			uint32_t index = 0U;

			if (m_size)
			{
				index = clast().m_offset;

				m_high_water_mark = index + 1U;

				decommit_pages_(index);

				if constexpr (c_generational)
				{
					memset(m_data + index + 1U, 0, static_cast<size_t>(m_end_data - m_data - index - 1U) * sizeof(Node));
				}

				m_skip_data[m_high_water_mark >> 6U] |= UINT64_MAX << static_cast<uint64_t>(m_high_water_mark & 63U);
				const size_t bytes_to_reset = get_skip_bytes_for_page_count_(m_page_count) - static_cast<size_t>((m_high_water_mark >> 6U) + 1U) * sizeof(uint64_t);
				memset(m_skip_data + (m_high_water_mark >> 6U) + 1U, ~0, bytes_to_reset);

				uint32_t new_free_list_index = UINT32_MAX;
				for (uint32_t current_index = index - 1U; current_index != UINT32_MAX; --current_index)
				{
					if (!is_alive(current_index))
					{
						m_data[current_index].free_list_index = new_free_list_index;
						new_free_list_index = current_index;
					}
				}

				m_free_list_index = new_free_list_index;

				return;
			}

			decommit_pages_(index);

			memset(m_skip_data, ~0, get_skip_bytes_for_page_count_(m_page_count));

			m_high_water_mark = 0U;
			m_free_list_index = UINT32_MAX;
		}


		void clear() noexcept
		{
			if constexpr (!std::is_trivially_destructible_v<T>)
			{
				for (iterator it = begin(); it != end(); ++it)
				{
					(*it).~T();
				}
			}

			memset(m_skip_data, ~0, static_cast<size_t>((m_high_water_mark >> 6U) + 1U) * sizeof(uint64_t));

			m_high_water_mark = 0U;
			m_size = 0U;
			m_free_list_index = UINT32_MAX;
		}

		// ITERATORS
		[[nodiscard]] iterator begin() noexcept
		{
			iterator to_return = iterator(m_data, m_skip_data, 0ULL, *m_skip_data & UINT64_MAX << 1ULL);

			return *m_skip_data & 1ULL ? to_return : ++to_return;
		}


		[[nodiscard]] iterator end() noexcept
		{
			return iterator(m_data, nullptr, static_cast<size_t>(m_high_water_mark), 0ULL);
		}


		[[nodiscard]] iterator last() noexcept
		{
			if (!m_size)
			{
				return begin();
			}

			const uint64_t shift_amount = static_cast<uint64_t>(~m_high_water_mark & 63U);
			const uint64_t word = m_skip_data[m_high_water_mark >> 6U] & UINT64_MAX >> shift_amount >> 1ULL;
			iterator to_return = iterator(m_data, m_skip_data + (m_high_water_mark >> 6U), static_cast<size_t>(m_high_water_mark), word);

			return --to_return;
		}


		[[nodiscard]] const_iterator begin() const noexcept
		{
			const_iterator to_return = const_iterator(m_data, m_skip_data, 0ULL, *m_skip_data & UINT64_MAX << 1ULL);

			return *m_skip_data & 1ULL ? to_return : ++to_return;
		}


		[[nodiscard]] const_iterator end() const noexcept
		{
			return const_iterator(m_data, nullptr, static_cast<size_t>(m_high_water_mark), 0ULL);
		}


		[[nodiscard]] const_iterator last() const noexcept
		{
			if (!m_size)
			{
				return begin();
			}

			const uint64_t shift_amount = static_cast<uint64_t>(~m_high_water_mark & 63U);
			const uint64_t word = m_skip_data[m_high_water_mark >> 6U] & UINT64_MAX >> shift_amount >> 1ULL;
			const_iterator to_return = const_iterator(m_data, m_skip_data + (m_high_water_mark >> 6U), static_cast<size_t>(m_high_water_mark), word);

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


		[[nodiscard]] branched_iterator begin_branched() noexcept
		{
			branched_iterator to_return = branched_iterator(m_data, m_skip_data, 0ULL, *m_skip_data);

			return *m_skip_data & 1ULL ? to_return : ++to_return;
		}


		[[nodiscard]] branched_iterator end_branched() noexcept
		{
			return branched_iterator(m_data, nullptr, static_cast<size_t>(m_high_water_mark), 0ULL);
		}


		[[nodiscard]] branched_iterator last_branched() noexcept
		{
			if (!m_size)
			{
				return begin_branched();
			}

			const uint64_t shift_amount = static_cast<uint64_t>(m_high_water_mark & 63U);
			const uint64_t word = m_skip_data[m_high_water_mark >> 6U] << shift_amount;
			branched_iterator to_return = branched_iterator(m_data, m_skip_data + (m_high_water_mark >> 6U), static_cast<size_t>(m_high_water_mark), word);

			return --to_return;
		}


		[[nodiscard]] const_branched_iterator begin_branched() const noexcept
		{
			const_branched_iterator to_return = const_branched_iterator(m_data, m_skip_data, 0ULL, *m_skip_data);

			return *m_skip_data & 1ULL ? to_return : ++to_return;
		}


		[[nodiscard]] const_branched_iterator end_branched() const noexcept
		{
			return const_branched_iterator(m_data, nullptr, static_cast<size_t>(m_high_water_mark), 0ULL);
		}


		[[nodiscard]] const_branched_iterator last_branched() const noexcept
		{
			if (!m_size)
			{
				return begin_branched();
			}

			const uint64_t shift_amount = static_cast<uint64_t>(m_high_water_mark & 63U);
			const uint64_t word = m_skip_data[m_high_water_mark >> 6U] << shift_amount;
			const_branched_iterator to_return = const_branched_iterator(m_data, m_skip_data + (m_high_water_mark >> 6U), static_cast<size_t>(m_high_water_mark), word);

			return --to_return;
		}


		[[nodiscard]] const_branched_iterator cbegin_branched() const noexcept
		{
			return begin_branched();
		}


		[[nodiscard]] const_branched_iterator cend_branched() const noexcept
		{
			return end_branched();
		}


		[[nodiscard]] const_branched_iterator clast_branched() const noexcept
		{
			return last_branched();
		}

	private: // IMPLEMENTATION
		void allocate_(uint32_t p_reserve_count)
		{
			[[maybe_unused]] static const bool _ = initialize_reserve_sizes_();

			const uint32_t elements_to_reserve = std::clamp(p_reserve_count, 1U, t_VM_reserve_elements);

			size_t reserve_size = get_allocation_bytes_for_element_count_(elements_to_reserve);
			size_t skip_reserve_size = get_skip_bytes_for_page_count_(static_cast<uint32_t>(reserve_size / platform::OS_PAGE_SIZE));

			m_page_count = static_cast<uint32_t>(reserve_size / platform::OS_PAGE_SIZE);

			m_data = static_cast<Node*>(platform::reserve(sm_reserved_bytes));
			m_skip_data = static_cast<uint64_t*>(platform::reserve(sm_skip_reserved_bytes));

			if (!m_data || !m_skip_data) [[unlikely]]
			{
				allocate_fail_();
			}
			if (!platform::commit(m_data, reserve_size) ||
				!platform::commit(m_skip_data, skip_reserve_size)) [[unlikely]]
			{
				allocate_fail_();
			}

			memset(m_skip_data, ~0, skip_reserve_size);

			m_end_data = m_data + reserve_size / sizeof(Node);
		}


		void allocate_fail_()
		{
			if (m_data)
			{
				if (!platform::free(m_data, sm_reserved_bytes)) [[unlikely]]
				{
					std::abort();
				}
			}
			if (m_skip_data)
			{
				if (!platform::free(m_skip_data, sm_skip_reserved_bytes)) [[unlikely]]
				{
					std::abort();
				}
			}

			throw std::bad_alloc();
		}


		void copy_bitset_map_(const bitset_map& other)
		{
			m_high_water_mark = other.m_high_water_mark;
			m_size = other.m_size;
			m_free_list_index = other.m_free_list_index;

			allocate_(m_high_water_mark);

			memcpy(m_skip_data, other.m_skip_data, get_skip_bytes_for_page_count_(m_page_count));

			if constexpr (std::is_trivially_copyable_v<T>)
			{
				memcpy(m_data, other.m_data, static_cast<size_t>(m_high_water_mark) * sizeof(Node));
			}
			else
			{
				uint32_t current_index = 0U;

				try
				{
					for (; current_index != m_high_water_mark; ++current_index)
					{
						if constexpr (c_generational)
						{
							m_data[current_index].generation = other.m_data[current_index].generation;
						}

						if (!is_alive(current_index))
						{
							::new(&m_data[current_index].free_list_index) uint32_t(other.m_data[current_index].free_list_index);
						}
						else
						{
							::new(&m_data[current_index].value) T(other.m_data[current_index].value);
						}
					}
				}
				catch (...)
				{
					deallocate_<true>(current_index);

					throw;
				}
			}
		}


		void steal_other_(bitset_map&& p_other) noexcept
		{
			m_data = p_other.m_data;
			m_end_data = p_other.m_end_data;
			m_skip_data = p_other.m_skip_data;
			m_page_count = p_other.m_page_count;
			m_high_water_mark = p_other.m_high_water_mark;
			m_size = p_other.m_size;
			m_free_list_index = p_other.m_free_list_index;

			p_other.m_data = nullptr;
			p_other.m_end_data = nullptr;
			p_other.m_skip_data = nullptr;
			p_other.m_page_count = 0U;
			p_other.m_high_water_mark = 0U;
			p_other.m_size = 0U;
			p_other.m_free_list_index = 0U;
		}


		template<bool t_enable_last_index = false>
		void deallocate_(uint32_t p_last_index = 0U) noexcept
		{
			if constexpr (!std::is_trivially_destructible_v<T>)
			{
				if (m_data && m_skip_data)
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
			if (m_skip_data)
			{
				if (!platform::free(m_skip_data, sm_skip_reserved_bytes)) [[unlikely]]
				{
					std::abort();
				}

				m_skip_data = nullptr;
			}
		}


		void grow_(uint32_t p_pages_to_commit)
		{
			const size_t old_bytes = static_cast<size_t>(m_page_count) * platform::OS_PAGE_SIZE;
			const size_t old_skip_array_bytes = get_skip_bytes_for_page_count_(m_page_count);
			const uint32_t max_committable_pages = static_cast<uint32_t>((sm_reserved_bytes - old_bytes) / platform::OS_PAGE_SIZE);
			p_pages_to_commit = std::min(p_pages_to_commit, max_committable_pages);
			const size_t bytes_to_commit = static_cast<size_t>(p_pages_to_commit) * platform::OS_PAGE_SIZE;
			const size_t skip_bytes_to_commit = get_skip_bytes_for_page_count_(m_page_count + p_pages_to_commit) - old_skip_array_bytes;

			if (bytes_to_commit)
			{
				if (!platform::commit(reinterpret_cast<char*>(m_data) + old_bytes, bytes_to_commit)) [[unlikely]]
				{
					throw std::bad_alloc();
				}
			}
			if (skip_bytes_to_commit)
			{
				if (!platform::commit(reinterpret_cast<char*>(m_skip_data) + old_skip_array_bytes, skip_bytes_to_commit)) [[unlikely]]
				{
					throw std::bad_alloc();
				}
			}

			memset(reinterpret_cast<char*>(m_skip_data) + old_skip_array_bytes, ~0, skip_bytes_to_commit);

			m_page_count = m_page_count + p_pages_to_commit;
			m_end_data = m_data + static_cast<size_t>(m_page_count) * platform::OS_PAGE_SIZE / sizeof(Node);
		}


		[[nodiscard]] uint32_t get_allocation_slot_()
		{
			if (m_free_list_index != UINT32_MAX)
			{
				const uint32_t return_index = m_free_list_index;
				m_free_list_index = m_data[m_free_list_index].free_list_index;
				++m_size;

				toggle_bit_(return_index);

				return return_index;
			}
			else if (m_data + m_high_water_mark + 1U == m_end_data)
			{
				grow_(m_page_count);
			}

			++m_high_water_mark;
			++m_size;

			return m_high_water_mark - 1U;
		}


		[[nodiscard]] uint32_t get_end_allocation_slot_()
		{
			if (m_data + m_high_water_mark + 1U == m_end_data)
			{
				grow_(m_page_count);
			}

			++m_high_water_mark;
			++m_size;

			return m_high_water_mark - 1U;
		}


		[[nodiscard]] uint32_t get_unchecked_allocation_slot_() noexcept
		{
			++m_high_water_mark;
			++m_size;

			return m_high_water_mark - 1U;
		}


		template<class... Args>
		[[nodiscard]] handle construct_in_slot_(uint32_t p_slot, Args&&... p_args) noexcept
			requires std::is_nothrow_constructible_v<T, Args...>
		{
			::new(&m_data[p_slot].value) T(std::forward<Args>(p_args)...);

			if constexpr (c_generational)
			{
				return { static_cast<uint32_t>(p_slot), m_data[p_slot].generation };
			}
			else
			{
				return { static_cast<uint32_t>(p_slot) };
			}
		}


		template<class... Args>
		[[nodiscard]] handle construct_in_slot_(uint32_t p_slot, uint32_t p_high_water_mark, Args&&... p_args)
		{
			try
			{
				::new(&m_data[p_slot].value) T(std::forward<Args>(p_args)...);
			}
			catch (...)
			{
				if (m_high_water_mark == p_high_water_mark)
				{
					m_data[p_slot].free_list_index = m_free_list_index;

					--m_size;
					m_free_list_index = p_slot;

					toggle_bit_(static_cast<uint32_t>(p_slot));
				}
				else
				{
					m_high_water_mark = p_high_water_mark;
					--m_size;
				}

				throw;
			}

			if constexpr (c_generational)
			{
				return { static_cast<uint32_t>(p_slot), m_data[p_slot].generation };
			}
			else
			{
				return { static_cast<uint32_t>(p_slot) };
			}
		}


		void destroy_element_(uint32_t p_index) noexcept
		{
			if constexpr (c_generational)
			{
				++m_data[p_index].generation;
			}

			if constexpr (!std::is_trivially_destructible_v<T>)
			{
				m_data[p_index].value.~T();
			}

			::new(&m_data[p_index].free_list_index) uint32_t(m_free_list_index);
			m_free_list_index = p_index;

			--m_size;

			toggle_bit_(p_index);
		}


		void decommit_pages_(uint32_t p_index) noexcept
		{
			size_t bytes_used = (static_cast<size_t>(p_index) + 1ULL) * sizeof(Node);
			bytes_used = align(bytes_used, platform::OS_PAGE_SIZE);
			const uint32_t pages_used = bytes_used / platform::OS_PAGE_SIZE;
			const size_t bytes_to_decommit = static_cast<size_t>(m_page_count - pages_used) * platform::OS_PAGE_SIZE;

			const size_t previous_skip_array_bytes = get_skip_bytes_for_page_count_(m_page_count);
			const size_t skip_array_bytes = get_skip_bytes_for_page_count_(pages_used);
			const size_t skip_array_bytes_to_decommit = (previous_skip_array_bytes - skip_array_bytes);

			if (bytes_to_decommit)
			{
				if (!platform::decommit(reinterpret_cast<char*>(m_data) + bytes_used, bytes_to_decommit)) [[unlikely]]
				{
					std::abort();
				}
			}
			if (skip_array_bytes_to_decommit)
			{
				if (!platform::decommit(reinterpret_cast<char*>(m_skip_data) + skip_array_bytes, skip_array_bytes_to_decommit)) [[unlikely]]
				{
					std::abort();
				}
			}

			m_end_data = m_data + bytes_used / sizeof(Node);
			m_page_count = pages_used;
		}


		[[nodiscard]] uint32_t index_of_(T* element) noexcept
		{
			return reinterpret_cast<Node*>(reinterpret_cast<char*>(element) - offsetof(Node, value)) - m_data;
		}


		[[nodiscard]] uint32_t index_of_(const T* element) const noexcept
		{
			return reinterpret_cast<const Node*>(reinterpret_cast<const char*>(element) - offsetof(Node, value)) - m_data;
		}


		[[nodiscard]] bool get_bit_(uint32_t p_index) const noexcept
		{
			return m_skip_data[p_index >> 6U] & 1ULL << static_cast<uint64_t>(p_index & 63U);
		}


		void toggle_bit_(uint32_t p_index) noexcept
		{
			m_skip_data[p_index >> 6U] ^= 1ULL << static_cast<uint64_t>(p_index & 63U);
		}


		[[nodiscard]] constexpr static size_t get_allocation_bytes_for_element_count_(uint32_t p_count) noexcept
		{
			return align(static_cast<size_t>(p_count) * sizeof(Node), platform::OS_PAGE_SIZE);
		}


		[[nodiscard]] constexpr static size_t get_skip_bytes_for_page_count_(uint32_t p_page_count) noexcept
		{
			size_t skip_array_bytes = static_cast<size_t>(p_page_count) * platform::OS_PAGE_SIZE / sizeof(Node) + 64ULL;
			skip_array_bytes = align(skip_array_bytes, 64ULL) >> 3ULL;

			return align(skip_array_bytes, platform::OS_PAGE_SIZE);
		}


		[[nodiscard]] static bool initialize_reserve_sizes_() noexcept
		{
			platform::initialize_system_page_data();

			sm_reserved_bytes = get_allocation_bytes_for_element_count_(static_cast<size_t>(t_VM_reserve_elements));
			sm_skip_reserved_bytes = get_skip_bytes_for_page_count_(static_cast<uint32_t>(sm_reserved_bytes / platform::OS_PAGE_SIZE));

			return false;
		}


		[[nodiscard]] constexpr static size_t align(size_t p_value, size_t p_alignment) noexcept
		{
			return (p_value + p_alignment - 1ULL) & ~(p_alignment - 1ULL);
		}

	private: // MEMBERS
		inline static size_t sm_reserved_bytes;
		inline static size_t sm_skip_reserved_bytes;

		Node* m_data = nullptr;
		Node* m_end_data = nullptr;
		uint64_t* m_skip_data = nullptr;
		uint32_t m_page_count = 0U;
		uint32_t m_high_water_mark = 0U;
		uint32_t m_size = 0U;
		uint32_t m_free_list_index = UINT32_MAX;
	};



	// ITERATOR
	namespace implementation
	{
		template<class T, uint32_t t_elements, use_generations_concept t_use_generations, const_iterator_concept t_is_const, iterator_branch_concept>
		class bitset_map_iterator_base
		{
		protected:
			template<class, uint32_t, use_generations_concept>
			friend class bitset_map;


			constexpr static bool c_constant = std::same_as<t_is_const, is_const>;


			using ValueType = std::conditional_t<c_constant, const T, T>;
			using DataValueType = std::conditional_t<c_constant, const typename bitset_map<T, t_elements, t_use_generations>::Node*, typename bitset_map<T, t_elements, t_use_generations>::Node*>;
			using SkipValueType = std::conditional_t<c_constant, const uint64_t*, uint64_t*>;

		public:
			using value_type = T;
			using difference_type = std::ptrdiff_t;
			using iterator_category = std::bidirectional_iterator_tag;

		public:
			bitset_map_iterator_base() noexcept = default;

			bitset_map_iterator_base(DataValueType p_data, SkipValueType p_skip_ptr, uint64_t p_offset, uint64_t p_word) noexcept :
				m_data(p_data), m_skip_ptr(p_skip_ptr), m_offset(p_offset), m_word(p_word) {}

		public:
			[[nodiscard]] ValueType& operator*() const noexcept
			{
				return m_data[m_offset].value;
			}


			[[nodiscard]] ValueType* operator->() const noexcept
			{
				return reinterpret_cast<ValueType*>(&m_data[m_offset].value);
			}


			bool operator==(const bitset_map_iterator_base& other) const noexcept { return m_data + m_offset == other.m_data + other.m_offset; }
			bool operator!=(const bitset_map_iterator_base& other) const noexcept { return m_data + m_offset != other.m_data + other.m_offset; }
			bool operator>(const bitset_map_iterator_base& other) const noexcept { return m_data + m_offset > other.m_data + other.m_offset; }
			bool operator<(const bitset_map_iterator_base& other) const noexcept { return m_data + m_offset < other.m_data + other.m_offset; }
			bool operator>=(const bitset_map_iterator_base& other) const noexcept { return m_data + m_offset >= other.m_data + other.m_offset; }
			bool operator<=(const bitset_map_iterator_base& other) const noexcept { return m_data + m_offset <= other.m_data + other.m_offset; }

		public:
			DataValueType m_data;
			SkipValueType m_skip_ptr;
			uint64_t m_offset;
			uint64_t m_word;
		};
	}


	template<class T, uint32_t t_elements, use_generations_concept t_use_generations, const_iterator_concept t_is_const>
	class bitset_map_iterator<T, t_elements, t_use_generations, t_is_const, bitscan> : public implementation::bitset_map_iterator_base<T, t_elements, t_use_generations, t_is_const, bitscan>
	{
	private:
		using Base = implementation::bitset_map_iterator_base<T, t_elements, t_use_generations, t_is_const, bitscan>;

	public:
		using Base::Base;

	public:
		bitset_map_iterator& operator++() noexcept
		{
			while (!this->m_word)
			{
				this->m_offset += 64ULL;
				this->m_word = this->m_skip_ptr[this->m_offset >> 6ULL];
			}

			const uint64_t zero_count = _tzcnt_u64(this->m_word);
			this->m_offset = (this->m_offset & ~63ULL) + zero_count;
			this->m_word = _blsr_u64(this->m_word);

			return *this;
		}


		bitset_map_iterator operator++(int) noexcept
		{
			const bitset_map_iterator other{ *this };
			++*this;

			return other;
		}


		bitset_map_iterator& operator--() noexcept
		{
			while (!this->m_word)
			{
				this->m_offset -= 64ULL;
				this->m_word = this->m_skip_ptr[this->m_offset >> 6ULL];
			}

			const uint64_t zero_count = _lzcnt_u64(this->m_word);
			this->m_offset = (this->m_offset | 63ULL) - zero_count;
			this->m_word = _bzhi_u64(this->m_word, 63ULL - zero_count);

			return *this;
		}


		bitset_map_iterator operator--(int) noexcept
		{
			const bitset_map_iterator other{ *this };
			--*this;

			return other;
		}
	};


	template<class T, uint32_t t_elements, use_generations_concept t_use_generations, const_iterator_concept t_is_const>
	class bitset_map_iterator<T, t_elements, t_use_generations, t_is_const, branched> : public implementation::bitset_map_iterator_base<T, t_elements, t_use_generations, t_is_const, branched>
	{
	private:
		using Base = implementation::bitset_map_iterator_base<T, t_elements, t_use_generations, t_is_const, branched>;

	public:
		using Base::Base;

	public:
		bitset_map_iterator& operator++() noexcept
		{
			do
			{
				if (this->m_word)
				{
					++this->m_offset;
					this->m_word >>= 1ULL;
				}
				else
				{
					this->m_offset = (this->m_offset + 63ULL) & ~63ULL;
					this->m_word = this->m_skip_ptr[this->m_offset >> 6ULL];

					while (!this->m_word)
					{
						this->m_offset += 64ULL;
						this->m_word = this->m_skip_ptr[this->m_offset >> 6ULL];
					}
				}
			} while (!(this->m_word & 1ULL));

			return *this;
		}


		bitset_map_iterator operator++(int) noexcept
		{
			const bitset_map_iterator other{ *this };
			++*this;

			return other;
		}


		bitset_map_iterator& operator--() noexcept
		{
			do
			{
				if (this->m_word)
				{
					--this->m_offset;
					this->m_word <<= 1ULL;
				}
				else
				{
					this->m_offset = (this->m_offset & ~63ULL) - 1ULL;
					this->m_word = this->m_skip_ptr[this->m_offset >> 6ULL];

					while (!this->m_word)
					{
						this->m_offset -= 64ULL;
						this->m_word = this->m_skip_ptr[this->m_offset >> 6ULL];
					}
				}
			} while (!(this->m_word & (1ULL << 63ULL)));

			return *this;
		}


		bitset_map_iterator operator--(int) noexcept
		{
			const bitset_map_iterator other{ *this };
			--*this;

			return other;
		}
	};



	// REMAP MAP, simplified, never needs to grow or make checks
	template<class Allocator>
	class remap_map
	{
	private:
		struct RemapNode
		{
			int32_t psl = -1;
			uint32_t key = 0U;
			uint32_t value = 0U;

			[[nodiscard]] bool is_empty() const { return psl == -1; };
		};

	private:
		using NodeAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<RemapNode>;

	private:
		remap_map() noexcept = default;

		remap_map(const Allocator& p_alloc) noexcept : m_state(p_alloc) {}

	public:
		remap_map(const remap_map& p_other) : m_state(p_other.m_state)
		{
			m_state.data = m_state.allocate(m_state.size);
			memcpy(m_state.data, p_other.m_state.data, m_state.size * sizeof(RemapNode));
		}


		remap_map& operator=(const remap_map& p_other)
		{
			if (this != &p_other)
			{
				CompressedState new_state = p_other.m_state;
				new_state.data = new_state.allocate(new_state.size);

				m_state.deallocate(m_state.data, m_state.size);
				memcpy(new_state.data, p_other.m_state.data, new_state.size * sizeof(RemapNode));

				m_state = std::move(new_state);
			}

			return *this;
		}


		remap_map(remap_map&& p_other) noexcept : m_state(std::move(p_other.m_state))
		{
			p_other.m_state.data = nullptr;
			p_other.m_state.size = 0ULL;
		}


		remap_map& operator=(remap_map&& p_other) noexcept
		{
			if (this != &p_other)
			{
				m_state.deallocate(m_state.data, m_state.size);

				m_state = std::move(p_other.m_state);

				p_other.m_state.data = nullptr;
				p_other.m_state.size = 0ULL;
			}

			return *this;
		}


		~remap_map() noexcept
		{
			if (m_state.data)
			{
				m_state.deallocate(m_state.data, m_state.size);
			}
		}

	private:
		void allocate(size_t p_element_count)
		{
			m_state.size = p_element_count;

			m_state.data = m_state.allocate(m_state.size);

			for (size_t index = 0ULL; index != m_state.size; ++index)
			{
				m_state.data[index] = RemapNode();
			}
		}


		void insert(uint32_t p_key, uint32_t p_value) noexcept
		{
			size_t index = std::hash<uint32_t>()(p_key) & (m_state.size - 1ULL);
			int32_t psl = 0;

			while (true)
			{
				if (m_state.data[index].is_empty())
				{
					m_state.data[index].psl = psl;
					m_state.data[index].key = p_key;
					m_state.data[index].value = p_value;

					return;
				}
				else if (psl > m_state.data[index].psl)
				{
					std::swap(psl, m_state.data[index].psl);
					std::swap(p_key, m_state.data[index].key);
					std::swap(p_value, m_state.data[index].value);
				}

				++psl;
				++index;

				if (index == m_state.size)
				{
					index = 0ULL;
				}
			}
		}

	public:
		[[nodiscard]] uint32_t find(uint32_t p_key) const noexcept
		{
			const size_t start = std::hash<uint32_t>()(p_key) & (m_state.size - 1ULL);
			size_t index = start;
			int32_t psl = 0;

			while (p_key != m_state.data[index].key)
			{
				if (psl > m_state.data[index].psl)
				{
					return p_key;
				}

				++psl;
				++index;

				if (index == m_state.size)
				{
					index = 0ULL;
				}
				if (index == start)
				{
					return p_key;
				}
			}

			return m_state.data[index].value;
		}


		[[nodiscard]] bool is_empty() const noexcept
		{
			return !m_state.size;
		}

	private:
		template<class, uint32_t, use_generations_concept>
		friend class bitset_map;


		struct CompressedState : public NodeAllocator
		{
			RemapNode* data = nullptr;
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
