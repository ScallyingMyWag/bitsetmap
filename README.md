# scw::bitset_map Slot Map
A fast Slot Map

## Attention
Currently relies on VirtualAlloc and hardware intrinsics. Needs to be ported to linux

## Features
A slot map is a data structure which features:

- O(1) insert
- O(1) erase
- O(1) lookup
- Stable references to stored elements (handle/index/pointer) through insert/erase

This slot map specifically provides:

- Fast allocation through virtual memory page commits, no reallocation
- Fully contiguous virtual memory
- Growable, up to reserved size
- Fast reinsert through intrusive free list
- Fast erase
- Direct, array indexed lookups
- True pointer stability
- Fast iteration through live elements, encoded in bitset
- Optional versioning/generations
- Optional, manually invoked compaction and handle remapping
- single header implementation
- C++20

## Usage
Similar to the stb series of libraries, define the the SCW_MAP_PLATFORM macro in a single translation unit (.cpp file).
This will give the container the syscalls it needs by including windows.h in that translation unit.
```cpp
// bitset_map_platform.cpp
#define SCW_MAP_PLATFORM
#include "bitset_map.h"


// main.cpp
#include "bitset_map.h"

constexpr static size_t MAX_ELEMENT_COUNT = 1'000'000;

// Type returned by emplace
using handle = scw::bitset_map<int, MAX_ELEMENT_COUNT>::handle;


int main()
{
    // Max elements to be stored is a template parameter
    // Reserves virtual address space for MAX_ELEMENT_COUNT elements, does not take up physical memory
    scw::bitset_map<int, MAX_ELEMENT_COUNT> slotMap;

    // Insert
    handle h1 = slotMap.emplace(12);
    handle h2 = slotMap.emplace(13);
    handle h3 = slotMap.emplace(14);

    // Lookup
    int element = slotMap.at(h1);

    // Iterate
    for (int i : slotMap)
    {
        element += i;
    }

    // Erase
    slotMap.erase(h1);

    for (auto it = slotMap.begin(); it != slotMap.end();)
    {
        if (rand() & 1)
        {
            it = slotMap.erase(it); // Returns next iterator
        }
        else
        {
            ++it;
        }
    }

    // Lookup
    if (slotMap.is_alive(h2))
    {
        element = slotMap.at(h2);
    }

    // Or, alternatively
    if (auto ptr = slotMap.try_at(h2); ptr)
    {
        element = *ptr;
    }

    // Compaction
    auto map = slotMap.compress();

    // Remap, find() returns passed in key on failure to find
    if (!map.is_empty())
    {
        h1.index = map.find(h1.index);
        h2.index = map.find(h2.index);
        h3.index = map.find(h3.index);
    }

    return 0;
}
```
## Benchmarks
The following benchmarks are not conclusive, consider them as if they only prove the container is worth benchmarking.

The benchmarks are compared to plf::colony. I love plf::colony, this is not meant to attack that container. Payload is a 32 byte struct. Operations done are reserve then insert, called allocate. Dense iterate, at full capacity. Randomly erase half of the elements. Sparse iterate, at 50% elements deleted. Reinsert erased elements. Finally, dense iteration again after reinsertion. Time is the average of 100 runs.

System is: Ryzen 7 5800x, 32 GB 3200 MHZ RAM

| Operation | Payload Count | plf::colony | scw::bitset_map | Relative to scw::bitset_map |
| --- | --- | --- | --- | --- |
| Allocate | 10k | 270 μs | 133 μs | 2.03x |
| Allocate | 100k | 2039 μs | 948 μs | 2.15x |
| Allocate | 1 Million | 19912 μs | 7878 μs | 2.52x |
| Dense Iteration | 10k | 51 μs | 13 μs | 3.92x |
| Dense Iteration | 100k | 378 μs | 106 μs | 3.57x |
| Dense Iteration | 1 Million | 3827 μs | 1041 μs | 3.68x |
| Erase | 10k | 120 μs | 52 μs | 2.31x |
| Erase | 100k | 957 μs | 420 μs | 2.28x |
| Erase | 1 Million | 9633 μs | 4132 μs | 2.33x |
| Sparse Iteration | 10k | 23 μs | 7 μs | 3.29x |
| Sparse Iteration | 100k | 187 μs | 58 μs | 3.22x |
| Sparse Iteration | 1 Million | 1896 μs | 573 μs | 3.31x |
| Reinsertion | 10k | 121 μs | 35 μs | 3.46x |
| Reinsertion | 100k | 968 μs | 285 μs | 3.4x |
| Reinsertion | 1 Million | 9688 μs | 2933 μs | 3.30x |
| Iterate after reinsert | 10k | 51 μs | 13 μs | 3.92x |
| Iterate after reinsert | 100k | 376 μs | 106 μs | 3.55x |
| Iterate after reinsert | 1 Million | 3832 μs | 1051 μs | 3.65x |

## Caution! None of the api is safe against fuzzed random integer inputs.
Contract is: Keys came from the container. Under that contract, ```try_X()``` is safe.

## Beginner API
| API | Suggestion |
| --- | --- |
| Constructor | Use any |
| Insertion | Use: ```emplace()``` or ```insert()``` |
| Erase | Use: ```erase(iterator)``` or ```try_erase(handle)``` |
| Access | Use: ```try_at(handle)``` or ```*iterator``` |
| Helpers | Use: ```is_alive(handle)```, ```size()```, ```is_empty()```, ```reserve(elements)``` |
| AVOID | ```at()```, ```density()```, ```compress()```, ```shrink_to_fit()```, ```clear()```, Branched iterators |

Ignore the rest of the api.

## API
| Template Parameters | Description |
| --- | --- |
| ```class T``` | Type to store |
| ```uint32_t t_VM_reserve_elements``` | Max elements which will ever be stored |
| ```use_generations_concept t_use_generations``` | Concept enables generations. Default is no_generations. Use: ```no_generations``` or ```use_generations``` |

| handle | Description |
| --- | --- |
| ```index``` | Offset into storage |
| ```generation``` | Generation of slot upon insertion, exists if using ```use_generations``` |

| Constructors | Description |
| --- | --- |
| ```bitset_map()``` | Allocates one page |
| ```bitset_map(uint32_t p_element_count, const T& p_value)``` | Fills with ```p_element_count``` count of ```p_value``` |
| ```bitset_map(t_range&& p_range)``` | Constructs using range |
| ```bitset_map(t_iterator p_first, t_iterator p_last)``` | Constructs using iterators |
| ```bitset_map(const bitset_map& p_other)``` | Deep copy, maintaining layout |
| ```bitset_map(bitset_map&& p_other)``` | Steals and resets other's members |
| ```~bitset_map()``` | Deallocates, nothrow |

- All insert invalidate iterators. Pointers and handles remain valid.
- Time complexity: O(1) except on page commit

| Insert | Description |
| --- | --- |
| ```handle emplace(Args&&... p_args)``` | Inserts element, forwards args, biases free list, returns handle |
| ```handle insert(const T& p_value)``` | Inserts by copy construction |
| ```handle insert(T&& p_value)``` | Inserts by move construction |
| ```handle emplace_back(Args&&... p_args)``` | Inserts element, forwards args, ignores free list and pushes to the end, returns handle |
| ```handle push_back(const T& p_value)``` | Back inserts by copy construction |
| ```handle push_back(T&& p_value)``` | Back inserts by move construction |
| ```handle emplace_back_unchecked(Args&&... p_args)``` | Inserts element, forwards args, ignores free list and pushes to the end, assuming avaliable space, otherwise UB, returns handle |
| ```handle push_back_unchecked(const T& p_value)``` | Back inserts by copy construction |
| ```handle push_back_unchecked(T&& p_value)``` | Back inserts by move construction |

- All erase invalidates iterators, pointers and handles remain valid.
- All erase is not bounds checked, assumes key is from container.
- All erase make handles stale in generational mode.
- ```is_generation()``` acts as liveness and freshness check for ```try_erase()```. For any key passed out by the container, ```is_generation() == true``` implies liveness.
- No iterator ```try_erase()```. Iterator passed implied to be pointing to live element.
- Time complexity: O(1)

| Erase | Description |
| --- | --- |
| ```void erase(uint32_t p_index)``` | Erases element by index |
| ```void erase(handle p_handle)``` | Erases element by handle |
| ```void erase(T* p_element)``` | Erases element by pointer to element |
| ```t_iterator erase(const t_iterator& p_iterator)``` | Erases element by iterator, invalidating that iterator, and returning a valid iterator to the next element |
| ```void try_erase(uint32_t p_index, uint32_t p_generation)``` | Erases element by index, if valid |
| ```void try_erase(uint32_t p_index)``` | Erases element by index, if valid |
| ```void try_erase(handle p_handle)``` | Erases element by handle, if valid |
| ```void try_erase(T* p_element, uint32_t p_generation)``` | Erases element by pointer to element, if valid |
| ```try_erase(T* p_element)``` | Erases element by pointer to element, if valid |

- at is direct pointer + offset access.
- All at is not bounds checked, assumes key is from container.
- No ```at(ptr)```. Use ```*ptr```.
- No ```at(iterator)```. Use ```*iterator```.
- Time complexity: O(1)

| Access | Description |
| --- | --- |
| ```[[nodiscard]] T& at(uint32_t p_index)``` | Access by index |
| ```[[nodiscard]] T& at(handle p_handle)``` | Access by handle |
| ```[[nodiscard]] T* try_at(uint32_t p_index, uint32_t p_generation)``` | Access by index, returns ```nullptr``` if invalid |
| ```[[nodiscard]] T* try_at(uint32_t p_index)``` | Access by index, returns ```nullptr``` if invalid |
| ```[[nodiscard]] T* try_at(handle p_handle)``` | Access by handle, returns ```nullptr``` if invalid |
| ```[[nodiscard]] T* try_at(T* p_element, uint32_t p_generation)``` | Access by pointer to element, returns ```nullptr``` if invalid |
| ```[[nodiscard]] T* try_at(T* p_element)``` | Access by pointer to element, returns ```nullptr``` if invalid |

- is alive is direct pointer + offset bit lookup.
- is alive used in non generational ```try_X()``` functions.
- All is alive is not bounds checked, assumes key is from container.
- No ```is_alive(iterator)```. ```iterator != end()``` implies liveness.
- Time complexity: O(1)

| Is Alive | Description |
| --- | --- |
| ```[[nodiscard]] bool is_alive(uint32_t p_index)``` | Check liveness by index |
| ```[[nodiscard]] bool is_alive(handle p_handle)``` | Check liveness by handle |
| ```[[nodiscard]] bool is_alive(T* p_element)``` | Check liveness by pointer to element |

- is generation is direct pointer + offset generation lookup and comparison.
- is generation used in generational mode ```try_X()``` functions.
- All is generation is not bounds checked, assumes key is from container.
- No ```is_generation(iterator)```. ```iterator != end()``` implies valid generation.
- Time complexity: O(1)

| Is Generation | Description |
| --- | --- |
| ```[[nodiscard]] bool is_generation(uint32_t p_index, uint32_t p_generation)``` | Check generation by index |
| ```[[nodiscard]] bool is_generation(handle p_handle)``` | Check generation by handle |
| ```[[nodiscard]] bool is_generation(T* p_element, uint32_t p_generation)``` | Check generation by pointer to element |

- Can return modifiable generation. Use as you will.
- get generation is direct pointer + offset generation lookup.
- All is generation is not bounds checked, assumes key is from container.
- Time complexity: O(1)

| Get Generation | Description |
| --- | --- |
| ```[[nodiscard]] uint32_t& get_generation(uint32_t p_index)``` | Get generation by index |
| ```[[nodiscard]] uint32_t& get_generation(handle p_handle)``` | Get generation by handle |
| ```[[nodiscard]] uint32_t& get_generation(T* p_element)``` | Get generation by pointer to element |
| ```[[nodiscard]] uint32_t& get_generation(const iterator& p_iterator)``` | Get generation by iterator |

| Helpers | Description |
| --- | --- |
| ```[[nodiscard]] bool is_empty()``` | ```true``` if empty. O(1) |
| ```[[nodiscard]] uint32_t size()``` | Get count of elements. O(1) |
| ```[[nodiscard]] uint32_t back_capacity()``` | Get count of slots you could back insert into without growth. O(1) |
| ```[[nodiscard]] float density()``` | Get percentage of filled slots. UB on fresh, never inserted into container. O(1) |
| ```[[nodiscard]] float try_density()``` | Get percentage of filled slots. Divide by zero checked. O(1) |
| ```void reserve(uint32_t p_reserve_count)``` | Commit virtual memory for ```p_reserve_count``` total elements. Not O(1) |
| ```template<return_remap_map_concept t_return_map, class Allocator> remap_map compress()``` | Compacts elements to the front of the container. Breaks pointer stability. Releases pages of memory past last compressed element. ```is_alive()``` and ```is_generation()``` can no longer be trusted for stale handles/heys, they are only valid for valid handles/keys, so call only if there is no stale or invalid handles/keys. ```return_remap_map_concept``` controls returning remap map. Default is ```remap_map```. Use: ```remap_map``` or ```no_map```. ```Allocator``` controls allocator for ```scw::remap_map```. Defailt is ```std::allocator<uint32_t>```. Not O(1) |
| ```void shrink_to_fit()``` | Releases pages of memory past last element. Maintains pointer stability. Like ```compress()```, ```is_alive()``` and ```is_generation()``` can no longer be trusted for stale handles/keys, they are only valid for valid handles/keys, so call only if there is no stale or invalid handles/keys. Not O(1) |
| ```void clear()``` | Resets container to empty state, without releasing committed memory. Leaves slot generation counts unchanged. Not O(1) |

- Iteration functions may advance to first live element on call.

| Iterators | Description |
| --- | --- |
| ```[[nodiscard]] iterator begin()``` | Returns iterator to first element using bitscan iteration. Not O(1) |
| ```[[nodiscard]] iterator end()``` | Returns iterator one past the conatiner end. ```operator++``` and ```operator--``` are invalid. O(1) |
| ```[[nodiscard]] iterator last()``` | Returns iterator one past the last element using bitscan iteration. O(1) |
| ```[[nodiscard]] const_iterator cbegin()``` | Returns const iterator to first element using bitscan iteration. Not O(1) |
| ```[[nodiscard]] const_iterator cend()``` | Returns const iterator one past the conatiner end. ```operator++``` and ```operator--``` are invalid. O(1) |
| ```[[nodiscard]] const_iterator clast()``` | Returns const iterator one past the last element using bitscan iteration. O(1) |
| ```[[nodiscard]] branched_iterator begin_branched()``` | Returns iterator to first element using branched iteration. Not O(1) |
| ```[[nodiscard]] branched_iterator end_branched()``` | Returns iterator one past the conatiner end. ```operator++``` and ```operator--``` are invalid. O(1) |
| ```[[nodiscard]] branched_iterator last_branched()``` | Returns iterator one past the last element using branched iteration. O(1) |
| ```[[nodiscard]] const_branched_iterator cbegin_branched()``` | Returns const iterator to first element using branched iteration. Not O(1) |
| ```[[nodiscard]] const_branched_iterator cend_branched()``` | Returns const iterator one past the conatiner end. ```operator++``` and ```operator--``` are invalid. O(1) |
| ```[[nodiscard]] const_branched_iterator clast_branched()``` | Returns const iterator one past the last element using branched iteration. O(1) |

- remap map is not user constructable.

| Remap Map | Description |
| --- | --- |
| ```[[nodiscard]] uint32_t find(uint32_t p_key)``` | Finds new index of ```p_key``` after ```compress()```. Failure to find key returns ```p_key``` to avoid branch in user code. Technically O(1), but may scan |
| ```[[nodiscard]] bool is_empty()``` | ```true``` if empty. O(1) |

## Rationale
Alright, after writing all that out I think it's worth writing more to justify why this container exists.

In terms of class, I position it as a competitor to sparse sets, that is, dense + sparse indirection arrays. And I position this in the same class as plf::colony.
Compared to a sparse set, this container should be faster in insert, erase, and lookups, and slower in iteration.

I will go over the exact work done in the fastest hot paths now:

Insert checks a free list member, which acts as an intrusive LIFO stack. It may pop off the stack, this requires a member write and memory read, no write. It then indexes into a bitset array and toggles a bit.
Otherwise, it checks for growth and appends to the end of the container, incrementing two members.

Back insert does not check for growth on the fastest path, it simply increments 2 members and constructs the element. No need to toggle bit on back append.

Erase increments the generation of the slot in generational mode, destroys the element, writes the free list head into the slot, and sets the free list head as the slot index. It decrements a member and toggles a bit.

Access is a direct base + index lookup.

is_alive() is a direct bit lookup.

is_generation() is a direct generation lookup, and comparison. Generation resides next to the payload, hopefully pulling it into the cache line on lookup.

Because the container does not shrink, is_generation() is sufficient as a liveness check for any handle returned by the container. No need to check is_alive().

Iteration is either branched or bitscan. Branched increments an offset and uses that to index the bitset. That is faster than bitscan for 100% dense data. Bitscan iteration reads words, uses _tzcnt_u64 to find a set bit corresponding to an element, and it skips empty 8 byte words.
