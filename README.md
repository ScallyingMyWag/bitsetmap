# scw::bitset_map Slot Map
A fast Slot Map

## Features
A slot map is a data structure which features:

- O(1) insert
- O(1) erase
- O(1) lookup
- Stable references to stored elements (handle/index/pointer) through insert/erase

This slot map specifically provides:

- Fast allocation through virtual memory page commits, no reallocation
- Fully contiguous virtual memory
- Relies on VirtualAlloc on windows, and mmap on Linux
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
Similar to the stb series of libraries, define the SCW_MAP_PLATFORM macro in a single translation unit (.cpp file).
This will give the container the syscalls it needs, and by including windows.h or the linux necessary headers in that translation unit. This is done to avoid polluting your code with windows.h
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

## API was Moved to github wiki

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
