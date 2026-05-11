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

https://docs.google.com/spreadsheets/d/1TGnwxBs8PnPyuRr0EmKAldLRpga1lj_0ZiZVR6fzJsQ/edit?usp=sharing

In general, expect this to be slower for iteration vs a sparse set, faster in iteration vs most or perhaps all other slot maps. Expect it to be faster for allocation compared to most slot maps, and expect it to be the fastest or near the fastest for insert and erase.
In terms of lookups, it is direct, like most non sparse -> dense slot maps. In terms of validation for dead or occupied slots, it is a direct lookup and comparison, compared to a non O(1) scan for something like plf::colony.

## API was Moved to github wiki

## Beginner API
| API | Suggestion |
| --- | --- |
| Constructor | Use any |
| Insertion | Use: ```emplace()``` or ```insert()``` |
| Erase | Use: ```erase(iterator)``` or ```try_erase(handle)``` |
| Access | Use: ```try_at(handle)``` or ```*iterator``` |
| Helpers | Use: ```is_alive(handle)```, ```size()```, ```is_empty()```, ```reserve(elements)``` |
| AVOID | ```at()```, ```density()```, ```compress()```, ```shrink_to_fit()```, ```clear()``` |

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

Bitscan iteration reads words, uses _tzcnt_u64 to find a set bit corresponding to an element, and it skips empty 8 byte words.
