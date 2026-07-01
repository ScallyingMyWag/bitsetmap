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
- ~1.5 bits (not bytes) of memory overhead per element, not including VM page overcommitting
- Fast reinsert through chunked free list
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

    // Or, alternatively
    // visits all elements, use for_each, for_each_while, or erase_if
    // Can cause the compiler to heavily flatten iteration code layout and can provide substantial performance improvement
    slotMap.for_each([&element](int i)
        {
            element += i;
        });

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

In general, expect this to be slightly to moderately slower for iteration vs a sparse set, faster in iteration vs most or perhaps all other slot maps. Expect it to be faster for allocation compared to most slot maps, and expect it to be the fastest or near the fastest for insert and erase.
In terms of lookups, it is direct, like most non sparse -> dense slot maps. In terms of validation for dead or occupied slots, it is a direct lookup and comparison, compared to a non O(1) scan for something like plf::colony.

In general, this container appears to be the fastest template container in the world when:
- Data order doesn't matter
- Churn through rate (insertions and deletions) are high, and may happen in moments of critical latency
- References to non deleted elements must remain stable
- Iteration happens frequently

And as a bonus:
- When handle validation needs to happen frequently

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

In terms of class, I position it as a competitor to sparse sets, that is, a dense array + a sparse indirection array. And I position this in the same class as plf::colony.
Compared to a sparse set, this container should be faster in insert, erase, and lookups, and slower in iteration.

Envision a scenario where one entity may hold a pointer to another in which it must lookup the position, and update it's own to follow. Either of these entities may be inserted or deleted at random, in large numbers.
The entity being followed will be iterated through each frame, and at random it will either be deleted, or have it's position updated.
Later, the entities that follow will be iterated through, and attempt to lookup their parents which they follow.

plf::colony fails here due to a lack of generations, a slot might be reused upon deletion and insertion, and a pointer that a follower holds would become stale but valid, causing a corruption of state.
Aside from that, bitset_map would likely simply have better insertion, erasure, and iteration performance in this scenario.

A sparse set with generations works here, but fails in performance due to several factors. The slower insertion speed, the slower erasure speed, and the double indirection on lookup for each follower entity.
The sparse set iteration speed would be faster, but bitset_map can be comparable here.
