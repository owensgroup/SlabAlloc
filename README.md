# SlabAlloc

This library is a warp-centric dynamic memory allocator for the GPU, initially designed to be used in [SlabHash](https://github.com/owensgroup/SlabHash). It is suitable for scenarios where there is a need for dynamic memory allocations of fixed-size (fixed number of bytes). It also strictly assumes that all threads within a warp are active and can run its operations concurrently.

## Publication:
This library is based on the original slab hash paper, initially proposed in the following IPDPS'18 paper:
* [Saman Ashkiani, Martin Farach-Colton, John Owens, *A Dynamic Hash Table for the GPU*, 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS)](https://ieeexplore.ieee.org/abstract/document/8425196)

This library is a rafactored and slightly redesigned version of the original code, so that it can be extended and be used in other research projects as well. It is still under continuous development. If you find any problem with the code, or suggestions for potential additions to the library, we will appreciate it if you can raise issues on github. We will address them as soon as possible.

### Current limitations and future developments:
It currently includes the light-version, where total amount allocated memory is upper-bounded to fit in a single contiguous memory array (<4GB). The more general approach is discussed in the paper, but has not yet been added to this repo.
