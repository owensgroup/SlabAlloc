/*
 * Copyright 2018 Saman Ashkiani
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <stdint.h>
#include <iostream>
#include "slab_alloc_global.cuh"

/*
 * This class does not own any memory, and will be shallowly copied into device
 * kernel
 */
template <uint32_t LOG_NUM_MEM_BLOCKS_,
          uint32_t NUM_SUPER_BLOCKS_ALLOCATOR_,
          uint32_t MEM_UNIT_WARP_MULTIPLES_ = 1>
class SlabAllocLightContext {
 public:
  // fixed parameters for the SlabAllocLight
  static constexpr uint32_t NUM_MEM_UNITS_PER_BLOCK_ = 1024;
  static constexpr uint32_t NUM_BITMAP_PER_MEM_BLOCK_ = 32;
  static constexpr uint32_t BITMAP_SIZE_ = 32;
  static constexpr uint32_t WARP_SIZE_ = 32;
  static constexpr uint32_t MEM_UNIT_SIZE_ = MEM_UNIT_WARP_MULTIPLES_ * WARP_SIZE_;
  static constexpr uint32_t SUPER_BLOCK_BIT_OFFSET_ALLOC_ = 27;
  static constexpr uint32_t MEM_BLOCK_BIT_OFFSET_ALLOC_ = 10;
  static constexpr uint32_t MEM_UNIT_BIT_OFFSET_ALLOC_ = 5;
  static constexpr uint32_t NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ = (1 << LOG_NUM_MEM_BLOCKS_);
  static constexpr uint32_t MEM_BLOCK_SIZE_ = NUM_MEM_UNITS_PER_BLOCK_ * MEM_UNIT_SIZE_;
  static constexpr uint32_t SUPER_BLOCK_SIZE_ =
      ((BITMAP_SIZE_ + MEM_BLOCK_SIZE_) * NUM_MEM_BLOCKS_PER_SUPER_BLOCK_);
  static constexpr uint32_t MEM_BLOCK_OFFSET_ =
      (BITMAP_SIZE_ * NUM_MEM_BLOCKS_PER_SUPER_BLOCK_);
  static constexpr uint32_t num_super_blocks_ = NUM_SUPER_BLOCKS_ALLOCATOR_;

  __device__ __host__ SlabAllocLightContext()
      : d_super_blocks_(nullptr)
      , hash_coef_(0)
      , num_attempts_(0)
      , resident_index_(0)
      , super_block_index_(0)
      , allocated_index_(0) {}

  __device__ __host__ SlabAllocLightContext(const SlabAllocLightContext& rhs)
      : d_super_blocks_(rhs.d_super_blocks_)
      , hash_coef_(rhs.hash_coef_)
      , num_attempts_(0)
      , resident_index_(0)
      , super_block_index_(0)
      , allocated_index_(0) {}

  SlabAllocLightContext& operator=(const SlabAllocLightContext& rhs) {
    d_super_blocks_ = rhs.d_super_blocks_;
    hash_coef_ = rhs.hash_coef_;
    num_attempts_ = 0;
    resident_index_ = 0;
    super_block_index_ = 0;
    allocated_index_ = 0;
    return *this;
  }

  __device__ __host__ ~SlabAllocLightContext() {}

  __device__ __host__ void initParameters(uint32_t* d_super_block, uint32_t hash_coef) {
    d_super_blocks_ = d_super_block;
    hash_coef_ = hash_coef;
  }

  __device__ __host__ void copyParameters(const SlabAllocLightContext& rhs) {
    d_super_blocks_ = rhs.d_super_blocks_;
    hash_coef_ = rhs.hash_coef_;
  }

  // =========
  // some helper inline address functions:
  // =========
  __device__ __host__ __forceinline__ uint32_t
  getSuperBlockIndex(SlabAllocAddressT address) const {
    return address >> SUPER_BLOCK_BIT_OFFSET_ALLOC_;
  }

  __device__ __host__ __forceinline__ uint32_t
  getMemBlockIndex(SlabAllocAddressT address) const {
    return ((address >> MEM_BLOCK_BIT_OFFSET_ALLOC_) & 0x1FFFF);
  }

  __device__ __host__ __forceinline__ SlabAllocAddressT
  getMemBlockAddress(SlabAllocAddressT address) const {
    return (MEM_BLOCK_OFFSET_ + getMemBlockIndex(address) * MEM_BLOCK_SIZE_);
  }

  __device__ __host__ __forceinline__ uint32_t
  getMemUnitIndex(SlabAllocAddressT address) const {
    return address & 0x3FF;
  }

  __device__ __host__ __forceinline__ SlabAllocAddressT
  getMemUnitAddress(SlabAllocAddressT address) {
    return getMemUnitIndex(address) * MEM_UNIT_SIZE_;
  }

  __device__ __forceinline__ uint32_t* getPointerFromSlab(const SlabAllocAddressT& next,
                                                          const uint32_t& laneId) {
    return reinterpret_cast<uint32_t*>(d_super_blocks_) + addressDecoder(next) + laneId;
  }

  __device__ __forceinline__ uint32_t* getPointerForBitmap(
      const uint32_t super_block_index,
      const uint32_t bitmap_index) {
    return d_super_blocks_ + super_block_index * SUPER_BLOCK_SIZE_ + bitmap_index;
  }

  // called at the beginning of the kernel:
  __device__ __forceinline__ void createMemBlockIndex(uint32_t global_warp_id) {
    super_block_index_ = global_warp_id % num_super_blocks_;
    resident_index_ = (hash_coef_ * global_warp_id) >> (32 - LOG_NUM_MEM_BLOCKS_);
  }

  // called when the allocator fails to find an empty unit to allocate:
  __device__ __forceinline__ void updateMemBlockIndex(uint32_t global_warp_id) {
    num_attempts_++;
    super_block_index_++;
    super_block_index_ =
        (super_block_index_ == num_super_blocks_) ? 0 : super_block_index_;
    resident_index_ =
        (hash_coef_ * (global_warp_id + num_attempts_)) >> (32 - LOG_NUM_MEM_BLOCKS_);
    // loading the assigned memory block:
    resident_bitmap_ = *((d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_) +
                         resident_index_ * BITMAP_SIZE_ + (threadIdx.x & 0x1f));
  }

  // Objective: each warp selects its own resident warp allocator:
  __device__ __forceinline__ void initAllocator(uint32_t& tid, uint32_t& laneId) {
    // hashing the memory block to be used:
    createMemBlockIndex(tid >> 5);

    // loading the assigned memory block:
    resident_bitmap_ = *(d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
                         resident_index_ * BITMAP_SIZE_ + laneId);
    allocated_index_ = 0xFFFFFFFF;
  }

  __device__ __forceinline__ uint32_t warpAllocate(const uint32_t& laneId) {
    // tries and allocate a new memory units within the resident memory block
    // if it returns 0xFFFFFFFF, then there was not any empty memory unit
    // a new resident block should be chosen, and repeat again
    // allocated result:  5  bits: super_block_index
    //                    17 bits: memory block index
    //                    5  bits: memory unit index (hi-bits of 10bit)
    //                    5  bits: memory unit index (lo-bits of 10bit)
    int empty_lane = -1;
    uint32_t free_lane;
    uint32_t read_bitmap = resident_bitmap_;
    uint32_t allocated_result = 0xFFFFFFFF;
    // works as long as <31 bit are used in the allocated_result
    // in other words, if there are 32 super blocks and at most 64k blocks per
    // super block

    while (allocated_result == 0xFFFFFFFF) {
      empty_lane = __ffs(~resident_bitmap_) - 1;
      free_lane = __ballot_sync(0xFFFFFFFF, empty_lane >= 0);
      if (free_lane == 0) {
        // all bitmaps are full: need to be rehashed again:
        updateMemBlockIndex((threadIdx.x + blockIdx.x * blockDim.x) >> 5);
        read_bitmap = resident_bitmap_;
        continue;
      }
      uint32_t src_lane = __ffs(free_lane) - 1;
      if (src_lane == laneId) {
        read_bitmap = atomicCAS(d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
                                    resident_index_ * BITMAP_SIZE_ + laneId,
                                resident_bitmap_,
                                resident_bitmap_ | (1 << empty_lane));
        if (read_bitmap == resident_bitmap_) {
          // successful attempt:
          resident_bitmap_ |= (1 << empty_lane);
          allocated_result = (super_block_index_ << SUPER_BLOCK_BIT_OFFSET_ALLOC_) |
                             (resident_index_ << MEM_BLOCK_BIT_OFFSET_ALLOC_) |
                             (laneId << MEM_UNIT_BIT_OFFSET_ALLOC_) | empty_lane;
        } else {
          // Not successful: updating the current bitmap
          resident_bitmap_ = read_bitmap;
        }
      }
      // asking for the allocated result;
      allocated_result = __shfl_sync(0xFFFFFFFF, allocated_result, src_lane);
    }
    return allocated_result;
  }

  __device__ __forceinline__ uint32_t warpAllocateBulk(uint32_t& laneId,
                                                       const uint32_t k) {
    // tries and allocate k consecutive memory units within the resident memory
    // block if it returns 0xFFFFFFFF, then there was not any empty memory unit
    // a new resident block should be chosen, and repeat again
    // allocated result:  5  bits: super_block_index
    //                    17 bits: memory block index
    //                    5  bits: memory unit index (hi-bits of 10bit)
    //                    5  bits: memory unit index (lo-bits of 10bit)
    int empty_lane = -1;
    uint32_t free_lane;
    uint32_t read_bitmap = resident_bitmap_;
    uint32_t allocated_result = 0xFFFFFFFF;
    // works as long as <31 bit are used in the allocated_result
    // in other words, if there are 32 super blocks and at most 64k blocks per
    // super block

    while (allocated_result == 0xFFFFFFFF) {
      empty_lane =
          32 -
          (__ffs(__brev(~resident_bitmap_)));  // reversing the order of assigning lanes
                                               // compared to single allocations
      const uint32_t mask = ((1 << k) - 1) << (empty_lane - k + 1);
      // mask = %x\n", context.resident_bitmap, empty_lane, mask);
      free_lane = __ballot_sync(
          0xFFFFFFFF,
          (empty_lane >= (k - 1)) &&
              !(resident_bitmap_ &
                mask));  // update true statement to make sure everything fits
      if (free_lane == 0) {
        // all bitmaps are full: need to be rehashed again:
        updateMemBlockIndex((threadIdx.x + blockIdx.x * blockDim.x) >> 5);
        read_bitmap = resident_bitmap_;
        continue;
      }
      uint32_t src_lane = __ffs(free_lane) - 1;

      if (src_lane == laneId) {
        read_bitmap = atomicCAS(d_super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
                                    resident_index_ * BITMAP_SIZE_ + laneId,
                                resident_bitmap_,
                                resident_bitmap_ | mask);
        if (read_bitmap == resident_bitmap_) {
          // successful attempt:
          resident_bitmap_ |= mask;
          allocated_result = (super_block_index_ << SUPER_BLOCK_BIT_OFFSET_ALLOC_) |
                             (resident_index_ << MEM_BLOCK_BIT_OFFSET_ALLOC_) |
                             (laneId << MEM_UNIT_BIT_OFFSET_ALLOC_) | empty_lane;
        } else {
          // Not successful: updating the current bitmap
          resident_bitmap_ = read_bitmap;
        }
      }
      // asking for the allocated result;
      allocated_result = __shfl_sync(0xFFFFFFFF, allocated_result, src_lane);
    }
    return allocated_result;
  }

  /*
  This function, frees a recently allocated memory unit by a single thread.
  Since it is untouched, there shouldn't be any worries for the actual memory
  contents to be reset again.
*/
  __device__ __forceinline__ void freeUntouched(SlabAllocAddressT ptr) {
    atomicAnd(d_super_blocks_ + getSuperBlockIndex(ptr) * SUPER_BLOCK_SIZE_ +
                  getMemBlockIndex(ptr) * BITMAP_SIZE_ + (getMemUnitIndex(ptr) >> 5),
              ~(1 << (getMemUnitIndex(ptr) & 0x1F)));
  }

  __host__ __device__ __forceinline__ SlabAllocAddressT
  addressDecoder(SlabAllocAddressT address_ptr_index) {
    return getSuperBlockIndex(address_ptr_index) * SUPER_BLOCK_SIZE_ +
           getMemBlockAddress(address_ptr_index) +
           getMemUnitIndex(address_ptr_index) * MEM_UNIT_WARP_MULTIPLES_ * WARP_SIZE_;
  }

  __host__ __device__ __forceinline__ void print_address(
      SlabAllocAddressT address_ptr_index) {
    printf(
        "Super block Index: %d, Memory block index: %d, Memory unit index: "
        "%d\n",
        getSuperBlockIndex(address_ptr_index),
        getMemBlockIndex(address_ptr_index),
        getMemUnitIndex(address_ptr_index));
  }

  __device__ __forceinline__ void print_info() {
    printf(
        "SlabAllocLightContext: Thread: %d, d_super_blocks_: %p, hash_coef_: %u, "
        "num_attempts_: %u, "
        "resident_index_: %u, resident_bitmap_: %x, super_block_index_: %u\n",
        threadIdx.x,
        d_super_blocks_,
        hash_coef_,
        num_attempts_,
        resident_index_,
        resident_bitmap_,
        super_block_index_,
        allocated_index_);
  }

 private:
  // a pointer to each super-block
  uint32_t* d_super_blocks_;

  // hash_coef (register): used as (16 bits, 16 bits) for hashing
  uint32_t hash_coef_;  // a random 32-bit

  // resident_index: (register)
  // should indicate what memory block and super block is currently resident
  // (16 bits       + 5 bits)
  // (memory block  + super block)
  uint32_t num_attempts_;
  uint32_t resident_index_;
  uint32_t resident_bitmap_;
  uint32_t super_block_index_;
  uint32_t allocated_index_;  // to be asked via shuffle after
};

/*
 * This class owns the memory for the allocator on the device
 */
template <uint32_t LOG_NUM_MEM_BLOCKS_,
          uint32_t NUM_SUPER_BLOCKS_ALLOCATOR_,
          uint32_t MEM_UNIT_WARP_MULTIPLES_ = 1>
class SlabAllocLight {
 private:
  // a pointer to each super-block
  uint32_t* d_super_blocks_;

  // hash_coef (register): used as (16 bits, 16 bits) for hashing
  uint32_t hash_coef_;  // a random 32-bit

  // the context class is actually copied shallowly into GPU device
  SlabAllocLightContext<LOG_NUM_MEM_BLOCKS_,
                        NUM_SUPER_BLOCKS_ALLOCATOR_,
                        MEM_UNIT_WARP_MULTIPLES_>
      slab_alloc_context_;

 public:
  // =========
  // constructor:
  // =========
  SlabAllocLight() : d_super_blocks_(nullptr), hash_coef_(0) {
    // random coefficients for allocator's hash function
    std::mt19937 rng(time(0));
    hash_coef_ = rng();

    // In the light version, we put num_super_blocks super blocks within a
    // single array
    CHECK_ERROR(cudaMalloc((void**)&d_super_blocks_,
                           slab_alloc_context_.SUPER_BLOCK_SIZE_ *
                               slab_alloc_context_.num_super_blocks_ * sizeof(uint32_t)));

    for (int i = 0; i < slab_alloc_context_.num_super_blocks_; i++) {
      // setting bitmaps into zeros:
      CHECK_ERROR(cudaMemset(d_super_blocks_ + i * slab_alloc_context_.SUPER_BLOCK_SIZE_,
                             0x00,
                             slab_alloc_context_.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ *
                                 slab_alloc_context_.BITMAP_SIZE_ * sizeof(uint32_t)));
      // setting empty memory units into ones:
      CHECK_ERROR(cudaMemset(d_super_blocks_ + i * slab_alloc_context_.SUPER_BLOCK_SIZE_ +
                                 (slab_alloc_context_.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ *
                                  slab_alloc_context_.BITMAP_SIZE_),
                             0xFF,
                             slab_alloc_context_.MEM_BLOCK_SIZE_ *
                                 slab_alloc_context_.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ *
                                 sizeof(uint32_t)));
    }

    // initializing the slab context:
    slab_alloc_context_.initParameters(d_super_blocks_, hash_coef_);
  }

  // =========
  // destructor:
  // =========
  ~SlabAllocLight() { CHECK_ERROR(cudaFree(d_super_blocks_)); }

  // =========
  // Helper member functions:
  // =========
  SlabAllocLightContext<LOG_NUM_MEM_BLOCKS_,
                        NUM_SUPER_BLOCKS_ALLOCATOR_,
                        MEM_UNIT_WARP_MULTIPLES_>*
  getContextPtr() {
    return &slab_alloc_context_;
  }
};