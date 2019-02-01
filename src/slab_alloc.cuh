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

template <uint32_t LOG_NUM_MEM_BLOCKS_,
          uint32_t NUM_SUPER_BLOCKS_ALLOCATOR_,
          uint32_t MEM_UNIT_WARP_MULTIPLES_ = 1>
class SlabAllocLight {
 public:
  // fixed parameters for the SlabAllocLight
  constexpr uint32_t NUM_MEM_UNITS_PER_BLOCK_ = 1024;
  constexpr uint32_t NUM_BITMAP_PER_MEM_BLOCK_ = 32;
  constexpr uint32_t BITMAP_SIZE_ = 32;
  constexpr uint32_t WARP_SIZE_ = 32;
  constexpr uint32_t NUM_MEM_UNIT_SIZE_ = MEM_UNIT_WARP_MULTIPLES_ * WARP_SIZE_;
  constexpr uint32_t SUPER_BLOCK_BIT_OFFSET_ALLOC_ = 27;
  constexpr uint32_t MEM_BLOCK_BIT_OFFSET_ALLOC_ = 10;
  constexpr uint32_t MEM_UNIT_BIT_OFFSET_ALLOC_ = 5;
  constexpr uint32_t NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ =
      (1 << LOG_NUM_MEM_BLOCKS_);
  constexpr uint32_t MEM_BLOCK_SIZE_ = NUM_MEM_UNITS_PER_BLOCK_ * MEM_UNIT_SIZE;
  constexpr uint32_t SUPER_BLOCK_SIZE_ =
      ((BITMAP_SIZE_ + MEM_BLOCK_SIZE_) * NUM_MEM_BLOCKS_PER_SUPER_BLOCK_);
  constexpr uint32_t MEM_BLOCK_OFFSET_ =
      (BITMAP_SIZE_ * NUM_MEM_BLOCKS_PER_SUPER_BLOCK_);

  using SlabAllocAddressT = uint32_t;
  using SlabAllocGlobalOffsetT = uint32_t;

  // a pointer to each super-block
  uint32_t* d_super_blocks;

  // hash_coef (register): used as (16 bits, 16 bits) for hashing
  uint32_t hash_coef;  // a random 32-bit
  uint32_t num_super_blocks;

  // resident_index: (register)
  // should indicate what memory block and super block is currently resident
  // (16 bits 			+ 5 bits)
  // (memory block 	+ super block)
  uint32_t num_attempts;
  uint32_t resident_index;
  uint32_t resident_bitmap;
  uint32_t super_block_index;
  uint32_t allocated_index;  // to be asked via shuffle after

  __device__ __host__ __forceinline__ uint32_t
  get_super_block_index(SlabAllocAddressT address) const {
    return address >> SUPER_BLOCK_BIT_OFFSET_ALLOC_;
  }

  __device__ __host__ __forceinline__ uint32_t
  get_mem_block_index(SlabAllocAddressT address) const {
    return ((address >> MEM_BLOCK_BIT_OFFSET_ALLOC_) & 0x1FFFF);
  }

  __device__ __host__ __forceinline__ SlabAllocGlobalOffsetT
      get_mem_block_address const(SlabAllocAddressT address) {
    return (MEM_BLOCK_OFFSET_ + get_mem_block_index(address) * MEM_BLOCK_SIZE_);
  }

  __device__ __host__ __forceinline__ uint32_t
  get_mem_unit_index(SlabAllocAddressT address) const {
    return address & 0x3FF;
  }

  __device__ __host__ __forceinline__ SlabAllocGlobalOffsetT
  get_mem_unit_address(SlabAllocAddressT address) {
    return get_mem_unit_index(address) * MEM_UNIT_SIZE_;
  }
};