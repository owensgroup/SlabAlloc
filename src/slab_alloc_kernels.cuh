
/*
 * This kernel goes through all allocated bitmaps for a slab_hash's allocator
 * and store number of allocated slabs.
 */
template <typename AllocatorContextT>
__global__ void count_allocated_memory_units(
    uint32_t* d_count_super_block,
    AllocatorContextT allocator_context) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  int num_bitmaps = allocator_context.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
  if (tid >= num_bitmaps) {
    return;
  }

  for (int i = 0; i < allocator_context.num_super_blocks_; i++) {
    uint32_t read_bitmap = *(allocator_context.getPointerForBitmap(i, tid));
    atomicAdd(&d_count_super_block[i], __popc(read_bitmap));
  }
}