cuda_add_executable(test_slab_alloc test_slab_alloc.cu 
		OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})
