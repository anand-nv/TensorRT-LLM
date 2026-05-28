#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Categorical sampling kernel that samples indices from probability distributions.
// Each thread samples one index from its corresponding probability vector.

/**
 * @brief Performs categorical sampling on GPU (FP16 version)
 *
 * @param probs Input probabilities [batch_size, vocab_size] - can be unnormalized (will be normalized internally)
 * @param output Sampled indices [batch_size]
 * @param batch_size Number of probability distributions to sample from
 * @param vocab_size Size of each probability distribution
 *
 * @note This function uses host-provided entropy, providing non-reproducible randomness.
 *       Each call should produce different samples even with the same input.
 */
void categoricalSampling(half const* probs, int* output, int batch_size, int vocab_size,
    cudaStream_t stream, unsigned long long host_seed);
