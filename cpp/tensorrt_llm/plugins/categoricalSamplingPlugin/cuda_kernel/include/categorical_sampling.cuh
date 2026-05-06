#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Categorical sampling kernel that samples indices from probability distributions
// Each thread samples one index from its corresponding probability vector
// Uses clock-based seeding for non-reproducible randomness

/**
 * @brief Performs categorical sampling on GPU with automatic clock-based seeding (FP16 version)
 *
 * @param probs Input probabilities [batch_size, vocab_size] - can be unnormalized (will be normalized internally)
 * @param output Sampled indices [batch_size]
 * @param batch_size Number of probability distributions to sample from
 * @param vocab_size Size of each probability distribution
 *
 * @note This function uses clock64() for seeding, providing non-reproducible randomness.
 *       Each call will produce different results even with the same input.
 */
void categoricalSampling(half const* probs, int* output, int batch_size, int vocab_size, cudaStream_t stream);
