#include "categorical_sampling.cuh"
#include <stdio.h>

// Categorical sampling kernel that initializes random states internally using clock
// Each thread handles one batch element
// Accepts unnormalized probabilities and normalizes them on-the-fly
// Uses clock() for non-reproducible random seeding
// FP16 version for memory efficiency
__global__ void categoricalSamplingKernel(
    half const* probs, int* output, int batch_size, int vocab_size,
    unsigned long long host_seed)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size*blockDim.x-1)
    {
        return;
    }

    // Use host-supplied entropy so each enqueue() call gets a different seed.
    // Per-thread offset uses a large prime multiplier so threads don't correlate.
    curandState localState;
    unsigned long long seed = host_seed + (unsigned long long)batch_idx * 6364136223846793005ULL;
    curand_init(seed, 0, 0, &localState);

    // Generate uniform random number [0, 1)
    float random_val = curand_uniform(&localState);

    // Compute sum of probabilities for normalization (convert to float for accuracy)
    // Also check for negative values which indicate logits were passed instead of probabilities
    half const* prob_row = probs + batch_idx * vocab_size;
    float sum = 0.0f;
    bool has_negative = false;
    float min_val = 1e6f;

    for (int i = 0; i < vocab_size; ++i)
    {
        float val = __half2float(prob_row[i]);
        if (val < 0.0f)
        {
            has_negative = true;
        }
        if (val < min_val)
        {
            min_val = val;
        }
        sum += val;
    }

    // Error out if negative probabilities detected
    if (has_negative)
    {
        //printf("ERROR in categoricalSampling: Negative probability detected! min_value=%f, sum=%f\n", min_val, sum);
        //printf("This kernel expects PROBABILITIES (positive values), not LOGITS.\n");
        //printf("Please apply softmax to your logits before passing to this plugin.\n");
        output[batch_idx] = -1; // Return error indicator
        return;
    }

    // Check for invalid sum
    if (sum < 0.0f)
    {
        printf("ERROR in categoricalSampling: Invalid probability sum=%f for batch_idx=%d\n", sum, batch_idx);
        output[batch_idx] = -1;
        return;
    }

    // Perform inverse transform sampling with normalization
    // Find the first index where normalized cumulative sum >= random_val
    float cumsum = 0.0f;
    int sampled_idx = vocab_size - 1; // Default to last index

    for (int i = 0; i < vocab_size; ++i)
    {
        cumsum += __half2float(prob_row[i]) / sum;
        if (random_val <= cumsum)
        {
            sampled_idx = i;
            break;
        }
    }

    output[batch_idx] = sampled_idx;
}

// Host function implementation
void categoricalSampling(half const* probs, int* output, int batch_size, int vocab_size,
    cudaStream_t stream, unsigned long long host_seed)
{
    int const threads_per_block = vocab_size;
    int const num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    categoricalSamplingKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        probs, output, batch_size, vocab_size, host_seed);
}
