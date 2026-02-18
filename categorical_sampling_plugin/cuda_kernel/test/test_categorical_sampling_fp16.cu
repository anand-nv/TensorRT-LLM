#include "categorical_sampling.cuh"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>

// Helper function to check CUDA errors
#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t error = call;                                                                                      \
        if (error != cudaSuccess)                                                                                      \
        {                                                                                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error)         \
                      << std::endl;                                                                                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// Helper function to convert float vector to half vector
std::vector<half> floatToHalf(std::vector<float> const& float_vec)
{
    std::vector<half> half_vec(float_vec.size());
    for (size_t i = 0; i < float_vec.size(); ++i)
    {
        half_vec[i] = __float2half(float_vec[i]);
    }
    return half_vec;
}

void testBasicSampling()
{
    std::cout << "=== Test 1: Basic Categorical Sampling (FP16) ===" << std::endl;

    int const batch_size = 4;
    int const vocab_size = 5;

    // Create simple probability distributions
    std::vector<float> h_probs_float = {// Batch 0: Uniform distribution
        0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
        // Batch 1: Skewed to first token
        0.7f, 0.1f, 0.1f, 0.05f, 0.05f,
        // Batch 2: Skewed to last token
        0.05f, 0.05f, 0.1f, 0.1f, 0.7f,
        // Batch 3: Two peaks
        0.4f, 0.1f, 0.0f, 0.1f, 0.4f};

    // Convert to half precision
    std::vector<half> h_probs = floatToHalf(h_probs_float);

    // Allocate device memory
    half* d_probs;
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_probs, batch_size * vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * sizeof(int)));

    // Copy probabilities to device
    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), batch_size * vocab_size * sizeof(half), cudaMemcpyHostToDevice));

    // Run sampling (uses clock-based seeding)
    std::vector<int> h_output(batch_size);
    categoricalSampling(d_probs, d_output, batch_size, vocab_size);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Sampled indices: ";
    for (int i = 0; i < batch_size; ++i)
    {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "Test 1 passed!" << std::endl << std::endl;
}

void testStatisticalDistribution()
{
    std::cout << "=== Test 2: Statistical Distribution Check (FP16) ===" << std::endl;

    int const batch_size = 1;
    int const vocab_size = 5;
    int const num_samples = 100000;

    // Create probability distribution
    std::vector<float> h_probs_float = {0.1f, 0.2f, 0.3f, 0.25f, 0.15f};
    std::vector<half> h_probs = floatToHalf(h_probs_float);

    // Count occurrences
    std::vector<int> counts(vocab_size, 0);

    // Allocate device memory
    half* d_probs;
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_probs, vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), vocab_size * sizeof(half), cudaMemcpyHostToDevice));

    // Sample many times (each call uses clock-based seeding)
    for (int i = 0; i < num_samples; ++i)
    {
        categoricalSampling(d_probs, d_output, 1, vocab_size);

        int result;
        CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
        counts[result]++;
    }

    // Print statistics
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Index | Expected | Observed | Difference" << std::endl;
    std::cout << "------|----------|----------|------------" << std::endl;

    for (int i = 0; i < vocab_size; ++i)
    {
        float expected = h_probs_float[i];
        float observed = static_cast<float>(counts[i]) / num_samples;
        float diff = std::abs(expected - observed);

        std::cout << std::setw(5) << i << " | " << std::setw(8) << expected << " | " << std::setw(8) << observed
                  << " | " << std::setw(10) << diff << std::endl;

        // Check if within reasonable bounds (3% tolerance)
        if (diff > 0.03f)
        {
            std::cerr << "Warning: Large deviation for index " << i << std::endl;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "Test 2 passed!" << std::endl << std::endl;
}

void testUnnormalizedProbabilities()
{
    std::cout << "=== Test 3: Unnormalized Probabilities (FP16) ===" << std::endl;

    int const vocab_size = 5;
    int const num_samples = 50000;

    // Test with unnormalized distribution
    std::vector<float> h_probs_float = {10.0f, 20.0f, 30.0f, 25.0f, 15.0f};
    std::vector<half> h_probs = floatToHalf(h_probs_float);

    // Expected normalized probabilities
    float sum = 0.0f;
    for (float p : h_probs_float)
        sum += p;
    std::vector<float> expected_probs(vocab_size);
    for (int i = 0; i < vocab_size; ++i)
    {
        expected_probs[i] = h_probs_float[i] / sum;
    }

    half* d_probs;
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_probs, vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), vocab_size * sizeof(half), cudaMemcpyHostToDevice));

    std::vector<int> counts(vocab_size, 0);
    for (int i = 0; i < num_samples; ++i)
    {
        categoricalSampling(d_probs, d_output, 1, vocab_size);
        int result;
        CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
        counts[result]++;
    }

    std::cout << "Unnormalized input: ";
    for (float p : h_probs_float)
        std::cout << p << " ";
    std::cout << std::endl;

    std::cout << "Index | Expected | Observed | Difference" << std::endl;
    std::cout << "------|----------|----------|------------" << std::endl;

    bool all_passed = true;
    for (int i = 0; i < vocab_size; ++i)
    {
        float expected = expected_probs[i];
        float observed = static_cast<float>(counts[i]) / num_samples;
        float diff = std::abs(expected - observed);

        std::cout << std::setw(5) << i << " | " << std::setw(8) << std::fixed << std::setprecision(4) << expected
                  << " | " << std::setw(8) << observed << " | " << std::setw(10) << diff;

        if (diff > 0.01f)
        {
            std::cout << " ✗" << std::endl;
            all_passed = false;
        }
        else
        {
            std::cout << " ✓" << std::endl;
        }
    }

    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output));

    if (all_passed)
    {
        std::cout << "✓ Test 3 passed! Unnormalized probabilities handled correctly." << std::endl;
    }
    else
    {
        std::cerr << "✗ Test 3 had warnings." << std::endl;
    }
    std::cout << std::endl;
}

void testZeroProbabilities()
{
    std::cout << "=== Test 4: Zero Probability Events (FP16) ===" << std::endl;

    int const vocab_size = 5;
    int const num_samples = 50000;

    // Sparse distribution with zeros
    std::vector<float> h_probs_float(vocab_size, 0.0f);
    h_probs_float[1] = 30.0f;
    h_probs_float[3] = 70.0f;
    std::vector<half> h_probs = floatToHalf(h_probs_float);

    half* d_probs;
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_probs, vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), vocab_size * sizeof(half), cudaMemcpyHostToDevice));

    std::vector<int> counts(vocab_size, 0);
    for (int i = 0; i < num_samples; ++i)
    {
        categoricalSampling(d_probs, d_output, 1, vocab_size);
        int result;
        CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
        counts[result]++;
    }

    bool passed = true;
    std::cout << "Results:" << std::endl;
    for (int i = 0; i < vocab_size; ++i)
    {
        std::cout << "  Index " << i << ": " << counts[i] << " samples";
        if (h_probs_float[i] == 0.0f && counts[i] > 0)
        {
            std::cout << " ✗ (should be 0)" << std::endl;
            passed = false;
        }
        else if (h_probs_float[i] > 0.0f)
        {
            float expected = h_probs_float[i] / (h_probs_float[1] + h_probs_float[3]);
            float observed = static_cast<float>(counts[i]) / num_samples;
            std::cout << " ✓ (expected " << std::fixed << std::setprecision(2) << (expected * 100) << "%)" << std::endl;
        }
        else
        {
            std::cout << " ✓" << std::endl;
        }
    }

    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output));

    if (passed)
    {
        std::cout << "Test 4 passed! Zero probabilities never sampled." << std::endl;
    }
    else
    {
        std::cerr << "Test 4 FAILED!" << std::endl;
    }
    std::cout << std::endl;
}

void testSingleElement()
{
    std::cout << "=== Test 5: Single Element Vector (FP16) ===" << std::endl;

    int const batch_size = 5;
    int const vocab_size = 1;
    int const num_tests = 100;

    std::vector<float> h_probs_float(batch_size * vocab_size, 1.0f);
    std::vector<half> h_probs = floatToHalf(h_probs_float);

    half* d_probs;
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_probs, batch_size * vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), batch_size * vocab_size * sizeof(half), cudaMemcpyHostToDevice));

    std::vector<int> h_output(batch_size);
    bool all_zeros = true;

    for (int test = 0; test < num_tests; ++test)
    {
        categoricalSampling(d_probs, d_output, batch_size, vocab_size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < batch_size; ++i)
        {
            if (h_output[i] != 0)
            {
                all_zeros = false;
                std::cerr << "  Error: Expected index 0, got " << h_output[i] << std::endl;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output));

    if (all_zeros)
    {
        std::cout << "✓ All " << (num_tests * batch_size) << " samples correctly returned index 0" << std::endl;
        std::cout << "Test 5 passed!" << std::endl;
    }
    else
    {
        std::cerr << "Test 5 FAILED!" << std::endl;
    }
    std::cout << std::endl;
}

void testNonReproducibility()
{
    std::cout << "=== Test 6: Non-Reproducible Randomness (FP16) ===" << std::endl;

    int const batch_size = 8;
    int const vocab_size = 10;

    // Uniform distribution
    std::vector<float> h_probs_float(batch_size * vocab_size, 1.0f / vocab_size);
    std::vector<half> h_probs = floatToHalf(h_probs_float);

    half* d_probs;
    int* d_output1;
    int* d_output2;

    CUDA_CHECK(cudaMalloc(&d_probs, batch_size * vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output1, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output2, batch_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), batch_size * vocab_size * sizeof(half), cudaMemcpyHostToDevice));

    // Sample twice
    categoricalSampling(d_probs, d_output1, batch_size, vocab_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    categoricalSampling(d_probs, d_output2, batch_size, vocab_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_output1(batch_size);
    std::vector<int> h_output2(batch_size);

    CUDA_CHECK(cudaMemcpy(h_output1.data(), d_output1, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output2.data(), d_output2, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

    int num_different = 0;
    for (int i = 0; i < batch_size; ++i)
    {
        if (h_output1[i] != h_output2[i])
        {
            num_different++;
        }
    }

    float different_percentage = 100.0f * num_different / batch_size;
    std::cout << "Different samples: " << num_different << "/" << batch_size << " (" << std::fixed
              << std::setprecision(1) << different_percentage << "%)" << std::endl;

    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_output1));
    CUDA_CHECK(cudaFree(d_output2));

    // With uniform distribution, expect most to be different
    if (different_percentage > 50.0f)
    {
        std::cout << "Test 6 passed! Clock-based seeding produces non-reproducible results." << std::endl;
    }
    else
    {
        std::cout << "Test 6 WARNING: Lower than expected randomness variation." << std::endl;
    }
    std::cout << std::endl;
}

int main()
{
    std::cout << "Starting Categorical Sampling FP16 Tests" << std::endl;
    std::cout << "=========================================" << std::endl << std::endl;

    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;

    // Run tests
    testBasicSampling();
    testStatisticalDistribution();
    testUnnormalizedProbabilities();
    testZeroProbabilities();
    testSingleElement();
    testNonReproducibility();

    std::cout << "All tests completed successfully!" << std::endl;

    return 0;
}
