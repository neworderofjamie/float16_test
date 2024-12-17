// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

// Standard C includes
#include <cassert>
#include <cmath>

// Platform includes
#ifdef _WIN32
#include <intrin.h>
#endif

// CUDA includes
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


//------------------------------------------------------------------------
// Macros
//------------------------------------------------------------------------
#define CHECK_CUDA_ERRORS(call) {                                                                   \
    cudaError_t error = call;                                                                       \
    if (error != cudaSuccess) {                                                                     \
            std::ostringstream errorMessageStream;                                                  \
            errorMessageStream << "cuda error:" __FILE__ << ": " << __LINE__ << " ";                \
            errorMessageStream << cudaGetErrorString(error) << "(" << error << ")" << std::endl;    \
            throw std::runtime_error(errorMessageStream.str());                                     \
        }                                                                                           \
    }

template<typename T>
using HostDeviceArray = std::pair < T*, T* > ;

int clz(unsigned int value)
{
#ifdef _WIN32
    unsigned long leadingZero = 0;
    if(_BitScanReverse(&leadingZero, value)) {
        return 31 - leadingZero;
    }
    else {
        return 32;
    }
#else
    return __builtin_clz(value);
#endif
}

//------------------------------------------------------------------------
// Timer
//------------------------------------------------------------------------
template<typename A = std::milli>
class Timer
{
public:
    Timer(const std::string &title) : m_Start(std::chrono::high_resolution_clock::now()), m_Title(title)
    {
    }

    ~Timer()
    {
        std::cout << m_Title << get() << std::endl;
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    double get() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = now - m_Start;
        return duration.count();
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
    std::string m_Title;
};

__device__ curandStatePhilox4_32_10_t d_rng;

__global__ void initializeRNGKernel(unsigned long long deviceRNGSeed) {
    if(threadIdx.x == 0) {
        curand_init(deviceRNGSeed, 0, 0, &d_rng);
    }
}

__global__ void initializeKernel32(float* RefracTime, float* V, curandState* rng, uint32_t numNeurons, unsigned long long deviceRNGSeed) {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    // only do this for existing neurons
    if(id < numNeurons) {
        curand_init(deviceRNGSeed, id, 0, &rng[id]);
        curandStatePhilox4_32_10_t localRNG = d_rng;
        skipahead_sequence((unsigned long long)id, &localRNG);
        {
            const float _scale = -5.000000000e+01f - -6.000000000e+01f;
            V[id] = -6.000000000e+01f + (curand_uniform(&localRNG) * _scale);
        }
        {
            RefracTime[id] = 0.0f;
        }
    }
}

__global__ void initializeKernel16(half* RefracTime, half* V, uint32_t numNeurons) {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    // only do this for existing neurons
    if(id < numNeurons) {
        curandStatePhilox4_32_10_t localRNG = d_rng;
        skipahead_sequence((unsigned long long)id, &localRNG);
        {
            const half _scale = static_cast<__half>(-5.000000000e+01) - static_cast<__half>(-6.000000000e+01);
            V[id] = static_cast<__half>(-6.000000000e+01) + (__float2half(curand_uniform(&localRNG)) * _scale);
        }
        {
            RefracTime[id] = 0.0f;
        }
    }
}

__global__ void updateNeuronsKernel32(float* RefracTime, float* V, uint32_t* recordSpk, curandState* rng, uint32_t numNeurons, unsigned int recordingTimestep)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ uint32_t shSpkRecord[1];
    if (threadIdx.x == 0) {
        shSpkRecord[0] = 0;
    }
    __syncthreads();
    // merged0
    if(id < numNeurons) {
        float Isyn = 0;
        float _lRefracTime = RefracTime[id];
        float _lV = V[id];
        {
            // current source 0
            Isyn += 0.000000000e+00f + (curand_normal(&rng[id]) * 1.000000000e+00f);
        }
        // test whether spike condition was fulfilled previously
        // calculate membrane potential
        if(_lRefracTime <= 0.0f) {
            float _alpha = ((Isyn + 0.000000000e+00f) * 2.000000000e+01f) + -4.900000000e+01f;
            _lV = _alpha - (9.512294245e-01f * (_alpha - _lV));
        }
        else {
            _lRefracTime -= 1.000000000e+00f;
        }
        
        // test for and register a true spike
        if ((_lRefracTime <= 0.0f && _lV >= -5.000000000e+01f)) {
            atomicOr(&shSpkRecord[0], 1 << threadIdx.x);
            // spike reset code
            _lV = -6.000000000e+01f;
            _lRefracTime = 5.000000000e+00f;
        }
        RefracTime[id] = _lRefracTime;
        V[id] = _lV;
    }
    __syncthreads();
    if(threadIdx.x < 1) {
        const unsigned int numRecordingWords = (numNeurons + 31) / 32;
        const unsigned int popWordIdx = (id / 32) + threadIdx.x;
        if(popWordIdx < numRecordingWords) {
            recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord[0];
        }
    }
}

__global__ void updateNeuronsKernelNaive16(half* RefracTime, half* V, uint32_t* recordSpk, curandState* rng, uint32_t numNeurons, unsigned int recordingTimestep)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ uint32_t shSpkRecord[1];
    if (threadIdx.x == 0) {
        shSpkRecord[0] = 0;
    }
    __syncthreads();
    // merged0
    if(id < numNeurons) {
        half Isyn = 0;
        half _lRefracTime = RefracTime[id];
        half _lV = V[id];
        {
            // current source 0
            Isyn += static_cast<__half>(0.000000000e+00) + (__float2half(curand_normal(&rng[id])) * static_cast<__half>(1.000000000e+00));
        }
        // test whether spike condition was fulfilled previously
        // calculate membrane potential
        if(_lRefracTime <= static_cast<__half>(0.0)) {
            half _alpha = ((Isyn + static_cast<__half>(0.000000000e+00)) * static_cast<__half>(2.000000000e+01)) + static_cast<__half>(-4.900000000e+01);
            _lV = _alpha - (static_cast<__half>(9.512294245e-01) * (_alpha - _lV));
        }
        else {
            _lRefracTime -= static_cast<__half>(1.000000000e+00);
        }
        
        // test for and register a true spike
        if ((_lRefracTime <= static_cast<__half>(0.0) && _lV >= static_cast<__half>(-5.000000000e+01))) {
            atomicOr(&shSpkRecord[0], 1 << threadIdx.x);
            // spike reset code
            _lV = static_cast<__half>(-6.000000000e+01);
            _lRefracTime = static_cast<__half>(5.000000000e+00);
        }
        RefracTime[id] = _lRefracTime;
        V[id] = _lV;
    }
    __syncthreads();
    if(threadIdx.x < 1) {
        const unsigned int numRecordingWords = (numNeurons + 31) / 32;
        const unsigned int popWordIdx = (id / 32) + threadIdx.x;
        if(popWordIdx < numRecordingWords) {
            recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord[0];
        }
    }
}

__global__ void updateNeuronsKernelVec16(half* RefracTime, half* V, uint32_t* recordSpk, curandState* rng, uint32_t numNeurons, unsigned int recordingTimestep)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ uint32_t shSpkRecord[2];
    if (threadIdx.x < 2) {
        shSpkRecord[threadIdx.x] = 0;
    }
    __syncthreads();
    // merged0
    if(id < (numNeurons / 2)) {
        half2 *RefracTimeVec = reinterpret_cast<half2*>(RefracTime);
        half2 *VVec = reinterpret_cast<half2*>(V);
        const half2 _lRefracTime = RefracTimeVec[id];
        const half2 _lV = VVec[id];
        
        // Lower
        half _llRefracTime = __low2half(_lRefracTime);
        half _llV = __low2half(_lV);
        {
            half Isyn = 0;
            
            {
                // current source 0
                Isyn += static_cast<__half>(0.000000000e+00) + (__float2half(curand_normal(&rng[id])) * static_cast<__half>(1.000000000e+00));
            }
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            if(_llRefracTime <= static_cast<__half>(0.0)) {
                half _alpha = ((Isyn + static_cast<__half>(0.000000000e+00)) * static_cast<__half>(2.000000000e+01)) + static_cast<__half>(-4.900000000e+01);
                _llV = _alpha - (static_cast<__half>(9.512294245e-01) * (_alpha - _llV));
            }
            else {
                _llRefracTime -= static_cast<__half>(1.000000000e+00);
            }
            
            // test for and register a true spike
            if ((_llRefracTime <= static_cast<__half>(0.0) && _llV >= static_cast<__half>(-5.000000000e+01))) {
                atomicOr(&shSpkRecord[threadIdx.x / 16], 1 << ((threadIdx.x % 16) * 2));
                // spike reset code
                _llV = static_cast<__half>(-6.000000000e+01);
                _llRefracTime = static_cast<__half>(5.000000000e+00);
            }
        }
        
        // Higher
        half _lhRefracTime = __high2half(_lRefracTime);
        half _lhV = __high2half(_lV);
        {
            half Isyn = 0;
            
            {
                // current source 0
                Isyn += static_cast<__half>(0.000000000e+00) + (__float2half(curand_normal(&rng[id])) * static_cast<__half>(1.000000000e+00));
            }
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            if(_lhRefracTime <= static_cast<__half>(0.0)) {
                half _alpha = ((Isyn + static_cast<__half>(0.000000000e+00)) * static_cast<__half>(2.000000000e+01)) + static_cast<__half>(-4.900000000e+01);
                _lhV = _alpha - (static_cast<__half>(9.512294245e-01) * (_alpha - _lhV));
            }
            else {
                _lhRefracTime -= static_cast<__half>(1.000000000e+00);
            }
            
            // test for and register a true spike
            if ((_lhRefracTime <= static_cast<__half>(0.0) && _lhV >= static_cast<__half>(-5.000000000e+01))) {
                atomicOr(&shSpkRecord[threadIdx.x / 16], 1 << (((threadIdx.x % 16) * 2) + 1));
                // spike reset code
                _lhV = static_cast<__half>(-6.000000000e+01);
                _lhRefracTime = static_cast<__half>(5.000000000e+00);
            }
        }
        
        
        // Re-combine
        RefracTimeVec[id] = __halves2half2(_llRefracTime, _lhRefracTime);
        VVec[id] = __halves2half2(_llV, _lhV);
    }
    __syncthreads();
    if(threadIdx.x < 2) {
        const unsigned int numRecordingWords = (numNeurons + 31) / 32;
        const unsigned int popWordIdx = (id / 16) + threadIdx.x;
        if(popWordIdx < numRecordingWords) {
            recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord[threadIdx.x];
        }
    }
}

//-----------------------------------------------------------------------------
// Host functions
//-----------------------------------------------------------------------------
template<typename T>
HostDeviceArray<T> allocateHostDevice(unsigned int count)
{
    T *array = nullptr;
    T *d_array = nullptr;
    CHECK_CUDA_ERRORS(cudaMallocHost(&array, count * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_array, count * sizeof(T)));

    return std::make_pair(array, d_array);
}
//-----------------------------------------------------------------------------
template<typename T>
void hostToDeviceCopy(HostDeviceArray<T> &array, unsigned int count, bool deleteHost=false)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.second, array.first, sizeof(T) * count, cudaMemcpyHostToDevice));
    if (deleteHost) {
        CHECK_CUDA_ERRORS(cudaFreeHost(array.first));
        array.first = nullptr;
    }
}
//-----------------------------------------------------------------------------
template<typename T>
void deviceToHostCopy(HostDeviceArray<T> &array, unsigned int count)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.first, array.second, sizeof(T) * count, cudaMemcpyDeviceToHost));
}
//-----------------------------------------------------------------------------
void saveSpikes(HostDeviceArray<uint32_t> &recordSpk, unsigned int numNeurons, unsigned int numTimesteps, const std::string &filename)
{
    const unsigned int numTimestepWords = (numNeurons + 31) / 32;
    const unsigned int numRecordWords = numTimesteps * numTimestepWords;
    deviceToHostCopy(recordSpk, numRecordWords);
            
    // Open file and write header
    std::ofstream file(filename);
    file << "Time [ms], Neuron ID" << std::endl;
    
    // Loop through timesteps
    const uint32_t *spkRecordWords = recordSpk.first;
    for(size_t t = 0; t < numTimesteps; t++) {
        // Loop through batched
        const double time = t * 1.0;

        // Loop through words representing timestep
        for(unsigned int w = 0; w < numTimestepWords; w++) {
            // Get word
            uint32_t spikeWord = *spkRecordWords++;
        
            // Calculate neuron id of highest bit of this word
            unsigned int neuronID = (w * 32) + 31;
        
            // While bits remain
            while(spikeWord != 0) {
                // Calculate leading zeros
                const int numLZ = clz(spikeWord);
            
                // If all bits have now been processed, zero spike word
                // Otherwise shift past the spike we have found
                spikeWord = (numLZ == 31) ? 0 : (spikeWord << (numLZ + 1));
            
                // Subtract number of leading zeros from neuron ID
                neuronID -= numLZ;
            
                // Add time and ID to vectors
                file << time << ", " << neuronID << std::endl;
            
                // New neuron id of the highest bit of this word
                neuronID--;
            }
        }
    }
}
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    try
    {
        const unsigned int numTimesteps = 1000;
        unsigned int numNeurons = (argc < 2) ? 1024 : std::stoul(argv[1]);
        const unsigned int numNeuronsVec = numNeurons / 2;
        assert((numNeurons % 32) == 0);
        CHECK_CUDA_ERRORS(cudaSetDevice(0));

        //------------------------------------------------------------------------
        // Configure input data
        //------------------------------------------------------------------------
        auto recordSpk = allocateHostDevice<uint32_t>(numTimesteps * ((numNeurons + 31) / 32));
        auto recordSpk16 = allocateHostDevice<uint32_t>(numTimesteps * ((numNeurons + 31) / 32));
        auto recordSpk16Vec = allocateHostDevice<uint32_t>(numTimesteps * ((numNeurons + 31) / 32));
        
        // Allocate device array for RNG
        curandState *d_rng = nullptr;
        float *d_refracTime = nullptr;
        float *d_v = nullptr;
        half *d_refracTime16 = nullptr;
        half *d_v16 = nullptr;
        half *d_refracTime16Vec = nullptr;
        half *d_v16Vec = nullptr;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_rng, numNeurons * sizeof(curandState)));
        CHECK_CUDA_ERRORS(cudaMalloc(&d_refracTime, numNeurons * sizeof(float)));
        CHECK_CUDA_ERRORS(cudaMalloc(&d_v, numNeurons * sizeof(float)));
        CHECK_CUDA_ERRORS(cudaMalloc(&d_refracTime16, numNeurons * sizeof(half)));
        CHECK_CUDA_ERRORS(cudaMalloc(&d_v16, numNeurons * sizeof(half)));
        CHECK_CUDA_ERRORS(cudaMalloc(&d_refracTime16Vec, numNeurons * sizeof(half)));
        CHECK_CUDA_ERRORS(cudaMalloc(&d_v16Vec, numNeurons * sizeof(half)));

        const dim3 threads(32, 1);
        const dim3 grid(((numNeurons + 31) / 32), 1);
        const dim3 vecGrid(((numNeuronsVec + 31) / 32), 1);
        
        // Initialize
        {
            unsigned long long deviceRNGSeed = 0;
            {
                std::random_device seedSource;
                uint32_t *deviceRNGSeedWord = reinterpret_cast<uint32_t*>(&deviceRNGSeed);
                for(int i = 0; i < 2; i++) {
                    deviceRNGSeedWord[i] = seedSource();
                }
            }
            initializeRNGKernel<<<1, 1>>>(deviceRNGSeed);
            CHECK_CUDA_ERRORS(cudaPeekAtLastError());
            {
                initializeKernel32<<<grid, threads>>>(d_refracTime, d_v, d_rng, numNeurons, deviceRNGSeed);
                CHECK_CUDA_ERRORS(cudaPeekAtLastError());
            }
            {
                initializeKernel16<<<grid, threads>>>(d_refracTime16, d_v16, numNeurons);
                CHECK_CUDA_ERRORS(cudaPeekAtLastError());
            }
            
            {
                initializeKernel16<<<grid, threads>>>(d_refracTime16Vec, d_v16Vec, numNeurons);
                CHECK_CUDA_ERRORS(cudaPeekAtLastError());
            }
        }

        {
            Timer<std::milli> t("Testing 32-bit:");
        
            for(int t = 0; t < numTimesteps; t++) {
                updateNeuronsKernel32<<<grid, threads>>>(d_refracTime, d_v, recordSpk.second, d_rng, numNeurons, t);
            }
            cudaDeviceSynchronize();
        }
        saveSpikes(recordSpk, numNeurons, numTimesteps, "spikes_32.csv");
        
         {
            Timer<std::milli> t("Testing 16-bit naive:");
        
            for(int t = 0; t < numTimesteps; t++) {
                updateNeuronsKernelNaive16<<<grid, threads>>>(d_refracTime16, d_v16, recordSpk16.second, d_rng, numNeurons, t);
            }
            cudaDeviceSynchronize();
        }
        saveSpikes(recordSpk16, numNeurons, numTimesteps, "spikes_16.csv");
        
        {
            Timer<std::milli> t("Testing 16-bit vec:");
        
            for(int t = 0; t < numTimesteps; t++) {
                updateNeuronsKernelVec16<<<vecGrid, threads>>>(d_refracTime16Vec, d_v16Vec, recordSpk16Vec.second, d_rng, numNeurons, t);
            }
            cudaDeviceSynchronize();
        }
        saveSpikes(recordSpk16Vec, numNeurons, numTimesteps, "spikes_16_vec.csv");
       
    }
    catch(const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

