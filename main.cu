#include <iostream>
#include <chrono>

using test_t = float;
constexpr std::size_t N = 5000;
constexpr std::size_t num_threads = 120;
constexpr std::size_t block_size = 1 << 8;
constexpr std::size_t test_count = 1 << 18;

__constant__ test_t const_mem[N];

template <std::size_t test_count>
__global__ void sequential_read(test_t* const result){
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	test_t sum = static_cast<test_t>(0);
	for(std::size_t i = 0; i < test_count; i++){
		for(std::size_t n = 0; n < N; n++){
			sum += const_mem[n];
		}
	}
	result[tid] = sum;
}
template <std::size_t test_count>
__global__ void sequential_read_dummy(test_t* const result){
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	test_t sum = static_cast<test_t>(0);
	for(std::size_t i = 0; i < test_count; i++){
		for(std::size_t n = 0; n < N; n++){
			sum += static_cast<test_t>(1.9);
		}
	}
	result[tid] = sum;
}

template <class Func>
double get_elapsed_time(Func func){
	const auto start_clock = std::chrono::system_clock::now();
	func();
	const auto end_clock = std::chrono::system_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() / 1.e6;
}

int main(){
	test_t *d_result;
	cudaMalloc(reinterpret_cast<void**>(&d_result), num_threads * sizeof(test_t));

	const auto t0 = get_elapsed_time(
			[&d_result](){
				sequential_read<test_count><<<(num_threads + block_size - 1)/block_size, block_size>>>(d_result);
				cudaDeviceSynchronize();
			});
	const auto t1 = get_elapsed_time(
			[&d_result](){
				sequential_read_dummy<test_count><<<(N + block_size - 1)/block_size, block_size>>>(d_result);
				cudaDeviceSynchronize();
			});
	std::cout<<"t0 - t1 = "<<(t0 - t1)<<std::endl;
	std::cout<<(N * test_count * sizeof(test_t) / (t0 - t1) / (1lu<<30))<<"GB/s"<<std::endl;
	cudaFree(d_result);
}
