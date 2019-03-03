#include <iostream>
#include <chrono>

using test_t = float;
constexpr std::size_t N = 5000;
constexpr std::size_t num_threads = 80;
constexpr std::size_t block_size = 1 << 7;
constexpr std::size_t test_count = 1 << 18;

__constant__ test_t const_mem[N];

template <std::size_t test_count>
__global__ void read_dummy(test_t* const result){
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	test_t sum = static_cast<test_t>(0);
	for(std::size_t i = 0; i < test_count; i++){
		for(std::size_t n = 0; n < N; n++){
			sum += static_cast<test_t>(1.9);
		}
	}
	result[tid] = sum;
}
template <std::size_t test_count>
__global__ void constant_read(test_t* const result){
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
__global__ void global_read(const test_t* const mem, test_t* const result){
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	test_t sum = static_cast<test_t>(0);
	for(std::size_t i = 0; i < test_count; i++){
		for(std::size_t n = 0; n < N; n++){
			sum += mem[n];
		}
	}
	result[tid] = sum;
}
template <std::size_t test_count>
__global__ void global_ldg_read(const test_t* const mem, test_t* const result){
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	test_t sum = static_cast<test_t>(0);
	for(std::size_t i = 0; i < test_count; i++){
		for(std::size_t n = 0; n < N; n++){
			sum += __ldg(mem + n);
		}
	}
	result[tid] = sum;
}
template <std::size_t test_count>
__global__ void global_ldg_dist_read(const test_t* const mem_head, test_t* const result){
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	const auto mem = mem_head + blockIdx.x * N;
	test_t sum = static_cast<test_t>(0);
	for(std::size_t i = 0; i < test_count; i++){
		for(std::size_t n = 0; n < N; n++){
			sum += __ldg(mem + n);
		}
	}
	result[tid] = sum;
}
template <std::size_t test_count>
__global__ void constant_shared_read(test_t* const result){
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	test_t sum = static_cast<test_t>(0);
	__shared__ test_t smem[N];
	for(std::size_t i = 0; i < test_count; i++){
		// はじめにgmem -> smem 
		for(std::size_t i = 0, index; (index = i + tid) < N; i+= block_size){
			smem[index] = const_mem[index];
		}
		for(std::size_t n = 0; n < N; n++){
			sum += *(smem + n);
		}
	}
	result[tid] = sum;
}
template <std::size_t test_count>
__global__ void global_shared_read(const test_t* const mem, test_t* const result){
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	test_t sum = static_cast<test_t>(0);
	__shared__ test_t smem[N];
	for(std::size_t i = 0; i < test_count; i++){
		// はじめにgmem -> smem 
		for(std::size_t i = 0, index; (index = i + tid) < N; i+= block_size){
			smem[index] = mem[index];
		}
		for(std::size_t n = 0; n < N; n++){
			sum += *(smem + n);
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
	auto get_speed = [](double dt){return (N * test_count * sizeof(test_t) / dt / (1lu<<30));};

	{
		test_t *d_result;
		cudaMalloc(reinterpret_cast<void**>(&d_result), num_threads * sizeof(test_t));
		const auto t0 = get_elapsed_time(
				[&d_result](){
				constant_read<test_count><<<(num_threads + block_size - 1)/block_size, block_size>>>(d_result);
				cudaDeviceSynchronize();
				});
		const auto t1 = get_elapsed_time(
				[&d_result](){
				read_dummy<test_count><<<(N + block_size - 1)/block_size, block_size>>>(d_result);
				cudaDeviceSynchronize();
				});
		std::cout<<"constant   : "<<get_speed(t0 - t1)<<" GB/s"<<std::endl;
		cudaFree(d_result);
	}
	{
		test_t *d_result;
		test_t *d_global;
		cudaMalloc(reinterpret_cast<void**>(&d_result), num_threads * sizeof(test_t));
		cudaMalloc(reinterpret_cast<void**>(&d_global), N * sizeof(test_t));
		const auto t0 = get_elapsed_time(
				[&d_result, &d_global](){
				global_read<test_count><<<(num_threads + block_size - 1)/block_size, block_size>>>(d_global, d_result);
				cudaDeviceSynchronize();
				});
		const auto t1 = get_elapsed_time(
				[&d_result, &d_global](){
				read_dummy<test_count><<<(N + block_size - 1)/block_size, block_size>>>(d_result);
				cudaDeviceSynchronize();
				});
		std::cout<<"global     : "<<get_speed(t0 - t1)<<" GB/s"<<std::endl;
		cudaFree(d_result);
		cudaFree(d_global);
	}
	{
		test_t *d_result;
		test_t *d_global;
		cudaMalloc(reinterpret_cast<void**>(&d_result), num_threads * sizeof(test_t));
		cudaMalloc(reinterpret_cast<void**>(&d_global), N * sizeof(test_t));
		const auto t0 = get_elapsed_time(
				[&d_result, &d_global](){
				global_ldg_read<test_count><<<(num_threads + block_size - 1)/block_size, block_size>>>(d_global, d_result);
				cudaDeviceSynchronize();
				});
		const auto t1 = get_elapsed_time(
				[&d_result, &d_global](){
				read_dummy<test_count><<<(N + block_size - 1)/block_size, block_size>>>(d_result);
				cudaDeviceSynchronize();
				});
		std::cout<<"global ldg : "<<get_speed(t0 - t1)<<" GB/s"<<std::endl;
		cudaFree(d_result);
		cudaFree(d_global);
	}
	{
		constexpr auto grid_size = (num_threads + block_size - 1)/block_size;
		test_t *d_result;
		test_t *d_global;
		cudaMalloc(reinterpret_cast<void**>(&d_result), num_threads * sizeof(test_t));
		cudaMalloc(reinterpret_cast<void**>(&d_global), N * sizeof(test_t) * grid_size);
		const auto t0 = get_elapsed_time(
				[&d_result, &d_global](){
				global_ldg_dist_read<test_count><<<grid_size, block_size>>>(d_global, d_result);
				cudaDeviceSynchronize();
				});
		const auto t1 = get_elapsed_time(
				[&d_result, &d_global](){
				read_dummy<test_count><<<grid_size, block_size>>>(d_result);
				cudaDeviceSynchronize();
				});
		std::cout<<"dist ldg   : "<<get_speed(t0 - t1)<<" GB/s"<<std::endl;
		cudaFree(d_result);
		cudaFree(d_global);
	}
	{
		test_t *d_result;
		cudaMalloc(reinterpret_cast<void**>(&d_result), num_threads * sizeof(test_t));
		const auto t0 = get_elapsed_time(
				[&d_result](){
				constant_shared_read<test_count><<<(num_threads + block_size - 1)/block_size, block_size>>>(d_result);
				cudaDeviceSynchronize();
				});
		const auto t1 = get_elapsed_time(
				[&d_result](){
				read_dummy<test_count><<<(N + block_size - 1)/block_size, block_size>>>(d_result);
				cudaDeviceSynchronize();
				});
		std::cout<<"s constant : "<<get_speed(t0 - t1)<<" GB/s"<<std::endl;
		cudaFree(d_result);
	}
	{
		constexpr auto grid_size = (num_threads + block_size - 1)/block_size;
		test_t *d_result;
		test_t *d_global;
		cudaMalloc(reinterpret_cast<void**>(&d_result), num_threads * sizeof(test_t));
		cudaMalloc(reinterpret_cast<void**>(&d_global), N * sizeof(test_t));
		const auto t0 = get_elapsed_time(
				[&d_result, &d_global](){
				global_shared_read<test_count><<<grid_size, block_size>>>(d_global, d_result);
				cudaDeviceSynchronize();
				});
		const auto t1 = get_elapsed_time(
				[&d_result, &d_global](){
				read_dummy<test_count><<<grid_size, block_size>>>(d_result);
				cudaDeviceSynchronize();
				});
		std::cout<<"shared     : "<<get_speed(t0 - t1)<<" GB/s"<<std::endl;
		cudaFree(d_result);
		cudaFree(d_global);
	}
}
