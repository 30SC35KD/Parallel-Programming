#include <iostream>
#include <cuda_runtime.h>

// 错误检查函数
void checkCudaError(const char* errorMessage) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "错误: " << errorMessage << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 打印计算能力对应的架构
void printArchitecture(int major, int minor) {
    std::cout << "  计算能力: " << major << "." << minor << " (";
    if (major == 8 && minor >= 6) {
        std::cout << "Ampere)";
    } else if (major == 7 && minor >= 0) {
        std::cout << "Volta)";
    } else if (major == 6 && minor >= 1) {
        std::cout << "Pascal)";
    } else if (major == 5 && minor >= 0) {
        std::cout << "Maxwell)";
    } else if (major == 3 && minor >= 0) {
        std::cout << "Kepler)";
    } else if (major == 2 && minor >= 0) {
        std::cout << "Fermi)";
    } else {
        std::cout << "未知架构)";
    }
    std::cout << std::endl;
}

int main() {
    // 检测CUDA驱动和设备
    int driverVersion, runtimeVersion;
    checkCudaError("获取CUDA版本信息失败");
    
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    checkCudaError("获取设备数量失败");
    
    if (deviceCount == 0) {
        std::cout << "未检测到支持CUDA的GPU" << std::endl;
        return 0;
    }
    
    // 输出基础信息
    std::cout << "===== GPU基础信息检测 =====" << std::endl;
    std::cout << "系统中检测到 " << deviceCount << " 个CUDA GPU" << std::endl;
    std::cout << "CUDA驱动版本: " << driverVersion / 1000 << "." << (driverVersion % 1000) / 10 << std::endl;
    std::cout << "CUDA运行时版本: " << runtimeVersion / 1000 << "." << (runtimeVersion % 1000) / 10 << std::endl;
    std::cout << "------------------------" << std::endl;
    
    // 逐个输出每个GPU的基础信息
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "GPU " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  设备编号: " << i << std::endl;
        
        // 计算能力和架构
        printArchitecture(deviceProp.major, deviceProp.minor);
        
        // 内存信息
        size_t totalMem, freeMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        std::cout << "  全局内存: " << totalMem / (1024.0 * 1024.0 * 1024.0) << " GB (总计)" << std::endl;
        std::cout << "  可用内存: " << freeMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
        
        // 计算单元数量
        std::cout << "  计算单元数量: " << deviceProp.multiProcessorCount << std::endl;
        
        std::cout << "------------------------" << std::endl;
    }
    
    return 0;
}