#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fstream>
#include <vector>
#include <iostream>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 4) return 64;
        std::ifstream source(argv[2], std::ios::binary | std::ios::ate);
        if (!source) return 65;
        auto bytes = source.tellg();
        if (bytes <= 0 || bytes % 8 != 0) return 65;
        source.seekg(0);
        std::vector<char> input(static_cast<size_t>(bytes));
        source.read(input.data(), bytes);
        uint32_t count = static_cast<uint32_t>(bytes / 8);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 69;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
            error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"gfk_probe"];
        id<MTLComputePipelineState> pipeline = function ?
            [device newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!pipeline) { std::cerr << [[error description] UTF8String]; return 70; }
        id<MTLBuffer> in = [device newBufferWithBytes:input.data()
            length:input.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [device newBufferWithLength:size_t(count) * 5 * 8
            options:MTLResourceStorageModeShared];
        if (!in || !out) return 71;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:in offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(64, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::cerr << [[command.error description] UTF8String]; return 72;
        }
        std::ofstream result(argv[3], std::ios::binary);
        result.write(static_cast<const char *>(out.contents), out.length);
        if (!result) return 73;
        std::cout << [device.name UTF8String] << ": " << count << " synchronous rows\n";
    }
    return 0;
}
