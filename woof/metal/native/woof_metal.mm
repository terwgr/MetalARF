#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <mach/mach.h>
#include <mach/task_info.h>
#include <mach/host_info.h>

#include "woof_metal.h"

#include <algorithm>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Context {
    woof_metal_handle handle = 0;
    id<MTLDevice> __strong device = nil;
    id<MTLCommandQueue> __strong queue = nil;
    id<MTLLibrary> __strong default_library = nil;
    id<MTLLibrary> __strong exact_library = nil;
    id<MTLComputePipelineState> __strong copy_pipeline = nil;
    id<MTLComputePipelineState> __strong zero_pipeline = nil;
    id<MTLComputePipelineState> __strong fill_pipeline = nil;
    id<MTLComputePipelineState> __strong acoustic_pipeline = nil;
    id<MTLComputePipelineState> __strong acoustic_exact_pipeline = nil;
    id<MTLArgumentEncoder> __strong copy_arguments = nil;
    id<MTLArgumentEncoder> __strong zero_arguments = nil;
    id<MTLArgumentEncoder> __strong fill_arguments = nil;
    struct KernelPipeline {
        id<MTLComputePipelineState> __strong pipeline = nil;
        id<MTLArgumentEncoder> __strong arguments = nil;
    };
    std::unordered_map<std::string, std::shared_ptr<KernelPipeline>> kernel_cache;
    struct FFTPlan {
        MPSGraph *__strong graph = nil;
        MPSGraphTensor *__strong input = nil;
        MPSGraphTensor *__strong output = nil;
    };
    std::unordered_map<std::string, std::shared_ptr<FFTPlan>> fft_cache;
    std::deque<std::string> fft_cache_order;
    std::mutex encode_mutex;
    std::mutex allocation_mutex;
    uint64_t live_allocated_bytes = 0;
    uint64_t live_requested_bytes = 0;
    uint64_t live_high_water_allocated_bytes = 0;
    uint64_t pending_allocated_bytes = 0;
    uint64_t live_allocated_by_category[WOOF_METAL_ALLOCATION_CATEGORY_COUNT] = {};
    uint64_t live_requested_by_category[WOOF_METAL_ALLOCATION_CATEGORY_COUNT] = {};
    uint64_t fft_plan_cache_builds = 0;
    uint64_t fft_plan_cache_hits = 0;
    uint64_t fft_plan_cache_evictions = 0;
    uint64_t fft_plan_cache_entries = 0;
    uint64_t fft_graph_executions_started = 0;
    uint64_t fft_graph_executions_completed = 0;
    uint64_t fft_graph_executions_failed = 0;
    uint64_t fft_graph_pending = 0;
};

static_assert(sizeof(woof_metal_calc_coefs_params) == 44,
              "calc_coefs ABI parameter size changed");
static_assert(offsetof(woof_metal_calc_coefs_params, dtau) == 28,
              "calc_coefs ABI scalar offset changed");
static_assert(sizeof(woof_metal_kernel_resource) == 32,
              "generic kernel resource ABI layout changed");
static_assert(offsetof(woof_metal_kernel_batch_item, entrypoint) == 0 &&
              offsetof(woof_metal_kernel_batch_item, cache_identity) == 8 &&
              offsetof(woof_metal_kernel_batch_item, exact_policy) == 16 &&
              offsetof(woof_metal_kernel_batch_item, resource_start) == 20 &&
              offsetof(woof_metal_kernel_batch_item, resource_count) == 24 &&
              offsetof(woof_metal_kernel_batch_item, parameter_offset) == 28 &&
              offsetof(woof_metal_kernel_batch_item, parameter_size) == 32 &&
              offsetof(woof_metal_kernel_batch_item, grid_size) == 40 &&
              offsetof(woof_metal_kernel_batch_item, threads_per_group) == 48 &&
              offsetof(woof_metal_kernel_batch_item, reserved) == 52 &&
              sizeof(woof_metal_kernel_batch_item) == 56,
              "kernel batch item ABI layout changed");
static_assert(sizeof(woof_metal_allocation_snapshot) == 200,
              "allocation snapshot ABI layout changed");

struct Allocation {
    std::shared_ptr<Context> context;
    id<MTLBuffer> __strong resource = nil;
    uint64_t allocated_bytes = 0;
    uint64_t requested_bytes = 0;
    uint32_t category = WOOF_METAL_ALLOCATION_BACKING;
    uint64_t pending_references = 0;

    ~Allocation() {
        if (!context || !allocated_bytes || category >= WOOF_METAL_ALLOCATION_CATEGORY_COUNT)
            return;
        std::lock_guard<std::mutex> lock(context->allocation_mutex);
        context->live_allocated_bytes -= allocated_bytes;
        context->live_requested_bytes -= requested_bytes;
        context->live_allocated_by_category[category] -= allocated_bytes;
        context->live_requested_by_category[category] -= requested_bytes;
        if (pending_references)
            context->pending_allocated_bytes -= allocated_bytes;
    }
};

std::shared_ptr<Allocation> track_allocation(
        const std::shared_ptr<Context> &context, id<MTLBuffer> resource,
        uint64_t requested_bytes, uint32_t category) {
    if (!resource || category >= WOOF_METAL_ALLOCATION_CATEGORY_COUNT)
        return nullptr;
    try {
        auto allocation = std::make_shared<Allocation>();
        allocation->context = context;
        allocation->resource = resource;
        allocation->allocated_bytes = static_cast<uint64_t>(resource.allocatedSize);
        allocation->requested_bytes = requested_bytes;
        allocation->category = category;
        {
            std::lock_guard<std::mutex> lock(context->allocation_mutex);
            context->live_allocated_bytes += allocation->allocated_bytes;
            context->live_requested_bytes += requested_bytes;
            context->live_allocated_by_category[category] += allocation->allocated_bytes;
            context->live_requested_by_category[category] += requested_bytes;
            context->live_high_water_allocated_bytes = std::max(
                context->live_high_water_allocated_bytes,
                context->live_allocated_bytes);
        }
        return allocation;
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

struct Buffer {
    woof_metal_handle handle = 0;
    std::shared_ptr<Context> context;
    id<MTLBuffer> __strong resource = nil;
    std::shared_ptr<Allocation> allocation;
    uint64_t generation = 0;
};

struct Submission {
    woof_metal_handle handle = 0;
    std::shared_ptr<Context> context;
    std::mutex mutex;
    std::condition_variable completed_condition;
    bool callback_driven = false;
    bool completed = false;
    int32_t completion_result = WOOF_METAL_OK;
    std::string completion_error;
    id<MTLCommandBuffer> __strong command = nil;
    id<MTLBuffer> __strong arguments = nil;
    std::shared_ptr<Allocation> argument_allocation;
    std::vector<id<MTLBuffer>> argument_buffers;
    std::vector<std::shared_ptr<Allocation>> argument_allocations;
    std::vector<std::shared_ptr<Buffer>> resources;
    std::vector<std::shared_ptr<Allocation>> allocations;
    bool allocation_pending = false;
    NSMutableArray<id> *__strong graph_resources = nil;
    std::shared_ptr<Context::FFTPlan> fft_plan;
};

void prepare_submission_allocations(
        const std::shared_ptr<Submission> &submission,
        const std::vector<std::shared_ptr<Buffer>> &buffers,
        const std::shared_ptr<Allocation> &argument = nullptr) {
    submission->allocations.reserve(buffers.size() + (argument ? 1u : 0u));
    for (const auto &buffer : buffers)
        if (buffer && buffer->allocation)
            submission->allocations.push_back(buffer->allocation);
    if (argument) submission->allocations.push_back(argument);
    auto &allocations = submission->allocations;
    std::sort(allocations.begin(), allocations.end(),
        [](const auto &a, const auto &b) {
            return std::less<const Allocation *>{}(a.get(), b.get());
        });
    allocations.erase(std::unique(allocations.begin(), allocations.end()),
                      allocations.end());
}

void prepare_submission_allocations(
        const std::shared_ptr<Submission> &submission,
        const std::vector<std::shared_ptr<Buffer>> &buffers,
        const std::vector<std::shared_ptr<Allocation>> &arguments) {
    submission->allocations.reserve(buffers.size() + arguments.size());
    for (const auto &buffer : buffers)
        if (buffer && buffer->allocation)
            submission->allocations.push_back(buffer->allocation);
    for (const auto &argument : arguments)
        if (argument)
            submission->allocations.push_back(argument);
    auto &allocations = submission->allocations;
    std::sort(allocations.begin(), allocations.end(),
        [](const auto &a, const auto &b) {
            return std::less<const Allocation *>{}(a.get(), b.get());
        });
    allocations.erase(std::unique(allocations.begin(), allocations.end()),
                      allocations.end());
}

void begin_submission_pending(const std::shared_ptr<Submission> &submission) {
    if (submission->allocation_pending) return;
    auto &context = submission->context;
    {
        std::lock_guard<std::mutex> lock(context->allocation_mutex);
        for (const auto &allocation : submission->allocations) {
            if (allocation->pending_references++ == 0)
                context->pending_allocated_bytes += allocation->allocated_bytes;
        }
    }
    submission->allocation_pending = true;
}

void release_submission_allocations(
        const std::shared_ptr<Submission> &submission) {
    if (!submission->allocation_pending) return;
    auto &context = submission->context;
    {
        std::lock_guard<std::mutex> lock(context->allocation_mutex);
        for (const auto &allocation : submission->allocations) {
            if (allocation->pending_references &&
                --allocation->pending_references == 0)
                context->pending_allocated_bytes -= allocation->allocated_bytes;
        }
    }
    submission->allocation_pending = false;
    submission->allocations.clear();
}

std::mutex g_registry_mutex;
std::unordered_map<woof_metal_handle, std::shared_ptr<Context>> g_contexts;
std::unordered_map<woof_metal_handle, std::shared_ptr<Buffer>> g_buffers;
std::unordered_map<woof_metal_handle, std::shared_ptr<Submission>> g_submissions;
woof_metal_handle g_next_handle = 1;
uint64_t g_next_generation = 1;
thread_local std::string g_last_error;

woof_metal_handle next_handle_locked() {
    if (g_next_handle == 0 || g_next_handle == UINT64_MAX)
        return 0;
    return g_next_handle++;
}

int32_t fail(int32_t code, const std::string &message) {
    g_last_error = message;
    return code;
}

std::shared_ptr<Context> context_for(woof_metal_handle handle) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto found = g_contexts.find(handle);
    return found == g_contexts.end() ? nullptr : found->second;
}

std::shared_ptr<Buffer> buffer_for(woof_metal_handle handle) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto found = g_buffers.find(handle);
    return found == g_buffers.end() ? nullptr : found->second;
}

std::shared_ptr<Submission> submission_for(woof_metal_handle handle) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto found = g_submissions.find(handle);
    return found == g_submissions.end() ? nullptr : found->second;
}

void forget_submission(woof_metal_handle handle) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_submissions.erase(handle);
}

int32_t finish_submission(const std::shared_ptr<Submission> &submission,
                          bool wait) {
    if (!submission)
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal submission");
    id<MTLCommandBuffer> command = nil;
    {
        std::unique_lock<std::mutex> lock(submission->mutex);
        if (submission->callback_driven && wait && !submission->completed)
            submission->completed_condition.wait(
                lock, [&] { return submission->completed; });
        if (submission->completed) {
            return submission->completion_result == WOOF_METAL_OK
                ? WOOF_METAL_OK
                : fail(submission->completion_result,
                       submission->completion_error);
        }
        if (submission->callback_driven)
            return WOOF_METAL_OK;
        command = submission->command;
    }
    if (!command)
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    "Metal submission lost its command before completion");
    if (wait) [command waitUntilCompleted];
    else if (command.status < MTLCommandBufferStatusCompleted)
        return WOOF_METAL_OK;

    int32_t result = WOOF_METAL_OK;
    std::string message;
    if (command.status == MTLCommandBufferStatusError) {
        NSString *description = command.error.localizedDescription;
        const char *utf8 = description ? description.UTF8String : nullptr;
        result = WOOF_METAL_SUBMISSION_FAILED;
        message = utf8 ? utf8 : "Metal command buffer failed";
    } else if (command.status != MTLCommandBufferStatusCompleted) {
        result = WOOF_METAL_RUNTIME_ERROR;
        message = "Metal command buffer ended without completing";
    }
    {
        std::lock_guard<std::mutex> lock(submission->mutex);
        if (!submission->completed) {
            submission->completed = true;
            submission->completion_result = result;
            submission->completion_error = message;
            submission->resources.clear();
            submission->arguments = nil;
            submission->argument_allocation.reset();
            submission->argument_buffers.clear();
            submission->argument_allocations.clear();
            release_submission_allocations(submission);
            submission->command = nil;
        }
        result = submission->completion_result;
        message = submission->completion_error;
    }
    if (result != WOOF_METAL_OK)
        return fail(result, message);
    g_last_error.clear();
    return WOOF_METAL_OK;
}

void retire_submission_after_completion(
        const std::shared_ptr<Submission> &submission,
        id<MTLCommandBuffer> command) {
    std::weak_ptr<Submission> weak_submission = submission;
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        auto retained = weak_submission.lock();
        if (!retained) return;
        int32_t result = WOOF_METAL_OK;
        std::string message;
        if (completed.status == MTLCommandBufferStatusError) {
            NSString *description = completed.error.localizedDescription;
            const char *utf8 = description ? description.UTF8String : nullptr;
            result = WOOF_METAL_SUBMISSION_FAILED;
            message = utf8 ? utf8 : "Metal command buffer failed";
        } else if (completed.status != MTLCommandBufferStatusCompleted) {
            result = WOOF_METAL_RUNTIME_ERROR;
            message = "Metal command buffer ended without completing";
        }
        std::lock_guard<std::mutex> lock(retained->mutex);
        if (retained->completed) return;
        retained->completed = true;
        retained->completion_result = result;
        retained->completion_error = std::move(message);
        retained->resources.clear();
        retained->arguments = nil;
        retained->argument_allocation.reset();
        retained->argument_buffers.clear();
        retained->argument_allocations.clear();
        release_submission_allocations(retained);
        retained->command = nil;
        retained->graph_resources = nil;
        retained->fft_plan.reset();
        retained->completed_condition.notify_all();
    }];
}

void complete_graph_submission(const std::weak_ptr<Submission> &weak_submission,
                               NSError *error) {
    auto submission = weak_submission.lock();
    if (!submission) return;
    std::lock_guard<std::mutex> lock(submission->mutex);
    if (submission->completed) return;
    if (error) {
        const char *utf8 = error.localizedDescription.UTF8String;
        submission->completion_result = WOOF_METAL_SUBMISSION_FAILED;
        submission->completion_error = utf8 ? utf8 : "MPSGraph execution failed";
    } else {
        submission->completion_result = WOOF_METAL_OK;
    }
    submission->completed = true;
    {
        std::lock_guard<std::mutex> allocation_lock(
            submission->context->allocation_mutex);
        if (submission->context->fft_graph_pending)
            --submission->context->fft_graph_pending;
        if (error)
            ++submission->context->fft_graph_executions_failed;
        else
            ++submission->context->fft_graph_executions_completed;
    }
    submission->resources.clear();
    release_submission_allocations(submission);
    submission->graph_resources = nil;
    submission->fft_plan.reset();
    submission->completed_condition.notify_all();
}

/*
 * The completion callback retires argument and resource ownership promptly;
 * the small status record remains until Python polls/waits and destroys the
 * token. This keeps ignored asynchronous submissions from pinning GPU memory.
 */
int32_t ensure_completed_submission(const std::shared_ptr<Submission> &submission) {
    std::lock_guard<std::mutex> lock(submission->mutex);
    if (!submission->completed)
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    "Metal submission has not completed");
    if (submission->completion_result != WOOF_METAL_OK)
        return fail(submission->completion_result, submission->completion_error);
    g_last_error.clear();
    return WOOF_METAL_OK;
}

bool valid_range(uint64_t offset, uint64_t size, uint64_t capacity) {
    return offset <= capacity && size <= capacity - offset;
}

bool tensor_byte_count(const uint64_t *shape, uint32_t rank,
                       uint64_t element_size, uint64_t *out_bytes) {
    if (!shape || !out_bytes || rank < 2 || rank > 16) return false;
    uint64_t elements = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (shape[i] == 0 || shape[i] > NSUIntegerMax ||
            elements > UINT64_MAX / shape[i])
            return false;
        elements *= shape[i];
    }
    if (elements > UINT64_MAX / element_size) return false;
    *out_bytes = elements * element_size;
    return true;
}

std::string fft_plan_key(const uint64_t *input_shape,
                         const uint64_t *output_shape, uint32_t rank,
                         bool inverse) {
    std::string key = inverse ? "irfft2:" : "rfft2:";
    key += std::to_string(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        key += ":";
        key += std::to_string(input_shape[i]);
    }
    key += "->";
    for (uint32_t i = 0; i < rank; ++i) {
        key += ":";
        key += std::to_string(output_shape[i]);
    }
    return key;
}

std::shared_ptr<Context::FFTPlan> fft_plan_for(
        const std::shared_ptr<Context> &context,
        const uint64_t *input_shape, const uint64_t *output_shape,
        uint32_t rank, bool inverse) {
    const std::string key = fft_plan_key(input_shape, output_shape, rank, inverse);
    auto found = context->fft_cache.find(key);
    if (found != context->fft_cache.end()) {
        std::lock_guard<std::mutex> lock(context->allocation_mutex);
        ++context->fft_plan_cache_hits;
        return found->second;
    }

    auto plan = std::make_shared<Context::FFTPlan>();
    plan->graph = [MPSGraph new];
    plan->graph.options = MPSGraphOptionsNone;
    NSMutableArray<NSNumber *> *input_dims =
        [NSMutableArray arrayWithCapacity:rank];
    NSMutableArray<NSNumber *> *output_dims =
        [NSMutableArray arrayWithCapacity:rank];
    for (uint32_t i = 0; i < rank; ++i) {
        [input_dims addObject:@(input_shape[i])];
        [output_dims addObject:@(output_shape[i])];
    }
    plan->input = [plan->graph placeholderWithShape:input_dims
        dataType:inverse ? MPSDataTypeComplexFloat32 : MPSDataTypeFloat32
        name:@"woof_fft2_input"];
    MPSGraphFFTDescriptor *descriptor = [MPSGraphFFTDescriptor descriptor];
    descriptor.inverse = inverse;
    descriptor.scalingMode = inverse ? MPSGraphFFTScalingModeSize
                                     : MPSGraphFFTScalingModeNone;
    descriptor.roundToOddHermitean = inverse && (output_shape[rank - 1] & 1u);
    NSArray<NSNumber *> *axes = @[@(rank - 2), @(rank - 1)];
    if (inverse) {
        plan->output = [plan->graph HermiteanToRealFFTWithTensor:plan->input
            axes:axes descriptor:descriptor name:@"woof_irfft2"];
    } else {
        plan->output = [plan->graph realToHermiteanFFTWithTensor:plan->input
            axes:axes descriptor:descriptor name:@"woof_rfft2"];
    }
    context->fft_cache.emplace(key, plan);
    context->fft_cache_order.push_back(key);
    {
        std::lock_guard<std::mutex> lock(context->allocation_mutex);
        ++context->fft_plan_cache_builds;
        ++context->fft_plan_cache_entries;
    }
    constexpr size_t kFFTPlanCacheCapacity = 8;
    while (context->fft_cache_order.size() > kFFTPlanCacheCapacity) {
        context->fft_cache.erase(context->fft_cache_order.front());
        context->fft_cache_order.pop_front();
        std::lock_guard<std::mutex> lock(context->allocation_mutex);
        ++context->fft_plan_cache_evictions;
        --context->fft_plan_cache_entries;
    }
    return plan;
}

int32_t synchronize_queue_locked(const std::shared_ptr<Context> &context) {
    id<MTLCommandBuffer> barrier = [context->queue commandBuffer];
    if (!barrier)
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    "could not create a Metal host-transfer barrier");
    [barrier commit];
    [barrier waitUntilCompleted];
    if (barrier.status == MTLCommandBufferStatusError) {
        NSString *description = barrier.error.localizedDescription;
        const char *utf8 = description ? description.UTF8String : nullptr;
        return fail(WOOF_METAL_SUBMISSION_FAILED,
                    utf8 ? utf8 : "Metal host-transfer barrier failed");
    }
    return WOOF_METAL_OK;
}

int32_t create_pipeline(id<MTLLibrary> library, NSString *name,
                        id<MTLComputePipelineState> __strong *pipeline,
                        id<MTLArgumentEncoder> __strong *arguments,
                        id<MTLDevice> device) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    std::string("missing Metal function ") + name.UTF8String);
    }
    *arguments = [function newArgumentEncoderWithBufferIndex:0];
    if (!*arguments || (*arguments).encodedLength == 0) {
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    std::string("could not create argument encoder for ") +
                        name.UTF8String);
    }
    *pipeline = [device newComputePipelineStateWithFunction:function
                                                      error:&error];
    if (!*pipeline) {
        NSString *description = error.localizedDescription;
        const char *utf8 = description ? description.UTF8String : nullptr;
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    utf8 ? utf8 : "could not create Metal compute pipeline");
    }
    if ((*pipeline).maxTotalThreadsPerThreadgroup < 128) {
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    "Metal pipeline supports fewer than 128 threads per group");
    }
    return WOOF_METAL_OK;
}

int32_t create_direct_pipeline(id<MTLLibrary> library, NSString *name,
                               id<MTLComputePipelineState> __strong *pipeline,
                               id<MTLDevice> device) {
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function)
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    std::string("missing Metal function ") + name.UTF8String);
    *pipeline = [device newComputePipelineStateWithFunction:function
                                                       error:&error];
    if (!*pipeline) {
        NSString *description = error.localizedDescription;
        const char *utf8 = description ? description.UTF8String : nullptr;
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    utf8 ? utf8 : "could not create Metal compute pipeline");
    }
    if ((*pipeline).maxTotalThreadsPerThreadgroup < 128)
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    "Metal calc_coefs pipeline supports fewer than 128 threads");
    return WOOF_METAL_OK;
}

}  // namespace

extern "C" int32_t woof_metal_context_create(
        const char *metallib_path, const char *exact_metallib_path,
        woof_metal_handle *out_context) {
    @autoreleasepool {
        if (!metallib_path || !exact_metallib_path || !out_context)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "both Metal libraries and output handle are required");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "Metal reports no available device");
        if (device.argumentBuffersSupport < MTLArgumentBuffersTier2)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "WOOF Metal requires argument buffer tier 2 support");
        NSString *path = [NSString stringWithUTF8String:metallib_path];
        if (!path)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "metallib path is not valid UTF-8");
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                                      error:&error];
        if (!library) {
            NSString *description = error.localizedDescription;
            const char *utf8 = description ? description.UTF8String : nullptr;
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        utf8 ? utf8 : "could not load Metal library");
        }
        NSString *exact_path = [NSString stringWithUTF8String:exact_metallib_path];
        if (!exact_path)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "exact metallib path is not valid UTF-8");
        NSError *exact_error = nil;
        id<MTLLibrary> exact_library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:exact_path] error:&exact_error];
        if (!exact_library) {
            NSString *description = exact_error.localizedDescription;
            const char *utf8 = description ? description.UTF8String : nullptr;
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        utf8 ? utf8 : "could not load exact Metal library");
        }
        auto context = std::make_shared<Context>();
        context->device = device;
        context->queue = [device newCommandQueue];
        context->default_library = library;
        context->exact_library = exact_library;
        if (!context->queue)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "could not create Metal command queue");
        int32_t result = create_pipeline(library, @"rk_copy_words",
                                         &context->copy_pipeline,
                                         &context->copy_arguments, device);
        if (result != WOOF_METAL_OK) return result;
        result = create_pipeline(library, @"rk_zero_words",
                                 &context->zero_pipeline,
                                 &context->zero_arguments, device);
        if (result != WOOF_METAL_OK) return result;
        result = create_pipeline(library, @"rk_fill_words",
                                 &context->fill_pipeline,
                                 &context->fill_arguments, device);
        if (result != WOOF_METAL_OK) return result;
        result = create_direct_pipeline(library, @"calc_coefs",
                                        &context->acoustic_pipeline, device);
        if (result != WOOF_METAL_OK) return result;
        result = create_direct_pipeline(exact_library, @"calc_coefs",
                                        &context->acoustic_exact_pipeline,
                                        device);
        if (result != WOOF_METAL_OK) return result;
        {
            std::lock_guard<std::mutex> lock(g_registry_mutex);
            context->handle = next_handle_locked();
            if (!context->handle)
                return fail(WOOF_METAL_RUNTIME_ERROR,
                            "Metal handle space exhausted");
            g_contexts.emplace(context->handle, context);
        }
        *out_context = context->handle;
        g_last_error.clear();
        return WOOF_METAL_OK;
    }
}

extern "C" int32_t woof_metal_context_destroy(woof_metal_handle handle) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto found = g_contexts.find(handle);
    if (found == g_contexts.end())
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal context handle");
    for (const auto &entry : g_buffers) {
        if (entry.second->context->handle == handle)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "Metal context still owns registered buffers");
    }
    for (const auto &entry : g_submissions) {
        if (entry.second->context->handle == handle)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "Metal context still owns in-flight submissions");
    }
    g_contexts.erase(found);
    g_last_error.clear();
    return WOOF_METAL_OK;
}

extern "C" int32_t woof_metal_context_device_info(
        woof_metal_handle handle, char *name, uint64_t capacity,
        uint64_t *registry_id,
        uint64_t *max_buffer_length, uint64_t *recommended_working_set) {
    auto context = context_for(handle);
    if (!context)
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal context handle");
    if (!name || capacity == 0 || !registry_id || !max_buffer_length ||
        !recommended_working_set)
        return fail(WOOF_METAL_INVALID_ARGUMENT,
                    "device info output pointers are required");
    const char *device_name = context->device.name.UTF8String;
    if (!device_name)
        return fail(WOOF_METAL_RUNTIME_ERROR, "Metal device has no UTF-8 name");
    const size_t bytes = std::strlen(device_name);
    if (bytes + 1 > capacity)
        return fail(WOOF_METAL_INVALID_ARGUMENT,
                    "device name output buffer is too small");
    std::memcpy(name, device_name, bytes + 1);
    *registry_id = context->device.registryID;
    *max_buffer_length = context->device.maxBufferLength;
    *recommended_working_set = context->device.recommendedMaxWorkingSetSize;
    g_last_error.clear();
    return WOOF_METAL_OK;
}

extern "C" int32_t woof_metal_context_allocation_snapshot(
        woof_metal_handle handle, woof_metal_allocation_snapshot *snapshot) {
    auto context = context_for(handle);
    if (!context)
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal context handle");
    if (!snapshot || snapshot->struct_size != sizeof(*snapshot))
        return fail(WOOF_METAL_INVALID_ARGUMENT,
                    "allocation snapshot has an invalid struct size");
    std::memset(snapshot, 0, sizeof(*snapshot));
    snapshot->struct_size = sizeof(*snapshot);
    std::lock_guard<std::mutex> lock(context->allocation_mutex);
    snapshot->category_count = WOOF_METAL_ALLOCATION_CATEGORY_COUNT;
    snapshot->live_allocated_bytes = context->live_allocated_bytes;
    snapshot->live_requested_bytes = context->live_requested_bytes;
    snapshot->live_high_water_allocated_bytes = context->live_high_water_allocated_bytes;
    snapshot->pending_allocated_bytes = context->pending_allocated_bytes;
    snapshot->device_current_allocated_bytes = context->device.currentAllocatedSize;
    snapshot->memory_observation_validity |=
        WOOF_METAL_MEMORY_OBS_DEVICE_ALLOCATION;
    snapshot->fft_plan_cache_entries = context->fft_plan_cache_entries;
    snapshot->fft_plan_cache_builds = context->fft_plan_cache_builds;
    snapshot->fft_plan_cache_hits = context->fft_plan_cache_hits;
    snapshot->fft_plan_cache_evictions = context->fft_plan_cache_evictions;
    snapshot->fft_graph_executions_started =
        context->fft_graph_executions_started;
    snapshot->fft_graph_executions_completed =
        context->fft_graph_executions_completed;
    snapshot->fft_graph_executions_failed =
        context->fft_graph_executions_failed;
    snapshot->fft_graph_pending = context->fft_graph_pending;
    for (uint32_t i = 0; i < WOOF_METAL_ALLOCATION_CATEGORY_COUNT; ++i) {
        snapshot->live_allocated_by_category[i] = context->live_allocated_by_category[i];
        snapshot->live_requested_by_category[i] = context->live_requested_by_category[i];
    }
    task_vm_info_data_t task_info_data = {};
    mach_msg_type_number_t task_info_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&task_info_data),
                  &task_info_count) == KERN_SUCCESS) {
        snapshot->process_phys_footprint_bytes = task_info_data.phys_footprint;
        snapshot->memory_observation_validity |=
            WOOF_METAL_MEMORY_OBS_PROCESS_FOOTPRINT;
    }
    vm_statistics64_data_t vm_info = {};
    mach_msg_type_number_t vm_info_count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm_info),
                          &vm_info_count) == KERN_SUCCESS) {
        const uint64_t available_pages = static_cast<uint64_t>(vm_info.free_count)
            + static_cast<uint64_t>(vm_info.inactive_count)
            + static_cast<uint64_t>(vm_info.speculative_count);
        snapshot->system_available_estimate_bytes = available_pages * vm_page_size;
        snapshot->memory_observation_validity |=
            WOOF_METAL_MEMORY_OBS_SYSTEM_AVAILABLE;
    }
    g_last_error.clear();
    return WOOF_METAL_OK;
}

extern "C" int32_t woof_metal_buffer_create(
        woof_metal_handle context_handle, uint64_t size,
        woof_metal_handle *out_buffer) {
    auto context = context_for(context_handle);
    if (!context)
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal context handle");
    if (!out_buffer || size == 0 || size > context->device.maxBufferLength ||
        size > std::numeric_limits<NSUInteger>::max())
        return fail(WOOF_METAL_INVALID_ARGUMENT,
                    "invalid Metal buffer size or output handle");
    id<MTLBuffer> resource = [context->device
        newBufferWithLength:static_cast<NSUInteger>(size)
                    options:MTLResourceStorageModeShared];
    if (!resource)
        return fail(WOOF_METAL_OUT_OF_MEMORY,
                    "Metal could not allocate a shared buffer");
    auto buffer = std::make_shared<Buffer>();
    buffer->context = context;
    buffer->resource = resource;
    buffer->allocation = track_allocation(
        context, resource, size, WOOF_METAL_ALLOCATION_BACKING);
    if (!buffer->allocation)
        return fail(WOOF_METAL_OUT_OF_MEMORY,
                    "could not record Metal buffer allocation");
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        buffer->handle = next_handle_locked();
        if (!buffer->handle)
            return fail(WOOF_METAL_RUNTIME_ERROR, "Metal handle space exhausted");
        buffer->generation = g_next_generation++;
        g_buffers.emplace(buffer->handle, buffer);
    }
    *out_buffer = buffer->handle;
    g_last_error.clear();
    return WOOF_METAL_OK;
}

extern "C" int32_t woof_metal_buffer_destroy(woof_metal_handle handle) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto found = g_buffers.find(handle);
    if (found == g_buffers.end())
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal buffer handle");
    g_buffers.erase(found);
    g_last_error.clear();
    return WOOF_METAL_OK;
}

extern "C" int32_t woof_metal_buffer_write(
        woof_metal_handle handle, uint64_t offset, const void *source,
        uint64_t size) {
    auto buffer = buffer_for(handle);
    if (!buffer)
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal buffer handle");
    if ((!source && size) || !valid_range(offset, size, buffer->resource.length))
        return fail(WOOF_METAL_INVALID_ARGUMENT, "Metal buffer write is out of bounds");
    std::lock_guard<std::mutex> transfer_lock(buffer->context->encode_mutex);
    const int32_t sync_result = synchronize_queue_locked(buffer->context);
    if (sync_result != WOOF_METAL_OK) return sync_result;
    if (size) std::memcpy(static_cast<uint8_t *>(buffer->resource.contents) + offset,
                          source, static_cast<size_t>(size));
    g_last_error.clear();
    return WOOF_METAL_OK;
}

extern "C" int32_t woof_metal_buffer_read(
        woof_metal_handle handle, uint64_t offset, void *destination,
        uint64_t size) {
    auto buffer = buffer_for(handle);
    if (!buffer)
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal buffer handle");
    if ((!destination && size) || !valid_range(offset, size, buffer->resource.length))
        return fail(WOOF_METAL_INVALID_ARGUMENT, "Metal buffer read is out of bounds");
    std::lock_guard<std::mutex> transfer_lock(buffer->context->encode_mutex);
    const int32_t sync_result = synchronize_queue_locked(buffer->context);
    if (sync_result != WOOF_METAL_OK) return sync_result;
    if (size) std::memcpy(destination,
                          static_cast<const uint8_t *>(buffer->resource.contents) + offset,
                          static_cast<size_t>(size));
    g_last_error.clear();
    return WOOF_METAL_OK;
}

extern "C" int32_t woof_metal_buffer_size(woof_metal_handle handle,
                                           uint64_t *size) {
    auto buffer = buffer_for(handle);
    if (!buffer)
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal buffer handle");
    if (!size)
        return fail(WOOF_METAL_INVALID_ARGUMENT, "buffer size output is required");
    *size = buffer->resource.length;
    g_last_error.clear();
    return WOOF_METAL_OK;
}

extern "C" int32_t woof_metal_submit_rk_bookkeeping(
        woof_metal_handle context_handle, int32_t zero, uint32_t pair_count,
        const woof_metal_handle *source_handles, const uint64_t *source_offsets,
        const woof_metal_handle *target_handles, const uint64_t *target_offsets,
        const uint64_t *word_counts, woof_metal_handle *out_submission) {
    @autoreleasepool {
        auto context = context_for(context_handle);
        if (!context)
            return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal context handle");
        if (!out_submission || !target_handles || !target_offsets || !word_counts ||
            (!zero && (!source_handles || !source_offsets)) ||
            pair_count == 0 || pair_count > 64)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "invalid RK bookkeeping resource table");

        std::vector<std::shared_ptr<Buffer>> sources(pair_count);
        std::vector<std::shared_ptr<Buffer>> targets(pair_count);
        uint64_t max_words = 0;
        for (uint32_t i = 0; i < pair_count; ++i) {
            targets[i] = buffer_for(target_handles[i]);
            if (!targets[i] || targets[i]->context != context)
                return fail(WOOF_METAL_INVALID_HANDLE,
                            "target buffer is invalid or belongs to another device");
            if ((target_offsets[i] & 3u) ||
                word_counts[i] > (targets[i]->resource.length -
                    std::min<uint64_t>(target_offsets[i], targets[i]->resource.length)) / 4u ||
                target_offsets[i] > targets[i]->resource.length)
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "RK target range is unaligned or out of bounds");
            if (!zero) {
                sources[i] = buffer_for(source_handles[i]);
                if (!sources[i] || sources[i]->context != context)
                    return fail(WOOF_METAL_INVALID_HANDLE,
                                "source buffer is invalid or belongs to another device");
                if ((source_offsets[i] & 3u) ||
                    source_offsets[i] > sources[i]->resource.length ||
                    word_counts[i] > (sources[i]->resource.length -
                        std::min<uint64_t>(source_offsets[i], sources[i]->resource.length)) / 4u)
                    return fail(WOOF_METAL_INVALID_ARGUMENT,
                                "RK source range is unaligned or out of bounds");
            }
            max_words = std::max(max_words, word_counts[i]);
        }
        if (!max_words)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "RK bookkeeping cannot dispatch empty arrays");

        std::lock_guard<std::mutex> encode_lock(context->encode_mutex);
        id<MTLArgumentEncoder> encoder = zero ? context->zero_arguments
                                               : context->copy_arguments;
        id<MTLBuffer> argument_buffer = [context->device
            newBufferWithLength:encoder.encodedLength
                        options:MTLResourceStorageModeShared];
        if (!argument_buffer)
            return fail(WOOF_METAL_OUT_OF_MEMORY,
                        "Metal could not allocate the RK argument buffer");
        auto argument_allocation = track_allocation(
            context, argument_buffer, encoder.encodedLength,
            WOOF_METAL_ALLOCATION_RK_ARGUMENT);
        if (!argument_allocation)
            return fail(WOOF_METAL_OUT_OF_MEMORY,
                        "could not record the RK argument buffer allocation");
        [encoder setArgumentBuffer:argument_buffer offset:0];
        uint64_t counts[64] = {};
        for (uint32_t i = 0; i < pair_count; ++i) {
            counts[i] = word_counts[i];
            if (!zero) {
                [encoder setBuffer:sources[i]->resource
                            offset:static_cast<NSUInteger>(source_offsets[i])
                           atIndex:i];
            }
            [encoder setBuffer:targets[i]->resource
                        offset:static_cast<NSUInteger>(target_offsets[i])
                       atIndex:64u + i];
        }
        void *count_data = [encoder constantDataAtIndex:128];
        if (!count_data)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "Metal argument buffer has no word-count data slot");
        std::memcpy(count_data, counts, sizeof(counts));

        id<MTLCommandBuffer> command = [context->queue commandBuffer];
        if (!command)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "could not allocate a Metal command buffer");
        id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
        if (!compute)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "could not allocate a Metal compute encoder");
        [compute setComputePipelineState:(zero ? context->zero_pipeline
                                               : context->copy_pipeline)];
        [compute setBuffer:argument_buffer offset:0 atIndex:0];
        for (uint32_t i = 0; i < pair_count; ++i) {
            if (!zero)
                [compute useResource:sources[i]->resource
                               usage:MTLResourceUsageRead];
            [compute useResource:targets[i]->resource
                           usage:MTLResourceUsageWrite];
        }
        const uint64_t groups = (max_words + 511u) / 512u;
        if (groups > std::numeric_limits<NSUInteger>::max()) {
            [compute endEncoding];
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "RK dispatch grid exceeds Metal geometry limits");
        }
        [compute dispatchThreadgroups:MTLSizeMake(
            static_cast<NSUInteger>(groups), pair_count, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [compute endEncoding];

        auto submission = std::make_shared<Submission>();
        submission->context = context;
        submission->command = command;
        submission->arguments = argument_buffer;
        submission->argument_allocation = argument_allocation;
        submission->resources.reserve(pair_count * (zero ? 1u : 2u));
        for (uint32_t i = 0; i < pair_count; ++i) {
            if (!zero) submission->resources.push_back(sources[i]);
            submission->resources.push_back(targets[i]);
        }
        prepare_submission_allocations(submission, submission->resources,
                                       argument_allocation);
        {
            std::lock_guard<std::mutex> lock(g_registry_mutex);
            submission->handle = next_handle_locked();
            if (!submission->handle)
                return fail(WOOF_METAL_RUNTIME_ERROR, "Metal handle space exhausted");
            g_submissions.emplace(submission->handle, submission);
        }
        begin_submission_pending(submission);
        retire_submission_after_completion(submission, command);
        [command commit];
        *out_submission = submission->handle;
        g_last_error.clear();
        return WOOF_METAL_OK;
    }
}

extern "C" int32_t woof_metal_submit_fill(
        woof_metal_handle context_handle, woof_metal_handle target_handle,
        uint64_t target_offset, uint64_t word_count,
        const uint32_t *word_pattern, uint32_t pattern_word_count,
        woof_metal_handle *out_submission) {
    @autoreleasepool {
        auto context = context_for(context_handle);
        auto target = buffer_for(target_handle);
        if (!context)
            return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal context handle");
        if (!target || target->context != context)
            return fail(WOOF_METAL_INVALID_HANDLE,
                        "fill target is invalid or belongs to another device");
        if (!out_submission || !word_pattern || pattern_word_count == 0 ||
            pattern_word_count > 4 || target_offset > target->resource.length ||
            (target_offset & 3u) ||
            word_count > (target->resource.length - target_offset) / 4u ||
            word_count == 0 || word_count % pattern_word_count != 0)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "Metal fill range is empty, unaligned, or out of bounds");

        std::lock_guard<std::mutex> encode_lock(context->encode_mutex);
        id<MTLArgumentEncoder> encoder = context->fill_arguments;
        id<MTLBuffer> argument_buffer = [context->device
            newBufferWithLength:encoder.encodedLength
                        options:MTLResourceStorageModeShared];
        if (!argument_buffer)
            return fail(WOOF_METAL_OUT_OF_MEMORY,
                        "Metal could not allocate the fill argument buffer");
        auto argument_allocation = track_allocation(
            context, argument_buffer, encoder.encodedLength,
            WOOF_METAL_ALLOCATION_FILL_ARGUMENT);
        if (!argument_allocation)
            return fail(WOOF_METAL_OUT_OF_MEMORY,
                        "could not record the fill argument buffer allocation");
        [encoder setArgumentBuffer:argument_buffer offset:0];
        [encoder setBuffer:target->resource offset:static_cast<NSUInteger>(target_offset)
                    atIndex:64];
        uint64_t counts[64] = {};
        counts[0] = word_count;
        void *count_data = [encoder constantDataAtIndex:128];
        if (!count_data)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "Metal fill argument buffer has no word-count data slot");
        std::memcpy(count_data, counts, sizeof(counts));

        id<MTLCommandBuffer> command = [context->queue commandBuffer];
        if (!command)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "could not allocate a Metal fill command buffer");
        id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
        if (!compute)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "could not allocate a Metal fill compute encoder");
        [compute setComputePipelineState:context->fill_pipeline];
        [compute setBuffer:argument_buffer offset:0 atIndex:0];
        [compute setBytes:word_pattern
                   length:pattern_word_count * sizeof(uint32_t) atIndex:1];
        [compute setBytes:&pattern_word_count
                   length:sizeof(pattern_word_count) atIndex:2];
        [compute useResource:target->resource usage:MTLResourceUsageWrite];
        const uint64_t groups = (word_count + 511u) / 512u;
        if (groups > std::numeric_limits<NSUInteger>::max()) {
            [compute endEncoding];
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "Metal fill dispatch grid exceeds geometry limits");
        }
        [compute dispatchThreadgroups:MTLSizeMake(
            static_cast<NSUInteger>(groups), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [compute endEncoding];

        auto submission = std::make_shared<Submission>();
        submission->context = context;
        submission->command = command;
        submission->arguments = argument_buffer;
        submission->argument_allocation = argument_allocation;
        submission->resources.push_back(target);
        prepare_submission_allocations(submission, submission->resources,
                                       argument_allocation);
        {
            std::lock_guard<std::mutex> lock(g_registry_mutex);
            submission->handle = next_handle_locked();
            if (!submission->handle)
                return fail(WOOF_METAL_RUNTIME_ERROR, "Metal handle space exhausted");
            g_submissions.emplace(submission->handle, submission);
        }
        begin_submission_pending(submission);
        retire_submission_after_completion(submission, command);
        [command commit];
        *out_submission = submission->handle;
        g_last_error.clear();
        return WOOF_METAL_OK;
    }
}

extern "C" int32_t woof_metal_submit_acoustic_calc_coefs(
        woof_metal_handle context_handle, const woof_metal_handle *handles,
        const uint64_t *offsets,
        const woof_metal_calc_coefs_params *parameters,
        woof_metal_handle *out_submission) {
    @autoreleasepool {
        auto context = context_for(context_handle);
        if (!context)
            return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal context handle");
        if (!handles || !offsets || !parameters || !out_submission ||
            parameters->struct_size != sizeof(*parameters) ||
            parameters->nz == 0 || parameters->ny == 0 || parameters->nx == 0 ||
            (parameters->moist_cq != 0 && parameters->moist_cq != 1) ||
            (parameters->top_lid != 0 && parameters->top_lid != 1) ||
            (parameters->wrf_exact != 0 && parameters->wrf_exact != 1))
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "invalid calc_coefs dimensions or parameter layout");

        const uint64_t columns = uint64_t(parameters->ny) * parameters->nx;
        const uint64_t layers = uint64_t(parameters->nz) + 1;
        if (columns > UINT64_MAX / layers ||
            layers * columns > UINT64_MAX / sizeof(float))
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "calc_coefs dimensions overflow byte-size arithmetic");
        const uint64_t full_words = uint64_t(parameters->nz) * columns;
        const uint64_t half_words = layers * columns;
        const uint64_t profile_words = parameters->nz;
        const uint64_t edge_words = layers;
        uint64_t required[16] = {
            full_words * sizeof(float), full_words * sizeof(float),
            columns * sizeof(float), full_words * sizeof(float),
            half_words * sizeof(float), half_words * sizeof(float),
            half_words * sizeof(float), profile_words * sizeof(float),
            profile_words * sizeof(float), profile_words * sizeof(float),
            profile_words * sizeof(float), edge_words * sizeof(float),
            edge_words * sizeof(float), columns * sizeof(float),
            (parameters->moist_cq ? half_words : 1) * sizeof(float), 0};
        std::vector<std::shared_ptr<Buffer>> buffers(15);
        for (uint32_t i = 0; i < buffers.size(); ++i) {
            buffers[i] = buffer_for(handles[i]);
            if (!buffers[i] || buffers[i]->context != context)
                return fail(WOOF_METAL_INVALID_HANDLE,
                            "calc_coefs resource is invalid or belongs to another device");
            if ((offsets[i] & 3u) || !valid_range(
                    offsets[i], required[i], buffers[i]->resource.length))
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "calc_coefs resource range is unaligned or out of bounds");
        }

        std::lock_guard<std::mutex> encode_lock(context->encode_mutex);
        id<MTLCommandBuffer> command = [context->queue commandBuffer];
        if (!command)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "could not allocate a Metal calc_coefs command buffer");
        id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
        if (!compute)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "could not allocate a Metal calc_coefs encoder");
        id<MTLComputePipelineState> pipeline = parameters->wrf_exact
            ? context->acoustic_exact_pipeline : context->acoustic_pipeline;
        [compute setComputePipelineState:pipeline];
        for (uint32_t i = 0; i < buffers.size(); ++i) {
            [compute setBuffer:buffers[i]->resource
                       offset:static_cast<NSUInteger>(offsets[i])
                      atIndex:i];
            [compute useResource:buffers[i]->resource
                           usage:(i >= 3 && i <= 6)
                               ? MTLResourceUsageWrite
                               : (i < 3 || i >= 7 ? MTLResourceUsageRead
                                                  : MTLResourceUsageRead)];
        }
        [compute setBytes:parameters length:sizeof(*parameters) atIndex:15];
        const uint64_t groups = (columns + 127u) / 128u;
        if (groups > std::numeric_limits<NSUInteger>::max()) {
            [compute endEncoding];
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "calc_coefs dispatch grid exceeds Metal limits");
        }
        [compute dispatchThreadgroups:MTLSizeMake(
            static_cast<NSUInteger>(groups), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [compute endEncoding];

        auto submission = std::make_shared<Submission>();
        submission->context = context;
        submission->command = command;
        submission->resources = std::move(buffers);
        prepare_submission_allocations(submission, submission->resources);
        {
            std::lock_guard<std::mutex> lock(g_registry_mutex);
            submission->handle = next_handle_locked();
            if (!submission->handle)
                return fail(WOOF_METAL_RUNTIME_ERROR,
                            "Metal handle space exhausted");
            g_submissions.emplace(submission->handle, submission);
        }
        begin_submission_pending(submission);
        retire_submission_after_completion(submission, command);
        [command commit];
        *out_submission = submission->handle;
        g_last_error.clear();
        return WOOF_METAL_OK;
    }
}

extern "C" int32_t woof_metal_submit_kernel(
        woof_metal_handle context_handle, const char *entrypoint_c,
        const char *cache_identity_c, int32_t exact_policy,
        const woof_metal_kernel_resource *resource_specs,
        uint32_t resource_count, const void *parameters,
        uint32_t parameter_size, uint64_t grid_size,
        uint32_t threads_per_group,
        woof_metal_handle *out_submission) {
    @autoreleasepool {
        auto context = context_for(context_handle);
        if (!context)
            return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal context handle");
        if (!entrypoint_c || !cache_identity_c || !resource_specs ||
            resource_count == 0 || resource_count > 64 || !parameters ||
            parameter_size == 0 || parameter_size > 4096 || grid_size == 0 ||
            threads_per_group == 0 || !out_submission ||
            (exact_policy != 0 && exact_policy != 1))
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "invalid generic Metal kernel submission metadata");
        NSString *entrypoint = [NSString stringWithUTF8String:entrypoint_c];
        if (!entrypoint)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "Metal entrypoint is not valid UTF-8");
        std::string cache_key(cache_identity_c);
        cache_key += exact_policy ? "|wrf-exact|" : "|default|";
        cache_key += entrypoint_c;

        std::vector<std::shared_ptr<Buffer>> buffers(resource_count);
        std::vector<uint64_t> offsets(resource_count);
        for (uint32_t i = 0; i < resource_count; ++i) {
            const auto &spec = resource_specs[i];
            if (spec.reserved != 0 ||
                (spec.access != MTLResourceUsageRead &&
                 spec.access != (MTLResourceUsageRead | MTLResourceUsageWrite)))
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "generic kernel resource access descriptor is invalid");
            buffers[i] = buffer_for(spec.buffer);
            if (!buffers[i] || buffers[i]->context != context)
                return fail(WOOF_METAL_INVALID_HANDLE,
                            "generic kernel resource is invalid or belongs to another device");
            if ((spec.byte_offset & 3u) || (spec.byte_length & 3u) ||
                !valid_range(spec.byte_offset, spec.byte_length,
                             buffers[i]->resource.length))
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "generic kernel resource range is unaligned or out of bounds");
            offsets[i] = spec.byte_offset;
        }

        std::lock_guard<std::mutex> encode_lock(context->encode_mutex);
        auto found = context->kernel_cache.find(cache_key);
        if (found == context->kernel_cache.end()) {
            id<MTLLibrary> selected_library = exact_policy
                ? context->exact_library : context->default_library;
            id<MTLFunction> function = [selected_library
                newFunctionWithName:entrypoint];
            if (!function)
                return fail(WOOF_METAL_RUNTIME_ERROR,
                            std::string("missing Metal function ") + entrypoint_c);
            auto cached = std::make_shared<Context::KernelPipeline>();
            cached->arguments = [function newArgumentEncoderWithBufferIndex:0];
            if (!cached->arguments || !cached->arguments.encodedLength)
                return fail(WOOF_METAL_RUNTIME_ERROR,
                            std::string("cannot encode argument buffer for ") + entrypoint_c);
            NSError *error = nil;
            cached->pipeline = [context->device
                newComputePipelineStateWithFunction:function error:&error];
            if (!cached->pipeline) {
                NSString *description = error.localizedDescription;
                const char *utf8 = description ? description.UTF8String : nullptr;
                return fail(WOOF_METAL_RUNTIME_ERROR,
                            utf8 ? utf8 : "could not create Metal compute pipeline");
            }
            if (threads_per_group > cached->pipeline.maxTotalThreadsPerThreadgroup)
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "requested threadgroup exceeds the pipeline limit");
            found = context->kernel_cache.emplace(cache_key, cached).first;
        }
        auto pipeline = found->second;
        if (threads_per_group > pipeline->pipeline.maxTotalThreadsPerThreadgroup)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "requested threadgroup exceeds the pipeline limit");

        id<MTLBuffer> argument_buffer = [context->device
            newBufferWithLength:pipeline->arguments.encodedLength
                        options:MTLResourceStorageModeShared];
        if (!argument_buffer)
            return fail(WOOF_METAL_OUT_OF_MEMORY,
                        "could not allocate the kernel argument buffer");
        auto argument_allocation = track_allocation(
            context, argument_buffer, pipeline->arguments.encodedLength,
            WOOF_METAL_ALLOCATION_KERNEL_ARGUMENT);
        if (!argument_allocation)
            return fail(WOOF_METAL_OUT_OF_MEMORY,
                        "could not record the kernel argument buffer allocation");
        [pipeline->arguments setArgumentBuffer:argument_buffer offset:0];
        for (uint32_t i = 0; i < resource_count; ++i)
            [pipeline->arguments setBuffer:buffers[i]->resource
                                    offset:static_cast<NSUInteger>(offsets[i])
                                   atIndex:i];

        id<MTLCommandBuffer> command = [context->queue commandBuffer];
        if (!command)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "could not allocate a generic Metal command buffer");
        id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
        if (!compute)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "could not allocate a generic Metal compute encoder");
        [compute setComputePipelineState:pipeline->pipeline];
        [compute setBuffer:argument_buffer offset:0 atIndex:0];
        [compute setBytes:parameters length:parameter_size atIndex:1];
        std::unordered_map<void *, MTLResourceUsage> usages;
        for (uint32_t i = 0; i < resource_count; ++i) {
            void *key = (__bridge void *)buffers[i]->resource;
            auto item = usages.find(key);
            if (item == usages.end())
                usages.emplace(key, static_cast<MTLResourceUsage>(resource_specs[i].access));
            else
                item->second = static_cast<MTLResourceUsage>(
                    item->second | static_cast<MTLResourceUsage>(resource_specs[i].access));
        }
        for (const auto &item : usages) {
            id<MTLBuffer> resource = (__bridge id<MTLBuffer>)item.first;
            [compute useResource:resource usage:item.second];
        }
        const uint64_t groups = grid_size / threads_per_group
                              + uint64_t(grid_size % threads_per_group != 0);
        if (groups > std::numeric_limits<NSUInteger>::max()) {
            [compute endEncoding];
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "generic Metal dispatch grid exceeds geometry limits");
        }
        [compute dispatchThreadgroups:MTLSizeMake(
            static_cast<NSUInteger>(groups), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
        [compute endEncoding];

        auto submission = std::make_shared<Submission>();
        submission->context = context;
        submission->command = command;
        submission->arguments = argument_buffer;
        submission->argument_allocation = argument_allocation;
        submission->resources = std::move(buffers);
        prepare_submission_allocations(submission, submission->resources,
                                       argument_allocation);
        {
            std::lock_guard<std::mutex> lock(g_registry_mutex);
            submission->handle = next_handle_locked();
            if (!submission->handle)
                return fail(WOOF_METAL_RUNTIME_ERROR,
                            "Metal handle space exhausted");
            g_submissions.emplace(submission->handle, submission);
        }
        begin_submission_pending(submission);
        retire_submission_after_completion(submission, command);
        [command commit];
        *out_submission = submission->handle;
        g_last_error.clear();
        return WOOF_METAL_OK;
    }
}

extern "C" int32_t woof_metal_submit_kernel_batch(
        woof_metal_handle context_handle,
        const woof_metal_kernel_batch_item *items,
        uint32_t item_count,
        const woof_metal_kernel_resource *resource_specs,
        uint32_t resource_count, const uint8_t *parameter_bytes,
        uint32_t parameter_bytes_size,
        woof_metal_handle *out_submission) {
    @autoreleasepool {
        auto context = context_for(context_handle);
        if (!context)
            return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal context handle");
        constexpr uint32_t max_batch_items = 64;
        constexpr uint32_t max_resources_per_item = 64;
        constexpr uint32_t max_parameter_size_per_item = 4096;
        constexpr uint32_t max_flat_resources =
            max_batch_items * max_resources_per_item;
        constexpr uint32_t max_parameter_bytes =
            max_batch_items * max_parameter_size_per_item;
        if (!items || item_count == 0 || item_count > max_batch_items ||
            !resource_specs || resource_count == 0 ||
            resource_count > max_flat_resources || !parameter_bytes ||
            parameter_bytes_size == 0 ||
            parameter_bytes_size > max_parameter_bytes || !out_submission)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "invalid Metal kernel batch table metadata");

        struct PreparedItem {
            std::shared_ptr<Context::KernelPipeline> pipeline;
            id<MTLBuffer> __strong arguments = nil;
            std::shared_ptr<Allocation> argument_allocation;
            uint32_t resource_start = 0;
            uint32_t resource_count = 0;
            uint32_t parameter_offset = 0;
            uint32_t parameter_size = 0;
            uint64_t grid_size = 0;
            uint32_t threads_per_group = 0;
        };
        std::vector<PreparedItem> prepared(item_count);
        std::vector<std::shared_ptr<Buffer>> buffers(resource_count);
        std::vector<uint64_t> offsets(resource_count);
        std::vector<bool> resource_slot_used(resource_count, false);
        std::vector<std::pair<uint32_t, uint32_t>> parameter_ranges;
        parameter_ranges.reserve(item_count);

        // Validate all descriptors and resolve every handle before allocating
        // command resources. The disjoint flat slices make ownership and
        // parameter provenance unambiguous to the Python caller.
        for (uint32_t i = 0; i < item_count; ++i) {
            const auto &item = items[i];
            if (!item.entrypoint || !item.cache_identity ||
                item.exact_policy < 0 || item.exact_policy > 1 ||
                item.reserved != 0 || item.resource_count == 0 ||
                item.resource_count > max_resources_per_item ||
                item.resource_start > resource_count ||
                item.resource_count > resource_count - item.resource_start ||
                item.parameter_size == 0 ||
                item.parameter_size > max_parameter_size_per_item ||
                item.parameter_offset > parameter_bytes_size ||
                item.parameter_size > parameter_bytes_size - item.parameter_offset ||
                item.grid_size == 0 || item.threads_per_group == 0)
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "invalid Metal kernel batch item descriptor");
            if (item.entrypoint[0] == '\0' || item.cache_identity[0] == '\0')
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "Metal kernel batch identity must be nonempty");
            NSString *entrypoint = [NSString stringWithUTF8String:item.entrypoint];
            if (!entrypoint)
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "Metal batch entrypoint is not valid UTF-8");
            parameter_ranges.emplace_back(item.parameter_offset,
                                          item.parameter_offset + item.parameter_size);
            auto &slot = prepared[i];
            slot.resource_start = item.resource_start;
            slot.resource_count = item.resource_count;
            slot.parameter_offset = item.parameter_offset;
            slot.parameter_size = item.parameter_size;
            slot.grid_size = item.grid_size;
            slot.threads_per_group = item.threads_per_group;
            for (uint32_t j = 0; j < item.resource_count; ++j) {
                const uint32_t resource_index = item.resource_start + j;
                if (resource_slot_used[resource_index])
                    return fail(WOOF_METAL_INVALID_ARGUMENT,
                                "Metal kernel batch resource slices overlap");
                resource_slot_used[resource_index] = true;
            }
        }
        std::sort(parameter_ranges.begin(), parameter_ranges.end());
        for (size_t i = 1; i < parameter_ranges.size(); ++i)
            if (parameter_ranges[i].first < parameter_ranges[i - 1].second)
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "Metal kernel batch parameter ranges overlap");
        for (uint32_t i = 0; i < resource_count; ++i) {
            if (!resource_slot_used[i])
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "Metal kernel batch contains an unreferenced resource");
            const auto &spec = resource_specs[i];
            if (spec.reserved != 0 ||
                (spec.access != MTLResourceUsageRead &&
                 spec.access != (MTLResourceUsageRead | MTLResourceUsageWrite)))
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "Metal batch resource access descriptor is invalid");
            buffers[i] = buffer_for(spec.buffer);
            if (!buffers[i] || buffers[i]->context != context)
                return fail(WOOF_METAL_INVALID_HANDLE,
                            "Metal batch resource is invalid or belongs to another device");
            if ((spec.byte_offset & 3u) || (spec.byte_length & 3u) ||
                !valid_range(spec.byte_offset, spec.byte_length,
                             buffers[i]->resource.length))
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "Metal batch resource range is unaligned or out of bounds");
            offsets[i] = spec.byte_offset;
        }

        std::lock_guard<std::mutex> encode_lock(context->encode_mutex);
        // Build/resolve every pipeline and argument buffer before creating a
        // command buffer. A failure here therefore submits no partial batch.
        for (uint32_t i = 0; i < item_count; ++i) {
            const auto &item = items[i];
            NSString *entrypoint = [NSString stringWithUTF8String:item.entrypoint];
            std::string cache_key(item.cache_identity);
            cache_key += item.exact_policy ? "|wrf-exact|" : "|default|";
            cache_key += item.entrypoint;
            auto found = context->kernel_cache.find(cache_key);
            if (found == context->kernel_cache.end()) {
                id<MTLLibrary> selected_library = item.exact_policy
                    ? context->exact_library : context->default_library;
                id<MTLFunction> function = [selected_library
                    newFunctionWithName:entrypoint];
                if (!function)
                    return fail(WOOF_METAL_RUNTIME_ERROR,
                                std::string("missing Metal function ") + item.entrypoint);
                auto cached = std::make_shared<Context::KernelPipeline>();
                cached->arguments = [function
                    newArgumentEncoderWithBufferIndex:0];
                if (!cached->arguments || !cached->arguments.encodedLength)
                    return fail(WOOF_METAL_RUNTIME_ERROR,
                                std::string("cannot encode argument buffer for ") +
                                item.entrypoint);
                NSError *error = nil;
                cached->pipeline = [context->device
                    newComputePipelineStateWithFunction:function error:&error];
                if (!cached->pipeline) {
                    NSString *description = error.localizedDescription;
                    const char *utf8 = description ? description.UTF8String : nullptr;
                    return fail(WOOF_METAL_RUNTIME_ERROR,
                                utf8 ? utf8 : "could not create Metal compute pipeline");
                }
                found = context->kernel_cache.emplace(cache_key, cached).first;
            }
            auto &slot = prepared[i];
            slot.pipeline = found->second;
            if (slot.threads_per_group >
                slot.pipeline->pipeline.maxTotalThreadsPerThreadgroup)
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "requested batch threadgroup exceeds a pipeline limit");
            id<MTLBuffer> argument_buffer = [context->device
                newBufferWithLength:slot.pipeline->arguments.encodedLength
                            options:MTLResourceStorageModeShared];
            if (!argument_buffer)
                return fail(WOOF_METAL_OUT_OF_MEMORY,
                            "could not allocate a Metal batch argument buffer");
            slot.argument_allocation = track_allocation(
                context, argument_buffer, slot.pipeline->arguments.encodedLength,
                WOOF_METAL_ALLOCATION_KERNEL_ARGUMENT);
            if (!slot.argument_allocation)
                return fail(WOOF_METAL_OUT_OF_MEMORY,
                            "could not record a Metal batch argument allocation");
            slot.arguments = argument_buffer;
            [slot.pipeline->arguments setArgumentBuffer:argument_buffer offset:0];
            for (uint32_t j = 0; j < slot.resource_count; ++j) {
                const uint32_t resource_index = slot.resource_start + j;
                [slot.pipeline->arguments setBuffer:buffers[resource_index]->resource
                    offset:static_cast<NSUInteger>(offsets[resource_index])
                    atIndex:j];
            }
        }

        id<MTLCommandBuffer> command = [context->queue commandBuffer];
        if (!command)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "could not allocate a Metal kernel batch command buffer");
        id<MTLComputeCommandEncoder> compute =
            [command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if (!compute)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "could not allocate a Metal kernel batch encoder");
        std::unordered_map<void *, MTLResourceUsage> usages;
        for (uint32_t i = 0; i < resource_count; ++i) {
            void *key = (__bridge void *)buffers[i]->resource;
            auto found = usages.find(key);
            const auto access = static_cast<MTLResourceUsage>(resource_specs[i].access);
            if (found == usages.end())
                usages.emplace(key, access);
            else
                found->second = static_cast<MTLResourceUsage>(found->second | access);
        }
        for (const auto &usage : usages) {
            id<MTLBuffer> resource = (__bridge id<MTLBuffer>)usage.first;
            [compute useResource:resource usage:usage.second];
        }
        for (uint32_t i = 0; i < item_count; ++i) {
            auto &slot = prepared[i];
            const uint64_t groups = slot.grid_size / slot.threads_per_group
                + uint64_t(slot.grid_size % slot.threads_per_group != 0);
            if (groups > std::numeric_limits<NSUInteger>::max()) {
                [compute endEncoding];
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "Metal batch dispatch grid exceeds geometry limits");
            }
            [compute setComputePipelineState:slot.pipeline->pipeline];
            [compute setBuffer:slot.arguments offset:0 atIndex:0];
            [compute setBytes:parameter_bytes + slot.parameter_offset
                       length:slot.parameter_size atIndex:1];
            [compute dispatchThreadgroups:MTLSizeMake(
                static_cast<NSUInteger>(groups), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(
                    slot.threads_per_group, 1, 1)];
        }
        [compute endEncoding];

        auto submission = std::make_shared<Submission>();
        submission->context = context;
        submission->command = command;
        submission->resources = std::move(buffers);
        submission->argument_buffers.reserve(item_count);
        submission->argument_allocations.reserve(item_count);
        for (const auto &slot : prepared) {
            submission->argument_buffers.push_back(slot.arguments);
            submission->argument_allocations.push_back(slot.argument_allocation);
        }
        prepare_submission_allocations(submission, submission->resources,
                                       submission->argument_allocations);
        {
            std::lock_guard<std::mutex> lock(g_registry_mutex);
            submission->handle = next_handle_locked();
            if (!submission->handle)
                return fail(WOOF_METAL_RUNTIME_ERROR,
                            "Metal handle space exhausted");
            g_submissions.emplace(submission->handle, submission);
        }
        begin_submission_pending(submission);
        retire_submission_after_completion(submission, command);
        [command commit];
        *out_submission = submission->handle;
        g_last_error.clear();
        return WOOF_METAL_OK;
    }
}

extern "C" int32_t woof_metal_submit_fft2(
        woof_metal_handle context_handle, woof_metal_handle input_handle,
        uint64_t input_offset, const uint64_t *input_shape,
        woof_metal_handle output_handle, uint64_t output_offset,
        const uint64_t *output_shape, uint32_t rank, int32_t inverse,
        woof_metal_handle *out_submission) {
    @autoreleasepool {
        auto context = context_for(context_handle);
        if (!context)
            return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal context handle");
        if (!input_shape || !output_shape || !out_submission ||
            (inverse != 0 && inverse != 1) || rank < 2 || rank > 16)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "invalid FFT shape, direction, or output handle");
        if (input_offset != 0 || output_offset != 0)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "MPSGraph FFT currently requires base arrays with zero byte offsets");
        for (uint32_t i = 0; i + 2 < rank; ++i) {
            if (input_shape[i] != output_shape[i])
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "FFT input/output batch dimensions do not match");
        }
        if (input_shape[rank - 2] != output_shape[rank - 2])
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "FFT input/output row dimensions do not match");
        if (inverse) {
            if (input_shape[rank - 1] != output_shape[rank - 1] / 2 + 1)
                return fail(WOOF_METAL_INVALID_ARGUMENT,
                            "irfft2 requires a half-spectrum with nx//2+1 bins");
        } else if (output_shape[rank - 1] != input_shape[rank - 1] / 2 + 1) {
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "rfft2 output requires nx//2+1 bins");
        }

        uint64_t input_bytes = 0, output_bytes = 0;
        const uint64_t input_element_size = inverse ? 8 : 4;
        const uint64_t output_element_size = inverse ? 4 : 8;
        if (!tensor_byte_count(input_shape, rank, input_element_size, &input_bytes) ||
            !tensor_byte_count(output_shape, rank, output_element_size, &output_bytes))
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "FFT dimensions are empty, oversized, or overflow storage size");
        auto input_buffer = buffer_for(input_handle);
        auto output_buffer = buffer_for(output_handle);
        if (!input_buffer || !output_buffer || input_buffer->context != context ||
            output_buffer->context != context)
            return fail(WOOF_METAL_INVALID_HANDLE,
                        "FFT buffer is invalid or belongs to another device");
        if (input_bytes > input_buffer->resource.length ||
            output_bytes > output_buffer->resource.length)
            return fail(WOOF_METAL_INVALID_ARGUMENT,
                        "FFT tensor shape exceeds its Metal buffer allocation");

        std::lock_guard<std::mutex> encode_lock(context->encode_mutex);
        auto plan = fft_plan_for(context, input_shape, output_shape, rank,
                                 inverse != 0);
        NSMutableArray<NSNumber *> *input_dims =
            [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber *> *output_dims =
            [NSMutableArray arrayWithCapacity:rank];
        for (uint32_t i = 0; i < rank; ++i) {
            [input_dims addObject:@(input_shape[i])];
            [output_dims addObject:@(output_shape[i])];
        }
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input_buffer->resource
            shape:input_dims
            dataType:inverse ? MPSDataTypeComplexFloat32 : MPSDataTypeFloat32];
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:output_buffer->resource
            shape:output_dims
            dataType:inverse ? MPSDataTypeFloat32 : MPSDataTypeComplexFloat32];
        if (!input_data || !output_data)
            return fail(WOOF_METAL_RUNTIME_ERROR,
                        "MPSGraph could not wrap the FFT Metal buffers");
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            [NSMutableDictionary dictionaryWithObject:input_data
                                               forKey:plan->input];
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
            [NSMutableDictionary dictionaryWithObject:output_data
                                               forKey:plan->output];
        MPSGraphExecutionDescriptor *execution =
            [MPSGraphExecutionDescriptor new];
        execution.waitUntilCompleted = NO;
        auto submission = std::make_shared<Submission>();
        submission->context = context;
        submission->callback_driven = true;
        submission->resources = {input_buffer, output_buffer};
        prepare_submission_allocations(submission, submission->resources);
        submission->fft_plan = plan;
        submission->graph_resources = [NSMutableArray arrayWithObjects:
            plan->graph, plan->input, plan->output, input_data, output_data,
            feeds, results, execution, nil];
        std::weak_ptr<Submission> weak_submission = submission;
        execution.completionHandler = ^(MPSGraphTensorDataDictionary *result_data,
                                        NSError *error) {
            complete_graph_submission(weak_submission, error);
        };
        {
            std::lock_guard<std::mutex> lock(g_registry_mutex);
            submission->handle = next_handle_locked();
            if (!submission->handle)
                return fail(WOOF_METAL_RUNTIME_ERROR,
                            "Metal handle space exhausted");
            g_submissions.emplace(submission->handle, submission);
        }
        begin_submission_pending(submission);
        {
            std::lock_guard<std::mutex> lock(context->allocation_mutex);
            ++context->fft_graph_executions_started;
            ++context->fft_graph_pending;
        }
        *out_submission = submission->handle;
        [plan->graph runAsyncWithMTLCommandQueue:context->queue
            feeds:feeds targetOperations:nil resultsDictionary:results
            executionDescriptor:execution];
        g_last_error.clear();
        return WOOF_METAL_OK;
    }
}

extern "C" int32_t woof_metal_submission_poll(
        woof_metal_handle handle, int32_t *complete) {
    auto submission = submission_for(handle);
    if (!submission)
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal submission handle");
    if (!complete)
        return fail(WOOF_METAL_INVALID_ARGUMENT, "completion output is required");
    bool completed_already = false;
    bool callback_driven = false;
    id<MTLCommandBuffer> command = nil;
    {
        std::lock_guard<std::mutex> lock(submission->mutex);
        completed_already = submission->completed;
        callback_driven = submission->callback_driven;
        command = submission->command;
    }
    if (completed_already) {
        *complete = 1;
        return ensure_completed_submission(submission);
    }
    if (callback_driven) {
        *complete = 0;
        g_last_error.clear();
        return WOOF_METAL_OK;
    }
    if (!command)
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    "Metal submission lost its command before completion");
    if (command.status < MTLCommandBufferStatusCompleted) {
        *complete = 0;
        g_last_error.clear();
        return WOOF_METAL_OK;
    }
    *complete = 1;
    int32_t result = finish_submission(submission, false);
    return result;
}

extern "C" int32_t woof_metal_submission_wait(woof_metal_handle handle) {
    auto submission = submission_for(handle);
    if (!submission)
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal submission handle");
    return finish_submission(submission, true);
}

extern "C" int32_t woof_metal_submission_destroy(woof_metal_handle handle) {
    auto submission = submission_for(handle);
    if (!submission)
        return fail(WOOF_METAL_INVALID_HANDLE, "invalid Metal submission handle");
    bool completed;
    {
        std::lock_guard<std::mutex> lock(submission->mutex);
        completed = submission->completed;
    }
    if (!completed)
        return fail(WOOF_METAL_RUNTIME_ERROR,
                    "cannot release an in-flight Metal submission");
    forget_submission(handle);
    g_last_error.clear();
    return WOOF_METAL_OK;
}

extern "C" const char *woof_metal_last_error(void) {
    return g_last_error.c_str();
}

extern "C" uint32_t woof_metal_abi_version(void) {
    return WOOF_METAL_ABI_VERSION;
}
