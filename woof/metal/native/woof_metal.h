#ifndef WOOF_METAL_NATIVE_H
#define WOOF_METAL_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef uint64_t woof_metal_handle;

#define WOOF_METAL_ABI_VERSION ((1u << 16) | 8u)

enum {
    WOOF_METAL_ALLOCATION_BACKING = 0,
    WOOF_METAL_ALLOCATION_RK_ARGUMENT = 1,
    WOOF_METAL_ALLOCATION_FILL_ARGUMENT = 2,
    WOOF_METAL_ALLOCATION_KERNEL_ARGUMENT = 3,
    WOOF_METAL_ALLOCATION_CATEGORY_COUNT = 4
};

enum {
    WOOF_METAL_MEMORY_OBS_PROCESS_FOOTPRINT = 1u,
    WOOF_METAL_MEMORY_OBS_SYSTEM_AVAILABLE = 2u,
    WOOF_METAL_MEMORY_OBS_DEVICE_ALLOCATION = 4u
};

typedef struct woof_metal_allocation_snapshot {
    uint32_t struct_size;
    uint32_t category_count;
    uint64_t live_allocated_bytes;
    uint64_t live_requested_bytes;
    uint64_t live_high_water_allocated_bytes;
    uint64_t pending_allocated_bytes;
    uint64_t live_allocated_by_category[WOOF_METAL_ALLOCATION_CATEGORY_COUNT];
    uint64_t live_requested_by_category[WOOF_METAL_ALLOCATION_CATEGORY_COUNT];
    uint64_t device_current_allocated_bytes;
    uint64_t fft_plan_cache_entries;
    uint64_t fft_plan_cache_builds;
    uint64_t fft_plan_cache_hits;
    uint64_t fft_plan_cache_evictions;
    uint64_t fft_graph_executions_started;
    uint64_t fft_graph_executions_completed;
    uint64_t fft_graph_executions_failed;
    uint64_t fft_graph_pending;
    uint64_t process_phys_footprint_bytes;
    uint64_t system_available_estimate_bytes;
    uint32_t memory_observation_validity;
    uint32_t reserved;
} woof_metal_allocation_snapshot;

typedef struct woof_metal_calc_coefs_params {
    uint32_t struct_size;
    uint32_t nz;
    uint32_t ny;
    uint32_t nx;
    int32_t moist_cq;
    int32_t top_lid;
    int32_t wrf_exact;
    float dtau;
    float epssm;
    float gravity;
    float gamma;
} woof_metal_calc_coefs_params;

typedef struct woof_metal_kernel_resource {
    woof_metal_handle buffer;
    uint64_t byte_offset;
    uint64_t byte_length;
    uint32_t access; /* 1 = read, 3 = read/write */
    uint32_t reserved;
} woof_metal_kernel_resource;

/* Flat batch descriptor. Resource and parameter ranges index the flat arrays
 * passed to woof_metal_submit_kernel_batch. */
typedef struct woof_metal_kernel_batch_item {
    const char *entrypoint;
    const char *cache_identity;
    int32_t exact_policy;
    uint32_t resource_start;
    uint32_t resource_count;
    uint32_t parameter_offset;
    uint32_t parameter_size;
    uint64_t grid_size;
    uint32_t threads_per_group;
    uint32_t reserved;
} woof_metal_kernel_batch_item;

enum {
    WOOF_METAL_OK = 0,
    WOOF_METAL_INVALID_ARGUMENT = 1,
    WOOF_METAL_INVALID_HANDLE = 2,
    WOOF_METAL_OUT_OF_MEMORY = 3,
    WOOF_METAL_RUNTIME_ERROR = 4,
    WOOF_METAL_SUBMISSION_FAILED = 5
};

int32_t woof_metal_context_create(const char *metallib_path,
                                 const char *exact_metallib_path,
                                 woof_metal_handle *out_context);
int32_t woof_metal_context_destroy(woof_metal_handle context);
int32_t woof_metal_context_device_info(woof_metal_handle context,
                                       char *name, uint64_t name_capacity,
                                       uint64_t *registry_id,
                                       uint64_t *max_buffer_length,
                                       uint64_t *recommended_working_set);
int32_t woof_metal_context_allocation_snapshot(
    woof_metal_handle context, woof_metal_allocation_snapshot *snapshot);

int32_t woof_metal_buffer_create(woof_metal_handle context, uint64_t size,
                                 woof_metal_handle *out_buffer);
int32_t woof_metal_buffer_destroy(woof_metal_handle buffer);
int32_t woof_metal_buffer_write(woof_metal_handle buffer, uint64_t offset,
                                const void *source, uint64_t size);
int32_t woof_metal_buffer_read(woof_metal_handle buffer, uint64_t offset,
                               void *destination, uint64_t size);
int32_t woof_metal_buffer_size(woof_metal_handle buffer, uint64_t *size);

int32_t woof_metal_submit_rk_bookkeeping(
    woof_metal_handle context, int32_t zero, uint32_t pair_count,
    const woof_metal_handle *source_buffers, const uint64_t *source_offsets,
    const woof_metal_handle *target_buffers, const uint64_t *target_offsets,
    const uint64_t *word_counts, woof_metal_handle *out_submission);
int32_t woof_metal_submit_fill(woof_metal_handle context,
                               woof_metal_handle target_buffer,
                               uint64_t target_offset, uint64_t word_count,
                               const uint32_t *word_pattern,
                               uint32_t pattern_word_count,
                               woof_metal_handle *out_submission);
int32_t woof_metal_submit_acoustic_calc_coefs(
    woof_metal_handle context, const woof_metal_handle *buffers,
    const uint64_t *byte_offsets,
    const woof_metal_calc_coefs_params *parameters,
    woof_metal_handle *out_submission);
int32_t woof_metal_submit_kernel(
    woof_metal_handle context, const char *entrypoint, const char *cache_identity,
    int32_t exact_policy, const woof_metal_kernel_resource *resources,
    uint32_t resource_count, const void *parameters, uint32_t parameter_size,
    uint64_t grid_size, uint32_t threads_per_group,
    woof_metal_handle *out_submission);
int32_t woof_metal_submit_kernel_batch(
    woof_metal_handle context, const woof_metal_kernel_batch_item *items,
    uint32_t item_count, const woof_metal_kernel_resource *resources,
    uint32_t resource_count, const uint8_t *parameter_bytes,
    uint32_t parameter_bytes_size, woof_metal_handle *out_submission);
int32_t woof_metal_submit_fft2(
    woof_metal_handle context, woof_metal_handle input_buffer,
    uint64_t input_offset, const uint64_t *input_shape,
    woof_metal_handle output_buffer, uint64_t output_offset,
    const uint64_t *output_shape, uint32_t rank, int32_t inverse,
    woof_metal_handle *out_submission);
int32_t woof_metal_submission_poll(woof_metal_handle submission,
                                   int32_t *complete);
int32_t woof_metal_submission_wait(woof_metal_handle submission);
int32_t woof_metal_submission_destroy(woof_metal_handle submission);
int32_t woof_metal_warm_pipelines(woof_metal_handle context,
                                  uint32_t *out_count, char *failed_names,
                                  uint32_t failed_names_size);

const char *woof_metal_last_error(void);
uint32_t woof_metal_abi_version(void);

#if defined(__cplusplus)
}
#endif

#endif
