#include <metal_stdlib>

using namespace metal;

// Resource indices are encoded by MTLArgumentEncoder. Each source and target
// binding remains an MTLBuffer plus byte offset; no host pointer is serialized
// into the shader ABI. Counts are element counts, matching rk_bookkeeping.cu.
struct RKWordArguments {
    array<device const uint *, 64> sources [[id(0)]];
    array<device uint *, 64> targets [[id(64)]];
    array<ulong, 64> word_counts [[id(128)]];
};

constant uint rk_lanes_per_group = 128;
constant uint rk_words_per_lane = 4;

kernel void rk_copy_words(constant RKWordArguments &arguments [[buffer(0)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]]) {
    const uint row = group.y;
    const ulong count = arguments.word_counts[row];
    const ulong first = ulong(group.x) * rk_lanes_per_group * rk_words_per_lane;
    if (first >= count) return;

    const device uint *source = arguments.sources[row];
    device uint *target = arguments.targets[row];
    const ulong lane_first = first + lane;
    for (uint word = 0; word < rk_words_per_lane; ++word) {
        const ulong index = lane_first + ulong(word) * rk_lanes_per_group;
        if (index < count) target[index] = source[index];
    }
}

kernel void rk_zero_words(constant RKWordArguments &arguments [[buffer(0)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]]) {
    const uint row = group.y;
    const ulong count = arguments.word_counts[row];
    const ulong first = ulong(group.x) * rk_lanes_per_group * rk_words_per_lane;
    if (first >= count) return;

    device uint *target = arguments.targets[row];
    const ulong lane_first = first + lane;
    for (uint word = 0; word < rk_words_per_lane; ++word) {
        const ulong index = lane_first + ulong(word) * rk_lanes_per_group;
        if (index < count) target[index] = 0u;
    }
}

kernel void rk_fill_words(constant RKWordArguments &arguments [[buffer(0)]],
                          constant uint *word_pattern [[buffer(1)]],
                          constant uint &pattern_word_count [[buffer(2)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]]) {
    const uint row = group.y;
    const ulong count = arguments.word_counts[row];
    const ulong first = ulong(group.x) * rk_lanes_per_group * rk_words_per_lane;
    if (first >= count) return;

    device uint *target = arguments.targets[row];
    const ulong lane_first = first + lane;
    for (uint word = 0; word < rk_words_per_lane; ++word) {
        const ulong index = lane_first + ulong(word) * rk_lanes_per_group;
        if (index < count)
            target[index] = word_pattern[index % pattern_word_count];
    }
}
