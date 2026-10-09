kernel void bound_tke(constant BoundTkeResources &r [[buffer(0)]],
                      constant BoundTkeParams &p [[buffer(1)]],
                      uint flat [[thread_position_in_grid]]) {
    if (flat >= uint(p.n)) return;
    r.tke[flat]=min(p.upper,max(r.tke[flat],0.0f));
}
