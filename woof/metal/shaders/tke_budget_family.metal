// Deliberately contains no kernel entry point.
//
// woof/core/kernels/tke_budget.cu requires FP64 values, FP64 thread-local and
// shared reduction state, and an FP64 output buffer. The current typed Metal
// ABI cannot express that resource and MetalArray documents eight-byte values
// as bit storage only. Adding an FP32 entry point here would silently change
// the budget's defining numerical behavior. See the packet handoff for the
// prerequisite and proposed integration.
