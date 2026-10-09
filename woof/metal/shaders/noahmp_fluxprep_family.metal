// No entry points are emitted for this packet yet.
//
// RAGRB and STOMATA depend on the pinned glibc r_pow/r_exp transcriptions;
// SFCDIF1 also uses r_log and r_atan. These routines and the source-ordered
// RN FP32 operations need exact binary64/gradual-underflow behavior. No
// reviewed Metal arithmetic layer is available. See
// woof/metal/handoffs/55_noahmp_fluxprep.md.
