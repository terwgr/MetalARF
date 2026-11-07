// The 128-level GFDRV kernel rebuilt at a 256-level column stride for
// 128 < nz <= 256; the source is gf_family.metal unchanged.
#define GF_KMAX 256
#define gf_gfdrv_stage gf_gfdrv_stage_k256
#include "gf_family.metal"
