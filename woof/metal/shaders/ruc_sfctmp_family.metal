#include <metal_stdlib>
#include "ruc_sfctmp_family_bindings.metalh"
#include "soft_binary32.metalh"

using namespace metal;
using RucSfctmpResources = RucSfctmpStage0Resources;
using RucSfctmpParams = RucSfctmpStage0Params;
#pragma clang fp contract(off)

static inline float ruc_fadd_rn(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ruc_fsub_rn(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ruc_fmul_rn(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ruc_fdiv_rn(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ruc_fused_min(float a, float b) { return (isnan(a) || isnan(b)) ? as_type<float>(0x7fc00000u) : min(a,b); }
static inline float ruc_fused_max(float a, float b) { return (isnan(a) || isnan(b)) ? as_type<float>(0x7fc00000u) : max(a,b); }

#define RUC_NZS 9
#define RUC_SFCTMP_FLAG_WORDS 15
#define RUC_SFCTMP_NZS 9
#define RUC_RUN (reinterpret_cast<device uint *>(r.raw + p.run_base))
#define RUC_ALIVE (reinterpret_cast<device uint *>(r.raw + p.alive_base))

#include "ruc_sfctmp_helpers.metalh"
#define a0 (reinterpret_cast<device float *>(r.raw + (p.table_base + p.table_offset_rstbl)))
#define a1 (reinterpret_cast<device float *>(r.raw + (p.table_base + p.table_offset_rgltbl)))
#define a2 (reinterpret_cast<device float *>(r.raw + (p.table_base + p.table_offset_z0tbl)))
#define a3 (reinterpret_cast<device float *>(r.raw + (p.table_base + p.table_offset_lemitbl)))
#define a4 (reinterpret_cast<device float *>(r.raw + (p.table_base + p.table_offset_tbq)))
#define a5 (reinterpret_cast<device float *>(r.raw + (p.table_base + p.table_offset_zshalf)))
#define a6 (reinterpret_cast<device float *>(r.raw + (168 * p.n)))
#define a7 (reinterpret_cast<device float *>(r.raw + (177 * p.n)))
#define a8 (reinterpret_cast<device float *>(r.raw + (186 * p.n)))
#define a9 (reinterpret_cast<device float *>(r.raw + (195 * p.n)))
#define a10 (reinterpret_cast<device float *>(r.raw + (213 * p.n)))
#define a11 (reinterpret_cast<device float *>(r.raw + (204 * p.n)))
#define a12 (reinterpret_cast<device float *>(r.raw + (222 * p.n)))
#define a13 (reinterpret_cast<device float *>(r.raw + (45 * p.n)))
#define a14 (reinterpret_cast<device float *>(r.raw + (38 * p.n)))
#define a15 (reinterpret_cast<device float *>(r.raw + (30 * p.n)))
#define a16 (reinterpret_cast<device float *>(r.raw + (80 * p.n)))
#define a17 (reinterpret_cast<device float *>(r.raw + (81 * p.n)))
#define a18 (reinterpret_cast<device float *>(r.raw + (55 * p.n)))
#define a19 (reinterpret_cast<device float *>(r.raw + (7 * p.n)))
#define a20 (reinterpret_cast<device float *>(r.raw + (107 * p.n)))
#define a21 (reinterpret_cast<device float *>(r.raw + (12 * p.n)))
#define a22 (reinterpret_cast<device float *>(r.raw + (35 * p.n)))
#define a23 (reinterpret_cast<device float *>(r.raw + (4 * p.n)))
#define a24 (reinterpret_cast<device float *>(r.raw + (49 * p.n)))
#define a25 (reinterpret_cast<device float *>(r.raw + (82 * p.n)))
#define a26 (reinterpret_cast<device float *>(r.raw + (83 * p.n)))
#define a27 (reinterpret_cast<device float *>(r.raw + (84 * p.n)))
#define a28 (reinterpret_cast<device float *>(r.raw + (85 * p.n)))
#define a29 (reinterpret_cast<device float *>(r.raw + (88 * p.n)))
#define a30 (reinterpret_cast<device float *>(r.raw + (89 * p.n)))
#define a31 (reinterpret_cast<device float *>(r.raw + (91 * p.n)))
#define a32 (reinterpret_cast<device float *>(r.raw + (93 * p.n)))
#define a33 (reinterpret_cast<device float *>(r.raw + (92 * p.n)))
#define a34 (reinterpret_cast<device float *>(r.raw + (90 * p.n)))
#define a35 (reinterpret_cast<device float *>(r.raw + (5 * p.n)))
#define a36 (reinterpret_cast<device float *>(r.raw + (94 * p.n)))
#define a37 (reinterpret_cast<device float *>(r.raw + (11 * p.n)))
#define a38 (reinterpret_cast<device float *>(r.raw + (105 * p.n)))
#define a39 (reinterpret_cast<device float *>(r.raw + (78 * p.n)))
#define a40 (reinterpret_cast<device float *>(r.raw + (39 * p.n)))
#define a41 (reinterpret_cast<device float *>(r.raw + (40 * p.n)))
#define a42 (reinterpret_cast<device float *>(r.raw + (41 * p.n)))
#define a43 (reinterpret_cast<device float *>(r.raw + (44 * p.n)))
#define a44 (reinterpret_cast<device float *>(r.raw + (86 * p.n)))
#define a45 (reinterpret_cast<device float *>(r.raw + (87 * p.n)))
#define a46 (reinterpret_cast<device float *>(r.raw + (95 * p.n)))
#define a47 (reinterpret_cast<device float *>(r.raw + (106 * p.n)))
#define a48 (reinterpret_cast<device float *>(r.raw + (96 * p.n)))
#define a49 (reinterpret_cast<device float *>(r.raw + (97 * p.n)))
#define a50 (reinterpret_cast<device float *>(r.raw + (99 * p.n)))
#define a51 (reinterpret_cast<device float *>(r.raw + (102 * p.n)))
#define a52 (reinterpret_cast<device float *>(r.raw + (103 * p.n)))
#define a53 (reinterpret_cast<device float *>(r.raw + (104 * p.n)))
#define a54 (reinterpret_cast<device float *>(r.raw + (101 * p.n)))
#define a55 (reinterpret_cast<device float *>(r.raw + (98 * p.n)))
#define a56 (reinterpret_cast<device float *>(r.raw + (100 * p.n)))
#define a57 (reinterpret_cast<device float *>(r.raw + (108 * p.n)))
#define a58 (reinterpret_cast<device float *>(r.raw + (29 * p.n)))
#define a59 (reinterpret_cast<device float *>(r.raw + (23 * p.n)))
#define a60 (reinterpret_cast<device float *>(r.raw + (27 * p.n)))
#define a61 (reinterpret_cast<device float *>(r.raw + (24 * p.n)))
#define a62 (reinterpret_cast<device float *>(r.raw + (109 * p.n)))
#define a63 (reinterpret_cast<device float *>(r.raw + (110 * p.n)))
#define a64 (reinterpret_cast<device float *>(r.raw + (22 * p.n)))
#define a65 (reinterpret_cast<device float *>(r.raw + (21 * p.n)))
#define a66 (reinterpret_cast<device float *>(r.raw + (111 * p.n)))
#define a67 (reinterpret_cast<device float *>(r.raw + (112 * p.n)))
#define a68 (reinterpret_cast<device float *>(r.raw + (113 * p.n)))
#define a69 (reinterpret_cast<device float *>(r.raw + (79 * p.n)))
#define a70 (reinterpret_cast<device int *>(r.raw + (p.int_base + 0 * p.n)))
#define a71 (reinterpret_cast<device int *>(r.raw + (p.int_base + 2 * p.n)))
#define a72 (reinterpret_cast<device int *>(r.raw + (p.int_base + 3 * p.n)))
#define a73 (reinterpret_cast<device int *>(r.raw + (p.int_base + 4 * p.n)))
#define a74 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a75 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a76 (reinterpret_cast<device int *>(r.raw + (232 * p.n)))
#define a77 (reinterpret_cast<device int *>(r.raw + (233 * p.n)))
#define a78 (reinterpret_cast<device int *>(r.raw + (233 * p.n)))
#define a79 (reinterpret_cast<device int *>(r.raw + (232 * p.n)))
#define a80 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a81 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a82 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a83 (reinterpret_cast<device float *>(r.raw + (234 * p.n)))
#define a84 (reinterpret_cast<device float *>(r.raw + (243 * p.n)))
#define a85 (reinterpret_cast<device float *>(r.raw + (252 * p.n)))
#define a86 (reinterpret_cast<device float *>(r.raw + (261 * p.n)))
#define a87 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a88 (reinterpret_cast<device float *>(r.raw + (271 * p.n)))
#define a89 (reinterpret_cast<device float *>(r.raw + (272 * p.n)))
#define a90 (reinterpret_cast<device float *>(r.raw + (273 * p.n)))
#define a91 (reinterpret_cast<device float *>(r.raw + (274 * p.n)))
#define a92 (reinterpret_cast<device float *>(r.raw + (275 * p.n)))
#define a93 (reinterpret_cast<device float *>(r.raw + (276 * p.n)))
#define a94 (reinterpret_cast<device float *>(r.raw + (277 * p.n)))
#define a95 (reinterpret_cast<device float *>(r.raw + (278 * p.n)))
#define a96 (reinterpret_cast<device float *>(r.raw + (279 * p.n)))
#define a97 (reinterpret_cast<device float *>(r.raw + (280 * p.n)))
#define a98 (reinterpret_cast<device float *>(r.raw + (281 * p.n)))
#define a99 (reinterpret_cast<device float *>(r.raw + (282 * p.n)))
#define a100 (reinterpret_cast<device float *>(r.raw + (283 * p.n)))
#define a101 (reinterpret_cast<device float *>(r.raw + (284 * p.n)))
#define a102 (reinterpret_cast<device float *>(r.raw + (285 * p.n)))
#define a103 (reinterpret_cast<device float *>(r.raw + (286 * p.n)))
#define a104 (reinterpret_cast<device float *>(r.raw + (287 * p.n)))
#define a105 (reinterpret_cast<device float *>(r.raw + (288 * p.n)))
#define a106 (reinterpret_cast<device float *>(r.raw + (289 * p.n)))
#define a107 (reinterpret_cast<device float *>(r.raw + (290 * p.n)))
#define a108 (reinterpret_cast<device float *>(r.raw + (291 * p.n)))
#define a109 (reinterpret_cast<device float *>(r.raw + (292 * p.n)))
#define a110 (reinterpret_cast<device float *>(r.raw + (293 * p.n)))
#define a111 (reinterpret_cast<device float *>(r.raw + (294 * p.n)))
#define a112 (reinterpret_cast<device float *>(r.raw + (295 * p.n)))
#define a113 (reinterpret_cast<device float *>(r.raw + (296 * p.n)))
#define a114 (reinterpret_cast<device float *>(r.raw + (297 * p.n)))
#define a115 (reinterpret_cast<device float *>(r.raw + (298 * p.n)))
#define a116 (reinterpret_cast<device float *>(r.raw + (299 * p.n)))
#define a117 (reinterpret_cast<device float *>(r.raw + (300 * p.n)))
#define a118 (reinterpret_cast<device float *>(r.raw + (301 * p.n)))
#define a119 (reinterpret_cast<device float *>(r.raw + (302 * p.n)))
#define a120 (reinterpret_cast<device float *>(r.raw + (303 * p.n)))
#define a121 (reinterpret_cast<device float *>(r.raw + (304 * p.n)))
#define a122 (reinterpret_cast<device float *>(r.raw + (305 * p.n)))
#define a123 (reinterpret_cast<device float *>(r.raw + (306 * p.n)))
#define a124 (reinterpret_cast<device float *>(r.raw + (307 * p.n)))
#define a125 (reinterpret_cast<device float *>(r.raw + (308 * p.n)))
#define a126 (reinterpret_cast<device float *>(r.raw + (309 * p.n)))
#define a127 (reinterpret_cast<device int *>(r.raw + (310 * p.n)))
#define a128 (reinterpret_cast<device float *>(r.raw + (243 * p.n)))
#define a129 (reinterpret_cast<device float *>(r.raw + (234 * p.n)))
#define a130 (reinterpret_cast<device float *>(r.raw + (311 * p.n)))
#define a131 (reinterpret_cast<device float *>(r.raw + (320 * p.n)))
#define a132 (reinterpret_cast<device float *>(r.raw + (329 * p.n)))
#define a133 (reinterpret_cast<device float *>(r.raw + (338 * p.n)))
#define a134 (reinterpret_cast<device float *>(r.raw + (300 * p.n)))
#define a135 (reinterpret_cast<device float *>(r.raw + (301 * p.n)))
#define a136 (reinterpret_cast<device float *>(r.raw + (302 * p.n)))
#define a137 (reinterpret_cast<device float *>(r.raw + (303 * p.n)))
#define a138 (reinterpret_cast<device float *>(r.raw + (304 * p.n)))
#define a139 (reinterpret_cast<device float *>(r.raw + (278 * p.n)))
#define a140 (reinterpret_cast<device float *>(r.raw + (305 * p.n)))
#define a141 (reinterpret_cast<device float *>(r.raw + (296 * p.n)))
#define a142 (reinterpret_cast<device float *>(r.raw + (293 * p.n)))
#define a143 (reinterpret_cast<device float *>(r.raw + (292 * p.n)))
#define a144 (reinterpret_cast<device float *>(r.raw + (290 * p.n)))
#define a145 (reinterpret_cast<device float *>(r.raw + (289 * p.n)))
#define a146 (reinterpret_cast<device float *>(r.raw + (308 * p.n)))
#define a147 (reinterpret_cast<device float *>(r.raw + (286 * p.n)))
#define a148 (reinterpret_cast<device float *>(r.raw + (307 * p.n)))
#define a149 (reinterpret_cast<device float *>(r.raw + (306 * p.n)))
#define a150 (reinterpret_cast<device float *>(r.raw + (309 * p.n)))
#define a151 (reinterpret_cast<device float *>(r.raw + (282 * p.n)))
#define a152 (reinterpret_cast<device float *>(r.raw + (277 * p.n)))
#define a153 (reinterpret_cast<device float *>(r.raw + (274 * p.n)))
#define a154 (reinterpret_cast<device float *>(r.raw + (272 * p.n)))
#define a155 (reinterpret_cast<device float *>(r.raw + (271 * p.n)))
#define a156 (reinterpret_cast<device float *>(r.raw + (347 * p.n)))
#define a157 (reinterpret_cast<device float *>(r.raw + (348 * p.n)))
#define a158 (reinterpret_cast<device float *>(r.raw + (279 * p.n)))
#define a159 (reinterpret_cast<device float *>(r.raw + (349 * p.n)))
#define a160 (reinterpret_cast<device float *>(r.raw + (350 * p.n)))
#define a161 (reinterpret_cast<device float *>(r.raw + (351 * p.n)))
#define a162 (reinterpret_cast<device float *>(r.raw + (352 * p.n)))
#define a163 (reinterpret_cast<device float *>(r.raw + (353 * p.n)))
#define a164 (reinterpret_cast<device float *>(r.raw + (354 * p.n)))
#define a165 (reinterpret_cast<device float *>(r.raw + (355 * p.n)))
#define a166 (reinterpret_cast<device float *>(r.raw + (356 * p.n)))
#define a167 (reinterpret_cast<device float *>(r.raw + (357 * p.n)))
#define a168 (reinterpret_cast<device float *>(r.raw + (358 * p.n)))
#define a169 (reinterpret_cast<device float *>(r.raw + (359 * p.n)))
#define a170 (reinterpret_cast<device float *>(r.raw + (360 * p.n)))
#define a171 (reinterpret_cast<device float *>(r.raw + (361 * p.n)))
#define a172 (reinterpret_cast<device float *>(r.raw + (362 * p.n)))
#define a173 (reinterpret_cast<device float *>(r.raw + (363 * p.n)))
#define a174 (reinterpret_cast<device float *>(r.raw + (364 * p.n)))
#define a175 (reinterpret_cast<device float *>(r.raw + (365 * p.n)))
#define a176 (reinterpret_cast<device float *>(r.raw + (366 * p.n)))
#define a177 (reinterpret_cast<device float *>(r.raw + (283 * p.n)))
#define a178 (reinterpret_cast<device float *>(r.raw + (288 * p.n)))
#define a179 (reinterpret_cast<device float *>(r.raw + (281 * p.n)))
#define a180 (reinterpret_cast<device float *>(r.raw + (367 * p.n)))
#define a181 (reinterpret_cast<device int *>(r.raw + (368 * p.n)))
#define a182 (reinterpret_cast<device int *>(r.raw + (310 * p.n)))
#define a183 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a184 (reinterpret_cast<device int *>(r.raw + (232 * p.n)))
#define a185 (reinterpret_cast<device int *>(r.raw + (233 * p.n)))
#define a186 (reinterpret_cast<device float *>(r.raw + (369 * p.n)))
#define a187 (reinterpret_cast<device float *>(r.raw + (378 * p.n)))
#define a188 (reinterpret_cast<device float *>(r.raw + (387 * p.n)))
#define a189 (reinterpret_cast<device float *>(r.raw + (396 * p.n)))
#define a190 (reinterpret_cast<device float *>(r.raw + (405 * p.n)))
#define a191 (reinterpret_cast<device float *>(r.raw + (414 * p.n)))
#define a192 (reinterpret_cast<device float *>(r.raw + (273 * p.n)))
#define a193 (reinterpret_cast<device float *>(r.raw + (423 * p.n)))
#define a194 (reinterpret_cast<device float *>(r.raw + (424 * p.n)))
#define a195 (reinterpret_cast<device float *>(r.raw + (425 * p.n)))
#define a196 (reinterpret_cast<device float *>(r.raw + (426 * p.n)))
#define a197 (reinterpret_cast<device float *>(r.raw + (427 * p.n)))
#define a198 (reinterpret_cast<device float *>(r.raw + (428 * p.n)))
#define a199 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a200 (reinterpret_cast<device float *>(r.raw + (430 * p.n)))
#define a201 (reinterpret_cast<device float *>(r.raw + (431 * p.n)))
#define a202 (reinterpret_cast<device float *>(r.raw + (432 * p.n)))
#define a203 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a204 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a205 (reinterpret_cast<device float *>(r.raw + (435 * p.n)))
#define a206 (reinterpret_cast<device float *>(r.raw + (436 * p.n)))
#define a207 (reinterpret_cast<device float *>(r.raw + (437 * p.n)))
#define a208 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a209 (reinterpret_cast<device float *>(r.raw + (439 * p.n)))
#define a210 (reinterpret_cast<device float *>(r.raw + (440 * p.n)))
#define a211 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a212 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a213 (reinterpret_cast<device int *>(r.raw + (443 * p.n)))
#define a214 (reinterpret_cast<device int *>(r.raw + (444 * p.n)))
#define a215 (reinterpret_cast<device int *>(r.raw + (443 * p.n)))
#define a216 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a217 (reinterpret_cast<device int *>(r.raw + (443 * p.n)))
#define a218 (reinterpret_cast<device int *>(r.raw + (445 * p.n)))
#define a219 (reinterpret_cast<device int *>(r.raw + (443 * p.n)))
#define a220 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a221 (reinterpret_cast<device int *>(r.raw + (447 * p.n)))
#define a222 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a223 (reinterpret_cast<device int *>(r.raw + (449 * p.n)))
#define a224 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a225 (reinterpret_cast<device int *>(r.raw + (447 * p.n)))
#define a226 (reinterpret_cast<device float *>(r.raw + (450 * p.n)))
#define a227 (reinterpret_cast<device float *>(r.raw + (451 * p.n)))
#define a228 (reinterpret_cast<device float *>(r.raw + (450 * p.n)))
#define a229 (reinterpret_cast<device float *>(r.raw + (452 * p.n)))
#define a230 (reinterpret_cast<device float *>(r.raw + (450 * p.n)))
#define a231 (reinterpret_cast<device float *>(r.raw + (452 * p.n)))
#define a232 (reinterpret_cast<device float *>(r.raw + (450 * p.n)))
#define a233 (reinterpret_cast<device float *>(r.raw + (452 * p.n)))
#define a234 (reinterpret_cast<device float *>(r.raw + (450 * p.n)))
#define a235 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a236 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a237 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a238 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a239 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a240 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a241 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a242 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a243 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a244 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a245 (reinterpret_cast<device float *>(r.raw + (454 * p.n)))
#define a246 (reinterpret_cast<device float *>(r.raw + (463 * p.n)))
#define a247 (reinterpret_cast<device float *>(r.raw + (472 * p.n)))
#define a248 (reinterpret_cast<device float *>(r.raw + (481 * p.n)))
#define a249 (reinterpret_cast<device float *>(r.raw + (490 * p.n)))
#define a250 (reinterpret_cast<device float *>(r.raw + (499 * p.n)))
#define a251 (reinterpret_cast<device float *>(r.raw + (508 * p.n)))
#define a252 (reinterpret_cast<device float *>(r.raw + (517 * p.n)))
#define a253 (reinterpret_cast<device float *>(r.raw + (526 * p.n)))
#define a254 (reinterpret_cast<device float *>(r.raw + (535 * p.n)))
#define a255 (reinterpret_cast<device float *>(r.raw + (544 * p.n)))
#define a256 (reinterpret_cast<device float *>(r.raw + (553 * p.n)))
#define a257 (reinterpret_cast<device float *>(r.raw + (562 * p.n)))
#define a258 (reinterpret_cast<device float *>(r.raw + (571 * p.n)))
#define a259 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a260 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a261 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a262 (reinterpret_cast<device float *>(r.raw + (580 * p.n)))
#define a263 (reinterpret_cast<device float *>(r.raw + (589 * p.n)))
#define a264 (reinterpret_cast<device float *>(r.raw + (598 * p.n)))
#define a265 (reinterpret_cast<device float *>(r.raw + (607 * p.n)))
#define a266 (reinterpret_cast<device float *>(r.raw + (452 * p.n)))
#define a267 (reinterpret_cast<device float *>(r.raw + (451 * p.n)))
#define a268 (reinterpret_cast<device float *>(r.raw + (616 * p.n)))
#define a269 (reinterpret_cast<device float *>(r.raw + (617 * p.n)))
#define a270 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a271 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a272 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a273 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a274 (reinterpret_cast<device float *>(r.raw + (571 * p.n)))
#define a275 (reinterpret_cast<device float *>(r.raw + (618 * p.n)))
#define a276 (reinterpret_cast<device float *>(r.raw + (619 * p.n)))
#define a277 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a278 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a279 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a280 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a281 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a282 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a283 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a284 (reinterpret_cast<device float *>(r.raw + (508 * p.n)))
#define a285 (reinterpret_cast<device float *>(r.raw + (619 * p.n)))
#define a286 (reinterpret_cast<device float *>(r.raw + (452 * p.n)))
#define a287 (reinterpret_cast<device float *>(r.raw + (620 * p.n)))
#define a288 (reinterpret_cast<device float *>(r.raw + (621 * p.n)))
#define a289 (reinterpret_cast<device float *>(r.raw + (622 * p.n)))
#define a290 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a291 (reinterpret_cast<device float *>(r.raw + (463 * p.n)))
#define a292 (reinterpret_cast<device float *>(r.raw + (562 * p.n)))
#define a293 (reinterpret_cast<device float *>(r.raw + (553 * p.n)))
#define a294 (reinterpret_cast<device float *>(r.raw + (544 * p.n)))
#define a295 (reinterpret_cast<device float *>(r.raw + (535 * p.n)))
#define a296 (reinterpret_cast<device float *>(r.raw + (526 * p.n)))
#define a297 (reinterpret_cast<device float *>(r.raw + (517 * p.n)))
#define a298 (reinterpret_cast<device float *>(r.raw + (623 * p.n)))
#define a299 (reinterpret_cast<device float *>(r.raw + (623 * p.n)))
#define a300 (reinterpret_cast<device float *>(r.raw + (618 * p.n)))
#define a301 (reinterpret_cast<device float *>(r.raw + (632 * p.n)))
#define a302 (reinterpret_cast<device float *>(r.raw + (633 * p.n)))
#define a303 (reinterpret_cast<device float *>(r.raw + (634 * p.n)))
#define a304 (reinterpret_cast<device float *>(r.raw + (616 * p.n)))
#define a305 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a306 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a307 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a308 (reinterpret_cast<device float *>(r.raw + (571 * p.n)))
#define a309 (reinterpret_cast<device float *>(r.raw + (517 * p.n)))
#define a310 (reinterpret_cast<device float *>(r.raw + (635 * p.n)))
#define a311 (reinterpret_cast<device float *>(r.raw + (636 * p.n)))
#define a312 (reinterpret_cast<device float *>(r.raw + (637 * p.n)))
#define a313 (reinterpret_cast<device float *>(r.raw + (638 * p.n)))
#define a314 (reinterpret_cast<device float *>(r.raw + (639 * p.n)))
#define a315 (reinterpret_cast<device float *>(r.raw + (639 * p.n)))
#define a316 (reinterpret_cast<device float *>(r.raw + (616 * p.n)))
#define a317 (reinterpret_cast<device float *>(r.raw + (633 * p.n)))
#define a318 (reinterpret_cast<device float *>(r.raw + (287 * p.n)))
#define a319 (reinterpret_cast<device float *>(r.raw + (640 * p.n)))
#define a320 (reinterpret_cast<device float *>(r.raw + (641 * p.n)))
#define a321 (reinterpret_cast<device float *>(r.raw + (642 * p.n)))
#define a322 (reinterpret_cast<device float *>(r.raw + (643 * p.n)))
#define a323 (reinterpret_cast<device float *>(r.raw + (644 * p.n)))
#define a324 (reinterpret_cast<device float *>(r.raw + (645 * p.n)))
#define a325 (reinterpret_cast<device float *>(r.raw + (646 * p.n)))
#define a326 (reinterpret_cast<device float *>(r.raw + (647 * p.n)))
#define a327 (reinterpret_cast<device float *>(r.raw + (647 * p.n)))
#define a328 (reinterpret_cast<device float *>(r.raw + (646 * p.n)))
#define a329 (reinterpret_cast<device float *>(r.raw + (647 * p.n)))
#define a330 (reinterpret_cast<device float *>(r.raw + (645 * p.n)))
#define a331 (reinterpret_cast<device float *>(r.raw + (647 * p.n)))
#define a332 (reinterpret_cast<device float *>(r.raw + (645 * p.n)))
#define a333 (reinterpret_cast<device float *>(r.raw + (647 * p.n)))
#define a334 (reinterpret_cast<device float *>(r.raw + (645 * p.n)))
#define a335 (reinterpret_cast<device float *>(r.raw + (647 * p.n)))
#define a336 (reinterpret_cast<device float *>(r.raw + (645 * p.n)))
#define a337 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a338 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a339 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a340 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a341 (reinterpret_cast<device float *>(r.raw + (517 * p.n)))
#define a342 (reinterpret_cast<device float *>(r.raw + (646 * p.n)))
#define a343 (reinterpret_cast<device float *>(r.raw + (644 * p.n)))
#define a344 (reinterpret_cast<device float *>(r.raw + (643 * p.n)))
#define a345 (reinterpret_cast<device float *>(r.raw + (642 * p.n)))
#define a346 (reinterpret_cast<device float *>(r.raw + (641 * p.n)))
#define a347 (reinterpret_cast<device float *>(r.raw + (640 * p.n)))
#define a348 (reinterpret_cast<device float *>(r.raw + (287 * p.n)))
#define a349 (reinterpret_cast<device float *>(r.raw + (633 * p.n)))
#define a350 (reinterpret_cast<device float *>(r.raw + (616 * p.n)))
#define a351 (reinterpret_cast<device float *>(r.raw + (639 * p.n)))
#define a352 (reinterpret_cast<device float *>(r.raw + (638 * p.n)))
#define a353 (reinterpret_cast<device float *>(r.raw + (637 * p.n)))
#define a354 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a355 (reinterpret_cast<device float *>(r.raw + (571 * p.n)))
#define a356 (reinterpret_cast<device float *>(r.raw + (571 * p.n)))
#define a357 (reinterpret_cast<device float *>(r.raw + (571 * p.n)))
#define a358 (reinterpret_cast<device float *>(r.raw + (571 * p.n)))
#define a359 (reinterpret_cast<device float *>(r.raw + (571 * p.n)))
#define a360 (reinterpret_cast<device float *>(r.raw + (637 * p.n)))
#define a361 (reinterpret_cast<device float *>(r.raw + (637 * p.n)))
#define a362 (reinterpret_cast<device float *>(r.raw + (638 * p.n)))
#define a363 (reinterpret_cast<device float *>(r.raw + (639 * p.n)))
#define a364 (reinterpret_cast<device float *>(r.raw + (638 * p.n)))
#define a365 (reinterpret_cast<device float *>(r.raw + (616 * p.n)))
#define a366 (reinterpret_cast<device float *>(r.raw + (638 * p.n)))
#define a367 (reinterpret_cast<device float *>(r.raw + (616 * p.n)))
#define a368 (reinterpret_cast<device float *>(r.raw + (638 * p.n)))
#define a369 (reinterpret_cast<device float *>(r.raw + (616 * p.n)))
#define a370 (reinterpret_cast<device float *>(r.raw + (638 * p.n)))
#define a371 (reinterpret_cast<device float *>(r.raw + (638 * p.n)))
#define a372 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a373 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a374 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a375 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a376 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a377 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a378 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a379 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a380 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a381 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a382 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a383 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a384 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a385 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a386 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a387 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a388 (reinterpret_cast<device float *>(r.raw + (571 * p.n)))
#define a389 (reinterpret_cast<device float *>(r.raw + (517 * p.n)))
#define a390 (reinterpret_cast<device float *>(r.raw + (562 * p.n)))
#define a391 (reinterpret_cast<device float *>(r.raw + (508 * p.n)))
#define a392 (reinterpret_cast<device float *>(r.raw + (607 * p.n)))
#define a393 (reinterpret_cast<device float *>(r.raw + (580 * p.n)))
#define a394 (reinterpret_cast<device float *>(r.raw + (499 * p.n)))
#define a395 (reinterpret_cast<device float *>(r.raw + (490 * p.n)))
#define a396 (reinterpret_cast<device float *>(r.raw + (481 * p.n)))
#define a397 (reinterpret_cast<device float *>(r.raw + (472 * p.n)))
#define a398 (reinterpret_cast<device float *>(r.raw + (623 * p.n)))
#define a399 (reinterpret_cast<device float *>(r.raw + (463 * p.n)))
#define a400 (reinterpret_cast<device float *>(r.raw + (598 * p.n)))
#define a401 (reinterpret_cast<device float *>(r.raw + (589 * p.n)))
#define a402 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a403 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a404 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a405 (reinterpret_cast<device float *>(r.raw + (454 * p.n)))
#define a406 (reinterpret_cast<device float *>(r.raw + (526 * p.n)))
#define a407 (reinterpret_cast<device float *>(r.raw + (535 * p.n)))
#define a408 (reinterpret_cast<device float *>(r.raw + (544 * p.n)))
#define a409 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a410 (reinterpret_cast<device float *>(r.raw + (616 * p.n)))
#define a411 (reinterpret_cast<device float *>(r.raw + (639 * p.n)))
#define a412 (reinterpret_cast<device float *>(r.raw + (633 * p.n)))
#define a413 (reinterpret_cast<device float *>(r.raw + (287 * p.n)))
#define a414 (reinterpret_cast<device float *>(r.raw + (640 * p.n)))
#define a415 (reinterpret_cast<device int *>(r.raw + (448 * p.n)))
#define a416 (reinterpret_cast<device int *>(r.raw + (231 * p.n)))
#define a417 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a418 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a419 (reinterpret_cast<device float *>(r.raw + (589 * p.n)))
#define a420 (reinterpret_cast<device float *>(r.raw + (641 * p.n)))
#define a421 (reinterpret_cast<device float *>(r.raw + (642 * p.n)))
#define a422 (reinterpret_cast<device float *>(r.raw + (643 * p.n)))
#define a423 (reinterpret_cast<device float *>(r.raw + (644 * p.n)))
#define a424 (reinterpret_cast<device float *>(r.raw + (646 * p.n)))
#define a425 (reinterpret_cast<device float *>(r.raw + (645 * p.n)))
#define a426 (reinterpret_cast<device float *>(r.raw + (647 * p.n)))
#define a427 (reinterpret_cast<device float *>(r.raw + (636 * p.n)))
#define a428 (reinterpret_cast<device float *>(r.raw + (635 * p.n)))
#define a429 (reinterpret_cast<device float *>(r.raw + (634 * p.n)))
#define a430 (reinterpret_cast<device float *>(r.raw + (632 * p.n)))
#define a431 (reinterpret_cast<device float *>(r.raw + (618 * p.n)))
#define a432 (reinterpret_cast<device float *>(r.raw + (622 * p.n)))
#define a433 (reinterpret_cast<device float *>(r.raw + (621 * p.n)))
#define a434 (reinterpret_cast<device float *>(r.raw + (620 * p.n)))
#define a435 (reinterpret_cast<device float *>(r.raw + (452 * p.n)))
#define a436 (reinterpret_cast<device float *>(r.raw + (619 * p.n)))
#define a437 (reinterpret_cast<device float *>(r.raw + (617 * p.n)))
#define a438 (reinterpret_cast<device float *>(r.raw + (451 * p.n)))
#define a439 (reinterpret_cast<device float *>(r.raw + (450 * p.n)))
#define a440 (reinterpret_cast<device float *>(r.raw + (648 * p.n)))
#define a441 (reinterpret_cast<device float *>(r.raw + (649 * p.n)))
#define a442 (reinterpret_cast<device float *>(r.raw + (650 * p.n)))
#define a443 (reinterpret_cast<device float *>(r.raw + (499 * p.n)))
#define a444 (reinterpret_cast<device int *>(r.raw + (651 * p.n)))
#define a445 (reinterpret_cast<device float *>(r.raw + (451 * p.n)))
#define a446 (reinterpret_cast<device float *>(r.raw + (632 * p.n)))
#define a447 (reinterpret_cast<device float *>(r.raw + (644 * p.n)))
#define a448 (reinterpret_cast<device float *>(r.raw + (544 * p.n)))
#define a449 (reinterpret_cast<device float *>(r.raw + (589 * p.n)))
#define a450 (reinterpret_cast<device float *>(r.raw + (454 * p.n)))
#define a451 (reinterpret_cast<device float *>(r.raw + (517 * p.n)))
#define a452 (reinterpret_cast<device float *>(r.raw + (598 * p.n)))
#define a453 (reinterpret_cast<device float *>(r.raw + (463 * p.n)))
#define a454 (reinterpret_cast<device float *>(r.raw + (623 * p.n)))
#define a455 (reinterpret_cast<device float *>(r.raw + (472 * p.n)))
#define a456 (reinterpret_cast<device float *>(r.raw + (481 * p.n)))
#define a457 (reinterpret_cast<device float *>(r.raw + (633 * p.n)))
#define a458 (reinterpret_cast<device float *>(r.raw + (643 * p.n)))
#define a459 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a460 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a461 (reinterpret_cast<device int *>(r.raw + (453 * p.n)))
#define a462 (reinterpret_cast<device float *>(r.raw + (481 * p.n)))
#define a463 (reinterpret_cast<device float *>(r.raw + (472 * p.n)))
#define a464 (reinterpret_cast<device float *>(r.raw + (642 * p.n)))
#define a465 (reinterpret_cast<device float *>(r.raw + (641 * p.n)))
#define a466 (reinterpret_cast<device float *>(r.raw + (287 * p.n)))
#define a467 (reinterpret_cast<device float *>(r.raw + (616 * p.n)))
#define a468 (reinterpret_cast<device float *>(r.raw + (652 * p.n)))
#define a469 (reinterpret_cast<device float *>(r.raw + (652 * p.n)))
#define a470 (reinterpret_cast<device float *>(r.raw + (643 * p.n)))
#define a471 (reinterpret_cast<device float *>(r.raw + (633 * p.n)))
#define a472 (reinterpret_cast<device float *>(r.raw + (644 * p.n)))
#define a473 (reinterpret_cast<device float *>(r.raw + (638 * p.n)))
#define a474 (reinterpret_cast<device float *>(r.raw + (653 * p.n)))
#define a475 (reinterpret_cast<device float *>(r.raw + (654 * p.n)))
#define a476 (reinterpret_cast<device float *>(r.raw + (655 * p.n)))
#define a477 (reinterpret_cast<device float *>(r.raw + (656 * p.n)))
#define a478 (reinterpret_cast<device float *>(r.raw + (657 * p.n)))
#define a479 (reinterpret_cast<device float *>(r.raw + (658 * p.n)))
#define a480 (reinterpret_cast<device float *>(r.raw + (659 * p.n)))
#define a481 (reinterpret_cast<device float *>(r.raw + (632 * p.n)))
#define a482 (reinterpret_cast<device float *>(r.raw + (632 * p.n)))
#define a483 (reinterpret_cast<device int *>(r.raw + (233 * p.n)))
#define a484 (reinterpret_cast<device int *>(r.raw + (233 * p.n)))
#define a485 (reinterpret_cast<device int *>(r.raw + (233 * p.n)))
#define a486 (reinterpret_cast<device int *>(r.raw + (233 * p.n)))
#define a487 (reinterpret_cast<device int *>(r.raw + (233 * p.n)))
#define a488 (reinterpret_cast<device int *>(r.raw + (233 * p.n)))
#define a489 (reinterpret_cast<device float *>(r.raw + (472 * p.n)))
#define a490 (reinterpret_cast<device int *>(r.raw + (659 * p.n)))
#define a491 (reinterpret_cast<device float *>(r.raw + (658 * p.n)))
#define a492 (reinterpret_cast<device float *>(r.raw + (657 * p.n)))
#define a493 (reinterpret_cast<device float *>(r.raw + (656 * p.n)))
#define a494 (reinterpret_cast<device float *>(r.raw + (655 * p.n)))
#define a495 (reinterpret_cast<device float *>(r.raw + (654 * p.n)))
#define a496 (reinterpret_cast<device float *>(r.raw + (653 * p.n)))
#define a497 (reinterpret_cast<device float *>(r.raw + (638 * p.n)))
#define a498 (reinterpret_cast<device float *>(r.raw + (644 * p.n)))
#define a499 (reinterpret_cast<device float *>(r.raw + (633 * p.n)))
#define a500 (reinterpret_cast<device float *>(r.raw + (643 * p.n)))
#define a501 (reinterpret_cast<device float *>(r.raw + (652 * p.n)))
#define a502 (reinterpret_cast<device float *>(r.raw + (616 * p.n)))
#define a503 (reinterpret_cast<device float *>(r.raw + (287 * p.n)))
#define a504 (reinterpret_cast<device float *>(r.raw + (641 * p.n)))
#define a505 (reinterpret_cast<device float *>(r.raw + (642 * p.n)))
#define a506 (reinterpret_cast<device float *>(r.raw + (651 * p.n)))
#define a507 (reinterpret_cast<device float *>(r.raw + (649 * p.n)))
#define a508 (reinterpret_cast<device float *>(r.raw + (648 * p.n)))
#define a509 (reinterpret_cast<device float *>(r.raw + (450 * p.n)))
#define a510 (reinterpret_cast<device float *>(r.raw + (617 * p.n)))
#define a511 (reinterpret_cast<device float *>(r.raw + (619 * p.n)))
#define a512 (reinterpret_cast<device float *>(r.raw + (452 * p.n)))
#define a513 (reinterpret_cast<device float *>(r.raw + (621 * p.n)))
#define a514 (reinterpret_cast<device float *>(r.raw + (622 * p.n)))
#define a515 (reinterpret_cast<device float *>(r.raw + (618 * p.n)))
#define a516 (reinterpret_cast<device float *>(r.raw + (634 * p.n)))
#define a517 (reinterpret_cast<device float *>(r.raw + (635 * p.n)))
#define a518 (reinterpret_cast<device int *>(r.raw + (233 * p.n)))
#define a519 (reinterpret_cast<device float *>(r.raw + (635 * p.n)))
#define a520 (reinterpret_cast<device int *>(r.raw + (443 * p.n)))
#define a521 (reinterpret_cast<device int *>(r.raw + (233 * p.n)))
#define a522 (reinterpret_cast<device int *>(r.raw + (443 * p.n)))
#define a523 (reinterpret_cast<device int *>(r.raw + (445 * p.n)))
#define a524 (reinterpret_cast<device float *>(r.raw + (635 * p.n)))
#define a525 (reinterpret_cast<device int *>(r.raw + (286 * p.n)))
#define a526 (reinterpret_cast<device float *>(r.raw + (368 * p.n)))
#define a527 (reinterpret_cast<device float *>(r.raw + (472 * p.n)))
#define a528 (reinterpret_cast<device float *>(r.raw + (369 * p.n)))
#define a529 (reinterpret_cast<device float *>(r.raw + (481 * p.n)))
#define a530 (reinterpret_cast<device float *>(r.raw + (481 * p.n)))
#define a531 (reinterpret_cast<device float *>(r.raw + (387 * p.n)))
#define a532 (reinterpret_cast<device float *>(r.raw + (369 * p.n)))
#define a533 (reinterpret_cast<device float *>(r.raw + (369 * p.n)))
#define a534 (reinterpret_cast<device float *>(r.raw + (405 * p.n)))
#define a535 (reinterpret_cast<device float *>(r.raw + (387 * p.n)))
#define a536 (reinterpret_cast<device float *>(r.raw + (387 * p.n)))
#define a537 (reinterpret_cast<device float *>(r.raw + (414 * p.n)))
#define a538 (reinterpret_cast<device float *>(r.raw + (405 * p.n)))
#define a539 (reinterpret_cast<device float *>(r.raw + (405 * p.n)))
#define a540 (reinterpret_cast<device float *>(r.raw + (414 * p.n)))
#define a541 (reinterpret_cast<device float *>(r.raw + (387 * p.n)))
#define a542 (reinterpret_cast<device int *>(r.raw + (445 * p.n)))
#define a543 (reinterpret_cast<device float *>(r.raw + (387 * p.n)))
#define a544 (reinterpret_cast<device float *>(r.raw + (634 * p.n)))
#define a545 (reinterpret_cast<device float *>(r.raw + (618 * p.n)))
#define a546 (reinterpret_cast<device float *>(r.raw + (622 * p.n)))
#define a547 (reinterpret_cast<device float *>(r.raw + (622 * p.n)))
#define a548 (reinterpret_cast<device float *>(r.raw + (618 * p.n)))
#define a549 (reinterpret_cast<device float *>(r.raw + (634 * p.n)))
#define a550 (reinterpret_cast<device float *>(r.raw + (634 * p.n)))
#define a551 (reinterpret_cast<device float *>(r.raw + (618 * p.n)))
#define a552 (reinterpret_cast<device float *>(r.raw + (622 * p.n)))
#define a553 (reinterpret_cast<device float *>(r.raw + (622 * p.n)))
#define a554 (reinterpret_cast<device float *>(r.raw + (618 * p.n)))
#define a555 (reinterpret_cast<device float *>(r.raw + (634 * p.n)))
#define a556 (reinterpret_cast<device float *>(r.raw + (634 * p.n)))
#define a557 (reinterpret_cast<device float *>(r.raw + (618 * p.n)))
#define a558 (reinterpret_cast<device float *>(r.raw + (622 * p.n)))
#define a559 (reinterpret_cast<device float *>(r.raw + (622 * p.n)))
#define a560 (reinterpret_cast<device float *>(r.raw + (427 * p.n)))
#define a561 (reinterpret_cast<device float *>(r.raw + (618 * p.n)))
#define a562 (reinterpret_cast<device float *>(r.raw + (618 * p.n)))
#define a563 (reinterpret_cast<device float *>(r.raw + (428 * p.n)))
#define a564 (reinterpret_cast<device float *>(r.raw + (427 * p.n)))
#define a565 (reinterpret_cast<device float *>(r.raw + (427 * p.n)))
#define a566 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a567 (reinterpret_cast<device float *>(r.raw + (428 * p.n)))
#define a568 (reinterpret_cast<device float *>(r.raw + (428 * p.n)))
#define a569 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a570 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a571 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a572 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a573 (reinterpret_cast<device float *>(r.raw + (428 * p.n)))
#define a574 (reinterpret_cast<device float *>(r.raw + (428 * p.n)))
#define a575 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a576 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a577 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a578 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a579 (reinterpret_cast<device float *>(r.raw + (428 * p.n)))
#define a580 (reinterpret_cast<device float *>(r.raw + (428 * p.n)))
#define a581 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a582 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a583 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a584 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a585 (reinterpret_cast<device float *>(r.raw + (428 * p.n)))
#define a586 (reinterpret_cast<device float *>(r.raw + (428 * p.n)))
#define a587 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a588 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a589 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a590 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a591 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a592 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a593 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a594 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a595 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a596 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a597 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a598 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a599 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a600 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a601 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a602 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a603 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a604 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a605 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a606 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a607 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a608 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a609 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a610 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a611 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a612 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a613 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a614 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a615 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a616 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a617 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a618 (reinterpret_cast<device float *>(r.raw + (439 * p.n)))
#define a619 (reinterpret_cast<device float *>(r.raw + (439 * p.n)))
#define a620 (reinterpret_cast<device float *>(r.raw + (368 * p.n)))
#define a621 (reinterpret_cast<device float *>(r.raw + (440 * p.n)))
#define a622 (reinterpret_cast<device float *>(r.raw + (440 * p.n)))
#define a623 (reinterpret_cast<device float *>(r.raw + (387 * p.n)))
#define a624 (reinterpret_cast<device float *>(r.raw + (378 * p.n)))
#define a625 (reinterpret_cast<device float *>(r.raw + (396 * p.n)))
#define a626 (reinterpret_cast<device float *>(r.raw + (368 * p.n)))
#define a627 (reinterpret_cast<device float *>(r.raw + (273 * p.n)))
#define a628 (reinterpret_cast<device float *>(r.raw + (439 * p.n)))
#define a629 (reinterpret_cast<device float *>(r.raw + (439 * p.n)))
#define a630 (reinterpret_cast<device float *>(r.raw + (423 * p.n)))
#define a631 (reinterpret_cast<device float *>(r.raw + (273 * p.n)))
#define a632 (reinterpret_cast<device float *>(r.raw + (273 * p.n)))
#define a633 (reinterpret_cast<device float *>(r.raw + (424 * p.n)))
#define a634 (reinterpret_cast<device float *>(r.raw + (423 * p.n)))
#define a635 (reinterpret_cast<device float *>(r.raw + (423 * p.n)))
#define a636 (reinterpret_cast<device float *>(r.raw + (425 * p.n)))
#define a637 (reinterpret_cast<device float *>(r.raw + (424 * p.n)))
#define a638 (reinterpret_cast<device float *>(r.raw + (424 * p.n)))
#define a639 (reinterpret_cast<device float *>(r.raw + (426 * p.n)))
#define a640 (reinterpret_cast<device float *>(r.raw + (425 * p.n)))
#define a641 (reinterpret_cast<device float *>(r.raw + (425 * p.n)))
#define a642 (reinterpret_cast<device float *>(r.raw + (430 * p.n)))
#define a643 (reinterpret_cast<device float *>(r.raw + (426 * p.n)))
#define a644 (reinterpret_cast<device float *>(r.raw + (426 * p.n)))
#define a645 (reinterpret_cast<device float *>(r.raw + (431 * p.n)))
#define a646 (reinterpret_cast<device float *>(r.raw + (430 * p.n)))
#define a647 (reinterpret_cast<device float *>(r.raw + (430 * p.n)))
#define a648 (reinterpret_cast<device float *>(r.raw + (432 * p.n)))
#define a649 (reinterpret_cast<device float *>(r.raw + (431 * p.n)))
#define a650 (reinterpret_cast<device float *>(r.raw + (431 * p.n)))
#define a651 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a652 (reinterpret_cast<device float *>(r.raw + (432 * p.n)))
#define a653 (reinterpret_cast<device float *>(r.raw + (432 * p.n)))
#define a654 (reinterpret_cast<device float *>(r.raw + (435 * p.n)))
#define a655 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a656 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a657 (reinterpret_cast<device float *>(r.raw + (436 * p.n)))
#define a658 (reinterpret_cast<device float *>(r.raw + (435 * p.n)))
#define a659 (reinterpret_cast<device float *>(r.raw + (435 * p.n)))
#define a660 (reinterpret_cast<device float *>(r.raw + (436 * p.n)))
#define a661 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a662 (reinterpret_cast<device float *>(r.raw + (436 * p.n)))
#define a663 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a664 (reinterpret_cast<device float *>(r.raw + (436 * p.n)))
#define a665 (reinterpret_cast<device float *>(r.raw + (436 * p.n)))
#define a666 (reinterpret_cast<device float *>(r.raw + (299 * p.n)))
#define a667 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a668 (reinterpret_cast<device float *>(r.raw + (299 * p.n)))
#define a669 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a670 (reinterpret_cast<device float *>(r.raw + (299 * p.n)))
#define a671 (reinterpret_cast<device float *>(r.raw + (299 * p.n)))
#define a672 (reinterpret_cast<device float *>(r.raw + (437 * p.n)))
#define a673 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a674 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a675 (reinterpret_cast<device float *>(r.raw + (440 * p.n)))
#define a676 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a677 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a678 (reinterpret_cast<device float *>(r.raw + (440 * p.n)))
#define a679 (reinterpret_cast<device float *>(r.raw + (440 * p.n)))
#define a680 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a681 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a682 (reinterpret_cast<device float *>(r.raw + (440 * p.n)))
#define a683 (reinterpret_cast<device float *>(r.raw + (440 * p.n)))
#define a684 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a685 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a686 (reinterpret_cast<device int *>(r.raw + (444 * p.n)))
#define a687 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a688 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a689 (reinterpret_cast<device int *>(r.raw + (444 * p.n)))
#define a690 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a691 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a692 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a693 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a694 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a695 (reinterpret_cast<device float *>(r.raw + (440 * p.n)))
#define a696 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a697 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a698 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a699 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a700 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a701 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a702 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a703 (reinterpret_cast<device int *>(r.raw + (444 * p.n)))
#define a704 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a705 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a706 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a707 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a708 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a709 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a710 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a711 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a712 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a713 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a714 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a715 (reinterpret_cast<device float *>(r.raw + (433 * p.n)))
#define a716 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a717 (reinterpret_cast<device float *>(r.raw + (275 * p.n)))
#define a718 (reinterpret_cast<device float *>(r.raw + (276 * p.n)))
#define a719 (reinterpret_cast<device float *>(r.raw + (275 * p.n)))
#define a720 (reinterpret_cast<device float *>(r.raw + (276 * p.n)))
#define a721 (reinterpret_cast<device float *>(r.raw + (275 * p.n)))
#define a722 (reinterpret_cast<device float *>(r.raw + (276 * p.n)))
#define a723 (reinterpret_cast<device float *>(r.raw + (275 * p.n)))
#define a724 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a725 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a726 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a727 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a728 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a729 (reinterpret_cast<device int *>(r.raw + (232 * p.n)))
#define a730 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a731 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a732 (reinterpret_cast<device int *>(r.raw + (232 * p.n)))
#define a733 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a734 (reinterpret_cast<device float *>(r.raw + (396 * p.n)))
#define a735 (reinterpret_cast<device float *>(r.raw + (378 * p.n)))
#define a736 (reinterpret_cast<device float *>(r.raw + (387 * p.n)))
#define a737 (reinterpret_cast<device float *>(r.raw + (414 * p.n)))
#define a738 (reinterpret_cast<device float *>(r.raw + (405 * p.n)))
#define a739 (reinterpret_cast<device float *>(r.raw + (369 * p.n)))
#define a740 (reinterpret_cast<device float *>(r.raw + (481 * p.n)))
#define a741 (reinterpret_cast<device float *>(r.raw + (472 * p.n)))
#define a742 (reinterpret_cast<device float *>(r.raw + (454 * p.n)))
#define a743 (reinterpret_cast<device float *>(r.raw + (499 * p.n)))
#define a744 (reinterpret_cast<device float *>(r.raw + (508 * p.n)))
#define a745 (reinterpret_cast<device float *>(r.raw + (562 * p.n)))
#define a746 (reinterpret_cast<device float *>(r.raw + (580 * p.n)))
#define a747 (reinterpret_cast<device float *>(r.raw + (607 * p.n)))
#define a748 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a749 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a750 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a751 (reinterpret_cast<device float *>(r.raw + (589 * p.n)))
#define a752 (reinterpret_cast<device float *>(r.raw + (544 * p.n)))
#define a753 (reinterpret_cast<device float *>(r.raw + (535 * p.n)))
#define a754 (reinterpret_cast<device float *>(r.raw + (526 * p.n)))
#define a755 (reinterpret_cast<device float *>(r.raw + (275 * p.n)))
#define a756 (reinterpret_cast<device float *>(r.raw + (276 * p.n)))
#define a757 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a758 (reinterpret_cast<device float *>(r.raw + (440 * p.n)))
#define a759 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a760 (reinterpret_cast<device int *>(r.raw + (232 * p.n)))
#define a761 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a762 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a763 (reinterpret_cast<device float *>(r.raw + (607 * p.n)))
#define a764 (reinterpret_cast<device float *>(r.raw + (437 * p.n)))
#define a765 (reinterpret_cast<device float *>(r.raw + (294 * p.n)))
#define a766 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a767 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a768 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a769 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a770 (reinterpret_cast<device int *>(r.raw + (442 * p.n)))
#define a771 (reinterpret_cast<device int *>(r.raw + (232 * p.n)))
#define a772 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a773 (reinterpret_cast<device float *>(r.raw + (481 * p.n)))
#define a774 (reinterpret_cast<device float *>(r.raw + (294 * p.n)))
#define a775 (reinterpret_cast<device float *>(r.raw + (275 * p.n)))
#define a776 (reinterpret_cast<device float *>(r.raw + (299 * p.n)))
#define a777 (reinterpret_cast<device float *>(r.raw + (436 * p.n)))
#define a778 (reinterpret_cast<device float *>(r.raw + (297 * p.n)))
#define a779 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a780 (reinterpret_cast<device float *>(r.raw + (378 * p.n)))
#define a781 (reinterpret_cast<device float *>(r.raw + (580 * p.n)))
#define a782 (reinterpret_cast<device float *>(r.raw + (562 * p.n)))
#define a783 (reinterpret_cast<device float *>(r.raw + (508 * p.n)))
#define a784 (reinterpret_cast<device float *>(r.raw + (499 * p.n)))
#define a785 (reinterpret_cast<device float *>(r.raw + (454 * p.n)))
#define a786 (reinterpret_cast<device float *>(r.raw + (472 * p.n)))
#define a787 (reinterpret_cast<device float *>(r.raw + (571 * p.n)))
#define a788 (reinterpret_cast<device float *>(r.raw + (571 * p.n)))
#define a789 (reinterpret_cast<device float *>(r.raw + (437 * p.n)))
#define a790 (reinterpret_cast<device float *>(r.raw + (298 * p.n)))
#define a791 (reinterpret_cast<device float *>(r.raw + (435 * p.n)))
#define a792 (reinterpret_cast<device float *>(r.raw + (432 * p.n)))
#define a793 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a794 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a795 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a796 (reinterpret_cast<device int *>(r.raw + (446 * p.n)))
#define a797 (reinterpret_cast<device float *>(r.raw + (607 * p.n)))
#define a798 (reinterpret_cast<device float *>(r.raw + (472 * p.n)))
#define a799 (reinterpret_cast<device float *>(r.raw + (291 * p.n)))
#define a800 (reinterpret_cast<device float *>(r.raw + (431 * p.n)))
#define a801 (reinterpret_cast<device float *>(r.raw + (430 * p.n)))
#define a802 (reinterpret_cast<device float *>(r.raw + (426 * p.n)))
#define a803 (reinterpret_cast<device float *>(r.raw + (425 * p.n)))
#define a804 (reinterpret_cast<device float *>(r.raw + (425 * p.n)))
#define a805 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a806 (reinterpret_cast<device float *>(r.raw + (435 * p.n)))
#define a807 (reinterpret_cast<device float *>(r.raw + (285 * p.n)))
#define a808 (reinterpret_cast<device float *>(r.raw + (424 * p.n)))
#define a809 (reinterpret_cast<device float *>(r.raw + (423 * p.n)))
#define a810 (reinterpret_cast<device float *>(r.raw + (273 * p.n)))
#define a811 (reinterpret_cast<device float *>(r.raw + (439 * p.n)))
#define a812 (reinterpret_cast<device float *>(r.raw + (368 * p.n)))
#define a813 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a814 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a815 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a816 (reinterpret_cast<device int *>(r.raw + (449 * p.n)))
#define a817 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a818 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a819 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a820 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a821 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a822 (reinterpret_cast<device float *>(r.raw + (429 * p.n)))
#define a823 (reinterpret_cast<device int *>(r.raw + (449 * p.n)))
#define a824 (reinterpret_cast<device int *>(r.raw + (449 * p.n)))
#define a825 (reinterpret_cast<device int *>(r.raw + (449 * p.n)))
#define a826 (reinterpret_cast<device int *>(r.raw + (449 * p.n)))
#define a827 (reinterpret_cast<device float *>(r.raw + (472 * p.n)))
#define a828 (reinterpret_cast<device float *>(r.raw + (434 * p.n)))
#define a829 (reinterpret_cast<device float *>(r.raw + (270 * p.n)))
#define a830 (reinterpret_cast<device float *>(r.raw + (295 * p.n)))
#define a831 (reinterpret_cast<device float *>(r.raw + (441 * p.n)))
#define a832 (reinterpret_cast<device float *>(r.raw + (368 * p.n)))
#define a833 (reinterpret_cast<device float *>(r.raw + (439 * p.n)))
#define a834 (reinterpret_cast<device float *>(r.raw + (273 * p.n)))
#define a835 (reinterpret_cast<device float *>(r.raw + (423 * p.n)))
#define a836 (reinterpret_cast<device float *>(r.raw + (424 * p.n)))
#define a837 (reinterpret_cast<device float *>(r.raw + (285 * p.n)))
#define a838 (reinterpret_cast<device float *>(r.raw + (435 * p.n)))
#define a839 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
#define a840 (reinterpret_cast<device int *>(r.raw + (449 * p.n)))
#define a841 (reinterpret_cast<device float *>(r.raw + (261 * p.n)))
#define a842 (reinterpret_cast<device float *>(r.raw + (261 * p.n)))
#define a843 (reinterpret_cast<device float *>(r.raw + (261 * p.n)))
#define a844 (reinterpret_cast<device float *>(r.raw + (261 * p.n)))
#define a845 (reinterpret_cast<device float *>(r.raw + (261 * p.n)))
#define a846 (reinterpret_cast<device float *>(r.raw + (438 * p.n)))
// Generated by tools/ruc_fused/gen_sfctmp.py from the retained oracle.
// Each elementwise operation is an explicit IEEE rounding boundary.
static inline void ruc_sfctmp_zero_outputs(constant RucSfctmpResources& r, constant RucSfctmpParams& p, int n, int c) {
    for (int k = 0; k < RUC_NZS; ++k) a128[k * n + c] = 0;
    for (int k = 0; k < RUC_NZS; ++k) a129[k * n + c] = 0;
    for (int k = 0; k < RUC_NZS; ++k) a130[k * n + c] = 0;
    for (int k = 0; k < RUC_NZS; ++k) a131[k * n + c] = 0;
    for (int k = 0; k < RUC_NZS; ++k) a132[k * n + c] = 0;
    for (int k = 0; k < RUC_NZS; ++k) a133[k * n + c] = 0;
    a525[c] = 0;
    a134[c] = 0;
    a135[c] = 0;
    a136[c] = 0;
    a137[c] = 0;
    a138[c] = 0;
    a139[c] = 0;
    a140[c] = 0;
    a141[c] = 0;
    a142[c] = 0;
    a143[c] = 0;
    a144[c] = 0;
    a145[c] = 0;
    a146[c] = 0;
    a524[c] = 0;
    a148[c] = 0;
    a149[c] = 0;
    a150[c] = 0;
    a151[c] = 0;
    a152[c] = 0;
    a153[c] = 0;
    a154[c] = 0;
    a155[c] = 0;
    a156[c] = 0;
    a157[c] = 0;
    a158[c] = 0;
    a159[c] = 0;
    a160[c] = 0;
    a161[c] = 0;
    a162[c] = 0;
    a163[c] = 0;
    a164[c] = 0;
    a165[c] = 0;
    a166[c] = 0;
    a167[c] = 0;
    a168[c] = 0;
    a169[c] = 0;
    a170[c] = 0;
    a171[c] = 0;
    a172[c] = 0;
    a173[c] = 0;
    a174[c] = 0;
    a175[c] = 0;
    a176[c] = 0;
    a177[c] = 0;
    a178[c] = 0;
    a179[c] = 0;
    a180[c] = 0;
    a182[c] = 0;
}
kernel void ruc_sfctmp_stage0(
    constant RucSfctmpResources& r [[buffer(0)]],
    constant RucSfctmpParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int n = p.n;
    const float delt = p.delt;
    const float c1sn = p.c1sn;
    const float c2sn = p.c2sn;
    const float rsmax = p.rsmax;
    const int isice = p.isice;
    const int urban = p.urban;
    const int ncategory = p.ncategory;
    const int limit = p.limit;
    const uint forms = uint(p.forms);
    const int c = int(tid);
    
    if (c >= n) return;
    r.raw[p.alive_base + uint(c)] = false;
    if (r.raw[p.run_base + uint(c)] == 0u) { ruc_sfctmp_zero_outputs(r, p, n, c); return; }
    bool failed = false;
    if (true) { if (limit <= 0) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a69[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 1, memory_order_relaxed); } }
    if (true) { a74[c] = (a69[c] < as_type<float>(0x00000000)); }
    if (true) { if (a74[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 2, memory_order_relaxed); } }
    if (true) { if (limit <= 3) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (limit <= 4) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (limit <= 5) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (limit <= 6) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (limit <= 7) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (limit <= 8) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a6[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 9, memory_order_relaxed); } } }
    if (true) { if (limit <= 10) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a7[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 11, memory_order_relaxed); } } }
    if (true) { if (limit <= 12) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a8[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 13, memory_order_relaxed); } } }
    if (true) { if (limit <= 14) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a9[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 15, memory_order_relaxed); } } }
    if (true) { if (limit <= 16) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a10[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 17, memory_order_relaxed); } } }
    if (true) { if (limit <= 18) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a11[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 19, memory_order_relaxed); } } }
    if (true) { if (limit <= 20) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a12[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 21, memory_order_relaxed); } }
    if (true) { if (limit <= 22) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a13[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 23, memory_order_relaxed); } }
    if (true) { if (limit <= 24) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 25, memory_order_relaxed); } }
    if (true) { if (limit <= 26) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a15[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 27, memory_order_relaxed); } }
    if (true) { if (limit <= 28) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a16[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 29, memory_order_relaxed); } }
    if (true) { if (limit <= 30) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a17[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 0, 1u << 31, memory_order_relaxed); } }
    if (true) { if (limit <= 32) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a18[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 1, memory_order_relaxed); } }
    if (true) { if (limit <= 34) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a19[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 3, memory_order_relaxed); } }
    if (true) { if (limit <= 36) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a20[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 5, memory_order_relaxed); } }
    if (true) { if (limit <= 38) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a21[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 7, memory_order_relaxed); } }
    if (true) { if (limit <= 40) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a22[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 9, memory_order_relaxed); } }
    if (true) { if (limit <= 42) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a23[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 11, memory_order_relaxed); } }
    if (true) { if (limit <= 44) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a24[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 13, memory_order_relaxed); } }
    if (true) { if (limit <= 46) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a25[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 15, memory_order_relaxed); } }
    if (true) { if (limit <= 48) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a26[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 17, memory_order_relaxed); } }
    if (true) { if (limit <= 50) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a27[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 19, memory_order_relaxed); } }
    if (true) { if (limit <= 52) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a28[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 21, memory_order_relaxed); } }
    if (true) { if (limit <= 54) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a29[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 23, memory_order_relaxed); } }
    if (true) { if (limit <= 56) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a30[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 25, memory_order_relaxed); } }
    if (true) { if (limit <= 58) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a31[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 27, memory_order_relaxed); } }
    if (true) { if (limit <= 60) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a32[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 29, memory_order_relaxed); } }
    if (true) { if (limit <= 62) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a33[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 1, 1u << 31, memory_order_relaxed); } }
    if (true) { if (limit <= 64) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a34[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 1, memory_order_relaxed); } }
    if (true) { if (limit <= 66) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a35[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 3, memory_order_relaxed); } }
    if (true) { if (limit <= 68) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a36[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 5, memory_order_relaxed); } }
    if (true) { if (limit <= 70) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a37[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 7, memory_order_relaxed); } }
    if (true) { if (limit <= 72) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a38[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 9, memory_order_relaxed); } }
    if (true) { if (limit <= 74) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a39[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 11, memory_order_relaxed); } }
    if (true) { if (limit <= 76) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 13, memory_order_relaxed); } }
    if (true) { if (limit <= 78) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a41[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 15, memory_order_relaxed); } }
    if (true) { if (limit <= 80) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a42[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 17, memory_order_relaxed); } }
    if (true) { if (limit <= 82) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a43[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 19, memory_order_relaxed); } }
    if (true) { if (limit <= 84) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 21, memory_order_relaxed); } }
    if (true) { if (limit <= 86) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a45[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 23, memory_order_relaxed); } }
    if (true) { if (limit <= 88) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a46[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 25, memory_order_relaxed); } }
    if (true) { if (limit <= 90) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a47[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 27, memory_order_relaxed); } }
    if (true) { if (limit <= 92) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a48[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 29, memory_order_relaxed); } }
    if (true) { if (limit <= 94) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a49[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 2, 1u << 31, memory_order_relaxed); } }
    if (true) { if (limit <= 96) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 1, memory_order_relaxed); } }
    if (true) { if (limit <= 98) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 3, memory_order_relaxed); } }
    if (true) { if (limit <= 100) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a52[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 5, memory_order_relaxed); } }
    if (true) { if (limit <= 102) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a53[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 7, memory_order_relaxed); } }
    if (true) { if (limit <= 104) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a54[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 9, memory_order_relaxed); } }
    if (true) { if (limit <= 106) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a55[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 11, memory_order_relaxed); } }
    if (true) { if (limit <= 108) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a56[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 13, memory_order_relaxed); } }
    if (true) { if (limit <= 110) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a57[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 15, memory_order_relaxed); } }
    if (true) { if (limit <= 112) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a58[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 17, memory_order_relaxed); } }
    if (true) { if (limit <= 114) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a59[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 19, memory_order_relaxed); } }
    if (true) { if (limit <= 116) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a60[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 21, memory_order_relaxed); } }
    if (true) { if (limit <= 118) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a61[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 23, memory_order_relaxed); } }
    if (true) { if (limit <= 120) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a62[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 25, memory_order_relaxed); } }
    if (true) { if (limit <= 122) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a63[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 27, memory_order_relaxed); } }
    if (true) { if (limit <= 124) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a64[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 29, memory_order_relaxed); } }
    if (true) { if (limit <= 126) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a65[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 3, 1u << 31, memory_order_relaxed); } }
    if (true) { if (limit <= 128) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a66[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 1, memory_order_relaxed); } }
    if (true) { if (limit <= 130) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a67[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 3, memory_order_relaxed); } }
    if (true) { if (limit <= 132) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (!isfinite(a68[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 5, memory_order_relaxed); } }
    if (true) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (limit <= 134) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (limit <= 135) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if ((a72[c] < 1 || a72[c] >= RUC_NZS)) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 8, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (true) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (limit <= 137) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { if (limit <= 138) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a7[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 11, memory_order_relaxed); } } }
    if (true) { if (!isfinite(a12[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 12, memory_order_relaxed); } }
    if (true) { if (!isfinite(a13[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 13, memory_order_relaxed); } }
    if (true) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 14, memory_order_relaxed); } }
    if (true) { if (!isfinite(a15[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 15, memory_order_relaxed); } }
    if (true) { if (!isfinite(a16[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 16, memory_order_relaxed); } }
    if (true) { if (!isfinite(a17[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 17, memory_order_relaxed); } }
    if (true) { if (!isfinite(a18[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 18, memory_order_relaxed); } }
    if (true) { if (!isfinite(a19[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 19, memory_order_relaxed); } }
    if (true) { if (!isfinite(a20[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 20, memory_order_relaxed); } }
    if (true) { if (!isfinite(a21[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 21, memory_order_relaxed); } }
    if (true) { if (!isfinite(a22[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 22, memory_order_relaxed); } }
    if (true) { if (!isfinite(a23[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 23, memory_order_relaxed); } }
    if (true) { if (!isfinite(a24[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 24, memory_order_relaxed); } }
    if (true) { if (!isfinite(a25[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 25, memory_order_relaxed); } }
    if (true) { if (!isfinite(a26[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 26, memory_order_relaxed); } }
    if (true) { if (!isfinite(a27[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 27, memory_order_relaxed); } }
    if (true) { if (!isfinite(a28[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 28, memory_order_relaxed); } }
    if (true) { if (!isfinite(a29[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 29, memory_order_relaxed); } }
    if (true) { if (!isfinite(a30[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 30, memory_order_relaxed); } }
    if (true) { if (!isfinite(a31[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 4, 1u << 31, memory_order_relaxed); } }
    if (true) { if (!isfinite(a32[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 0, memory_order_relaxed); } }
    if (true) { if (!isfinite(a33[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 1, memory_order_relaxed); } }
    if (true) { if (!isfinite(a34[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 2, memory_order_relaxed); } }
    if (true) { if (!isfinite(a35[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 3, memory_order_relaxed); } }
    if (true) { if (!isfinite(a36[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 4, memory_order_relaxed); } }
    if (true) { if (!isfinite(a37[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 5, memory_order_relaxed); } }
    if (true) { if (limit <= 166) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { a75[c] = (a70[c] < 1); }
    if (true) { a76[c] = (a70[c] > ncategory); }
    if (true) { a77[c] = (a75[c] || a76[c]); }
    if (true) { if (a77[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 7, memory_order_relaxed); } }
    if (true) { a78[c] = (a71[c] < 1); }
    if (true) { a79[c] = (a71[c] > ncategory); }
    if (true) { a80[c] = (a78[c] || a79[c]); }
    if (true) { if (a80[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 8, memory_order_relaxed); } }
    if (true) { if ((isice < 1 || isice > ncategory)) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 9, memory_order_relaxed); } }
    if (true) { a81[c] = (a32[c] <= as_type<float>(0x00000000)); }
    if (true) { if (a81[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 10, memory_order_relaxed); } }
    if (true) { a82[c] = (a35[c] >= as_type<float>(0x3f800000)); }
    if (true) { if (a82[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 11, memory_order_relaxed); } }
    if (true) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { ruc_snow_preparation(a7, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a70, a71, a2, a3, delt, c1sn, c2sn, isice, urban, 2, a83, a84, a85, a86, a87, a88, a89, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a120, a121, a122, a123, a124, a125, a126, a127, n, c, forms); }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a83[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 12, memory_order_relaxed); } } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a84[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 13, memory_order_relaxed); } } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a85[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 14, memory_order_relaxed); } } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a86[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 15, memory_order_relaxed); } } }
    if (true) { if (!isfinite(a87[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 16, memory_order_relaxed); } }
    if (true) { if (!isfinite(a88[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 17, memory_order_relaxed); } }
    if (true) { if (!isfinite(a89[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 18, memory_order_relaxed); } }
    if (true) { if (!isfinite(a90[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 19, memory_order_relaxed); } }
    if (true) { if (!isfinite(a91[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 20, memory_order_relaxed); } }
    if (true) { if (!isfinite(a92[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 21, memory_order_relaxed); } }
    if (true) { if (!isfinite(a93[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 22, memory_order_relaxed); } }
    if (true) { if (!isfinite(a94[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 23, memory_order_relaxed); } }
    if (true) { if (!isfinite(a95[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 24, memory_order_relaxed); } }
    if (true) { if (!isfinite(a96[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 25, memory_order_relaxed); } }
    if (true) { if (!isfinite(a97[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 26, memory_order_relaxed); } }
    if (true) { if (!isfinite(a98[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 27, memory_order_relaxed); } }
    if (true) { if (!isfinite(a99[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 28, memory_order_relaxed); } }
    if (true) { if (!isfinite(a100[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 29, memory_order_relaxed); } }
    if (true) { if (!isfinite(a101[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 30, memory_order_relaxed); } }
    if (true) { if (!isfinite(a102[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 5, 1u << 31, memory_order_relaxed); } }
    if (true) { if (!isfinite(a103[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 0, memory_order_relaxed); } }
    if (true) { if (!isfinite(a104[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 1, memory_order_relaxed); } }
    if (true) { if (!isfinite(a105[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 2, memory_order_relaxed); } }
    if (true) { if (!isfinite(a106[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 3, memory_order_relaxed); } }
    if (true) { if (!isfinite(a107[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 4, memory_order_relaxed); } }
    if (true) { if (!isfinite(a108[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 5, memory_order_relaxed); } }
    if (true) { if (!isfinite(a109[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 6, memory_order_relaxed); } }
    if (true) { if (!isfinite(a110[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 7, memory_order_relaxed); } }
    if (true) { if (!isfinite(a111[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 8, memory_order_relaxed); } }
    if (true) { if (!isfinite(a112[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 9, memory_order_relaxed); } }
    if (true) { if (!isfinite(a113[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 10, memory_order_relaxed); } }
    if (true) { if (!isfinite(a114[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 11, memory_order_relaxed); } }
    if (true) { if (!isfinite(a115[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 12, memory_order_relaxed); } }
    if (true) { if (!isfinite(a116[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 13, memory_order_relaxed); } }
    if (true) { if (!isfinite(a117[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 14, memory_order_relaxed); } }
    if (true) { if (!isfinite(a118[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 15, memory_order_relaxed); } }
    if (true) { if (!isfinite(a119[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 16, memory_order_relaxed); } }
    if (true) { if (!isfinite(a120[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 17, memory_order_relaxed); } }
    if (true) { if (!isfinite(a121[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 18, memory_order_relaxed); } }
    if (true) { if (!isfinite(a122[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 19, memory_order_relaxed); } }
    if (true) { if (!isfinite(a123[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 20, memory_order_relaxed); } }
    if (true) { if (!isfinite(a124[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 21, memory_order_relaxed); } }
    if (true) { if (!isfinite(a125[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 22, memory_order_relaxed); } }
    if (true) { if (!isfinite(a126[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 23, memory_order_relaxed); } }
    if (true) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a128[k * n + c] = a6[k * n + c]; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a129[k * n + c] = a7[k * n + c]; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a130[k * n + c] = a8[k * n + c]; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a131[k * n + c] = a9[k * n + c]; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a132[k * n + c] = a10[k * n + c]; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a133[k * n + c] = a11[k * n + c]; } }
    if (true) { a134[c] = a118[c]; }
    if (true) { a135[c] = a119[c]; }
    if (true) { a136[c] = a120[c]; }
    if (true) { a137[c] = a121[c]; }
    if (true) { a138[c] = a95[c]; }
    if (true) { a139[c] = a122[c]; }
    if (true) { a140[c] = a25[c]; }
    if (true) { a141[c] = a26[c]; }
    if (true) { a142[c] = a27[c]; }
    if (true) { a143[c] = a28[c]; }
    if (true) { a144[c] = a23[c]; }
    if (true) { a145[c] = a125[c]; }
    if (true) { a146[c] = a47[c]; }
    if (true) { a147[c] = a124[c]; }
    if (true) { a148[c] = a123[c]; }
    if (true) { a149[c] = a126[c]; }
    if (true) { a150[c] = a21[c]; }
    if (true) { a151[c] = a58[c]; }
    if (true) { a152[c] = a15[c]; }
    if (true) { a153[c] = 0; }
    if (true) { a154[c] = a59[c]; }
    if (true) { a155[c] = a60[c]; }
    if (true) { a156[c] = a61[c]; }
    if (true) { a157[c] = a96[c]; }
    if (true) { a158[c] = a62[c]; }
    if (true) { a159[c] = a63[c]; }
    if (true) { a160[c] = a64[c]; }
    if (true) { a161[c] = a22[c]; }
    if (true) { a162[c] = a65[c]; }
    if (true) { a163[c] = 0; }
    if (true) { a164[c] = 0; }
    if (true) { a165[c] = 0; }
    if (true) { a166[c] = 0; }
    if (true) { a167[c] = 0; }
    if (true) { a168[c] = 0; }
    if (true) { a169[c] = a66[c]; }
    if (true) { a170[c] = a67[c]; }
    if (true) { a171[c] = a68[c]; }
    if (true) { a172[c] = 0; }
    if (true) { a173[c] = 0; }
    if (true) { a174[c] = 0; }
    if (true) { a175[c] = 0; }
    if (true) { a176[c] = a100[c]; }
    if (true) { a177[c] = a105[c]; }
    if (true) { a178[c] = a98[c]; }
    if (true) { a179[c] = 0; }
    if (true) { a180[c] = 0; }
    if (true) { a181[c] = a127[c]; }
    if (true) { if (limit <= 216) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (true) { a182[c] = a73[c]; }
    if (true) { a183[c] = (a12[c] >= as_type<float>(0x3f000000)); }
    if (true) { a184[c] = (a135[c] > as_type<float>(0x00000000)); }
    if (true) { a185[c] = (a90[c] == as_type<float>(0x3f800000)); }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a186[k * n + c] = 0; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a187[k * n + c] = 0; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a188[k * n + c] = 0; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a189[k * n + c] = 0; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a190[k * n + c] = 0; } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a191[k * n + c] = 0; } }
    if (true) { a192[c] = 0; }
    if (true) { a193[c] = 0; }
    if (true) { a194[c] = 0; }
    if (true) { a195[c] = 0; }
    if (true) { a196[c] = 0; }
    if (true) { a197[c] = 0; }
    if (true) { a198[c] = 0; }
    if (true) { a199[c] = 0; }
    if (true) { a200[c] = 0; }
    if (true) { a201[c] = 0; }
    if (true) { a202[c] = 0; }
    if (true) { a203[c] = 0; }
    if (true) { a204[c] = 0; }
    if (true) { a205[c] = 0; }
    if (true) { a206[c] = 0; }
    if (true) { a207[c] = 0; }
    if (true) { a208[c] = 0; }
    if (true) { a209[c] = 0; }
    if (true) { a210[c] = 0; }
    if (true) { a211[c] = 0; }
    if (true) { a212[c] = (a184[c] && a185[c]); }
    if (true) { a213[c] = !(a183[c]); }
    if (true) { a214[c] = (a212[c] && a213[c]); }
    if (true) { a215[c] = (a184[c] && a185[c]); }
    if (true) { a216[c] = (a215[c] && a183[c]); }
    if (true) { a217[c] = !(a183[c]); }
    if (true) { a218[c] = (a184[c] && a217[c]); }
    if (true) { a219[c] = (a184[c] && a183[c]); }
    if (true) { a220[c] = !(a184[c]); }
    if (true) { a221[c] = !(a184[c]); }
    if (true) { a222[c] = !(a183[c]); }
    if (true) { a223[c] = (a221[c] && a222[c]); }
    if (true) { a224[c] = !(a184[c]); }
    if (true) { a225[c] = (a224[c] && a183[c]); }
    if (a214[c]) { a226[c] = ruc_fsub_rn(as_type<float>(0x3f800000), a24[c]); }
    if (a214[c]) { a227[c] = ruc_fmul_rn(a111[c], a226[c]); }
    if (a214[c]) { a228[c] = ruc_fmul_rn(as_type<float>(0x33738beb), a150[c]); }
    if (a214[c]) { a229[c] = ruc_fmul_rn(a228[c], a150[c]); }
    if (a214[c]) { a230[c] = ruc_fmul_rn(a229[c], a150[c]); }
    if (a214[c]) { a231[c] = ruc_fmul_rn(a230[c], a150[c]); }
    if (a214[c]) { a232[c] = ruc_fsub_rn(a43[c], a231[c]); }
    if (a214[c]) { a233[c] = ruc_fmul_rn(a115[c], a232[c]); }
    if (a214[c]) { a234[c] = ruc_fadd_rn(a227[c], a233[c]); }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a128[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 25, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a129[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 26, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a130[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 27, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a131[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 28, memory_order_relaxed); } } }
    if (a214[c]) { if ((a72[c] < 1 || a72[c] >= RUC_NZS)) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 29, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (a214[c]) { if (!isfinite(a16[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 30, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a97[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 6, 1u << 31, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a39[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 0, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 1, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a41[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 2, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a43[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 3, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a227[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 4, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a111[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 5, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a115[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 6, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a234[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 7, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 8, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a45[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 9, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a46[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 10, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a148[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 11, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a104[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 12, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a108[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 13, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a42[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 14, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a101[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 15, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a19[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 16, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a48[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 17, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a49[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 18, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 19, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 20, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a52[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 21, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a53[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 22, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a54[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 23, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a55[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 24, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a56[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 25, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a20[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 26, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a57[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 27, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 28, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a150[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 29, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a154[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 30, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a155[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 7, 1u << 31, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a156[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 0, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a47[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 1, memory_order_relaxed); } }
    if (a214[c]) { a235[c] = (a50[c] <= as_type<float>(0x00000000)); }
    if (a214[c]) { if (a235[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 2, memory_order_relaxed); } }
    if (a214[c]) { a236[c] = (a54[c] >= as_type<float>(0x00000000)); }
    if (a214[c]) { if (a236[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 3, memory_order_relaxed); } }
    if (a214[c]) { a237[c] = (a55[c] <= as_type<float>(0x00000000)); }
    if (a214[c]) { if (a237[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 4, memory_order_relaxed); } }
    if (a214[c]) { a238[c] = (a20[c] <= as_type<float>(0x00000000)); }
    if (a214[c]) { if (a238[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 5, memory_order_relaxed); } }
    if (a214[c]) { a239[c] = (a42[c] <= as_type<float>(0x00000000)); }
    if (a214[c]) { a240[c] = (a39[c] <= as_type<float>(0x00000000)); }
    if (a214[c]) { a241[c] = (a239[c] || a240[c]); }
    if (a214[c]) { if (a241[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 6, memory_order_relaxed); } }
    if (a214[c]) { a242[c] = (a47[c] < as_type<float>(0x00000000)); }
    if (a214[c]) { a243[c] = (a47[c] > as_type<float>(0x3f800000)); }
    if (a214[c]) { a244[c] = (a242[c] || a243[c]); }
    if (a214[c]) { if (a244[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 7, memory_order_relaxed); } }
    if (a214[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a245[k * n + c] = a128[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a246[k * n + c] = a129[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a247[k * n + c] = a130[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a248[k * n + c] = a131[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a249[k * n + c] = a246[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a250[k * n + c] = a245[k * n + c]; } }
    if (a214[c]) { ruc_soil_phase_partition(a245, a246, a247, a248, a50, a51, a54, a55, 1, a251, a252, a253, a254, a255, a256, a257, a258, n, c); }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a258[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 8, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a257[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 9, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a253[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 10, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a248[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 11, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a245[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 12, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a251[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 13, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a252[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 14, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a254[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 15, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a255[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 16, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a256[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 17, memory_order_relaxed); } } }
    if (a214[c]) { if (!isfinite(a48[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 18, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a49[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 19, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 20, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 21, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a54[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 22, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a55[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 23, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a56[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 24, memory_order_relaxed); } }
    if (a214[c]) { a259[c] = (a55[c] <= as_type<float>(0x00000000)); }
    if (a214[c]) { if (a259[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 25, memory_order_relaxed); } }
    if (a214[c]) { a260[c] = (a54[c] >= as_type<float>(0x00000000)); }
    if (a214[c]) { if (a260[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 26, memory_order_relaxed); } }
    if (a214[c]) { a261[c] = (a56[c] < as_type<float>(0x00000000)); }
    if (a214[c]) { if (a261[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 27, memory_order_relaxed); } }
    if (a214[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a214[c]) { ruc_soil_properties(a258, a257, a253, a248, a245, a251, a252, a254, a255, a256, a48, a49, a50, a51, a54, a55, a56, as_type<float>(0x3f666667), a262, a263, a264, a265, n, c, forms); if (p.spp_lsm != 0) { for (int k = 0; k < RUC_NZS; ++k) { const int index = k * n + c; const float original = a264[index]; const float pattern = r.pattern_spp_lsm[index]; if (p.has_field_sf != 0) r.field_sf[index] = ruc_fmul_rn(original, pattern); a264[index] = ruc_fmul_rn(original, ruc_fadd_rn(1.0f, pattern)); } } }
    if (a214[c]) { ruc_soil_canopy_setup(a245, a40, a155, a154, a44, a148, a20, a57, a51, a52, a266, a267, a268, a269, n, c); }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a251[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 28, memory_order_relaxed); } } }
    if (a214[c]) { if ((a72[c] < 1 || a72[c] >= RUC_NZS)) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 29, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (a214[c]) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 30, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a19[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 8, 1u << 31, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a111[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 0, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 1, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 2, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a52[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 3, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a53[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 4, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a46[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 5, memory_order_relaxed); } }
    if (a214[c]) { a270[c] = (a70[c] < 1); }
    if (a214[c]) { a271[c] = (a70[c] > ncategory); }
    if (a214[c]) { a272[c] = (a270[c] || a271[c]); }
    if (a214[c]) { if (a272[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 6, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (a214[c]) { a273[c] = (a52[c] <= a53[c]); }
    if (a214[c]) { if (a273[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 7, memory_order_relaxed); } }
    if (a214[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a214[c]) { ruc_transpiration(a251, a14, a19, a111, a50, a51, a52, a53, a46, a70, a0, a1, a5, a72, rsmax, a274, a275, n, c); }
    if (true) { a276[c] = 1; }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a262[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 8, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a265[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 9, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a246[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 10, memory_order_relaxed); } } }
    if (a214[c]) { if ((a72[c] < 1 || a72[c] >= RUC_NZS)) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 11, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (a214[c]) { if (!isfinite(a16[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 12, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a97[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 13, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a39[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 14, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 15, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 16, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a41[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 17, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a115[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 18, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a234[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 19, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 20, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a45[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 21, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a46[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 22, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a42[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 23, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a101[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 24, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a19[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 25, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a268[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 26, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a267[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 27, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a275[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 28, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a266[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 29, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a47[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 30, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a269[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 9, 1u << 31, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a276[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 0, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 1, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 2, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a55[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 3, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a150[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 4, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a154[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 5, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a155[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 6, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a156[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 7, memory_order_relaxed); } }
    if (a214[c]) { a277[c] = (a262[0 * n + c] <= as_type<float>(0x00000000)); }
    if (a214[c]) { if (a277[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 8, memory_order_relaxed); } }
    if (a214[c]) { a278[c] = (a265[0 * n + c] <= as_type<float>(0x00000000)); }
    if (a214[c]) { if (a278[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 9, memory_order_relaxed); } }
    if (a214[c]) { a279[c] = (a39[c] <= as_type<float>(0x00000000)); }
    if (a214[c]) { if (a279[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 10, memory_order_relaxed); } }
    if (a214[c]) { a280[c] = (a42[c] <= as_type<float>(0x00000000)); }
    if (a214[c]) { if (a280[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 11, memory_order_relaxed); } }
    if (a214[c]) { a281[c] = (a47[c] < as_type<float>(0x00000000)); }
    if (a214[c]) { a282[c] = (a47[c] > as_type<float>(0x3f800000)); }
    if (a214[c]) { a283[c] = (a281[c] || a282[c]); }
    if (a214[c]) { if (a283[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 12, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a69[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 13, memory_order_relaxed); } }
    if (a214[c]) { a290[c] = (a69[c] < as_type<float>(0x00000000)); }
    if (a214[c]) { if (a290[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 14, memory_order_relaxed); } }
    if (a214[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a214[c]) { ruc_soil_temperature_step(a262, a265, a246, a16, a97, a39, a14, a40, a115, a234, a44, a45, a42, a101, a268, a267, a275, a47, a269, a150, a154, a72, a4, delt, a69, as_type<float>(0x4a7f4f60), a284, a285, a286, a287, a288, a289, n, c); }
    if (a214[c]) { ruc_soil_phase_partition(a245, a284, a247, a248, a50, a51, a54, a55, 0, a291, a292, a293, a294, a295, a296, a297, a298, n, c); }
    if (a214[c]) { ruc_soil_prepare_moisture(a40, a287, a44, a42, a101, a268, a274, a72, a108, a299, a300, a301, a302, a303, n, c); }
    if (true) { a304[c] = 0; }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a263[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 15, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a264[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 16, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a299[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 17, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a292[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 18, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a245[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 19, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a291[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 20, memory_order_relaxed); } } }
    if (a214[c]) { if (!isfinite(a287[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 21, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a286[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 22, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a288[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 23, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a41[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 24, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 25, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a302[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 26, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 27, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a104[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 28, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a301[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 29, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a304[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 30, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a101[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 10, 1u << 31, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a304[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 0, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a269[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 1, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 2, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 3, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a52[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 4, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a56[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 5, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a303[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 6, memory_order_relaxed); } }
    if (a214[c]) { a305[c] = (a50[c] <= as_type<float>(0x00000000)); }
    if (a214[c]) { if (a305[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 7, memory_order_relaxed); } }
    if (a214[c]) { a306[c] = (a52[c] <= a51[c]); }
    if (a214[c]) { if (a306[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 8, memory_order_relaxed); } }
    if (a214[c]) { a307[c] = (a56[c] < as_type<float>(0x00000000)); }
    if (a214[c]) { if (a307[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 9, memory_order_relaxed); } }
    if (a214[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a214[c]) { ruc_soil_moisture_step(a263, a264, a299, a292, a245, a291, a287, a286, a288, a41, a40, a302, a44, a104, a301, a304, a101, a304, a269, a50, a51, a52, a56, a303, delt, a308, a309, a310, a311, a312, a313, a314, n, c); }
    if (a214[c]) { ruc_soil_finalize(a292, a284, a249, a308, a250, a248, a262, a265, a148, a301, a285, a286, a287, a288, a300, a267, a269, a303, a45, a42, a14, a39, a44, a40, a101, a234, a16, a289, delt, a315, a316, a317, a318, a319, a320, a321, a322, a323, a324, a325, a326, n, c); }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a308[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 10, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a284[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 11, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a247[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 12, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a248[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 13, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a292[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 14, memory_order_relaxed); } } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a309[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 15, memory_order_relaxed); } } }
    if (a214[c]) { if (!isfinite(a315[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 16, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a301[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 17, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a285[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 18, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a286[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 19, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a287[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 20, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a288[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 21, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a316[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 22, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a317[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 23, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a318[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 24, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a319[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 25, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a320[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 26, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a321[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 27, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a322[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 28, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a323[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 29, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a324[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 30, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a325[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 11, 1u << 31, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a311[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 0, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a312[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 1, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a310[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 2, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a313[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 3, memory_order_relaxed); } }
    if (a214[c]) { if (!isfinite(a326[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 4, memory_order_relaxed); } }
    if (a214[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a186[k * n + c] = a308[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a187[k * n + c] = a284[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a188[k * n + c] = a247[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a189[k * n + c] = a248[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a190[k * n + c] = a292[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a191[k * n + c] = a309[k * n + c]; } }
    if (a214[c]) { a192[c] = a301[c]; }
    if (a214[c]) { a193[c] = a285[c]; }
    if (a214[c]) { a194[c] = a286[c]; }
    if (a214[c]) { a195[c] = a287[c]; }
    if (a214[c]) { a196[c] = a288[c]; }
    if (a214[c]) { a197[c] = a316[c]; }
    if (a214[c]) { a198[c] = a317[c]; }
    if (a214[c]) { a199[c] = a318[c]; }
    if (a214[c]) { a200[c] = a319[c]; }
    if (a214[c]) { a201[c] = a320[c]; }
    if (a214[c]) { a202[c] = a321[c]; }
    if (a214[c]) { a203[c] = a322[c]; }
    if (a214[c]) { a204[c] = a323[c]; }
    if (a214[c]) { a205[c] = a324[c]; }
    if (a214[c]) { a206[c] = a325[c]; }
    if (a214[c]) { a207[c] = a311[c]; }
    if (a214[c]) { a208[c] = a312[c]; }
    if (a214[c]) { a209[c] = a310[c]; }
    if (a214[c]) { a210[c] = a313[c]; }
    if (a214[c]) { a211[c] = a315[c]; }
    if (a214[c]) { a177[c] = a326[c]; }
    if (a216[c]) { a327[c] = ruc_fsub_rn(as_type<float>(0x3f800000), a112[c]); }
    if (a216[c]) { a328[c] = ruc_fmul_rn(a111[c], a327[c]); }
    if (a216[c]) { a329[c] = ruc_fmul_rn(as_type<float>(0x33738beb), a150[c]); }
    if (a216[c]) { a330[c] = ruc_fmul_rn(a329[c], a150[c]); }
    if (a216[c]) { a331[c] = ruc_fmul_rn(a330[c], a150[c]); }
    if (a216[c]) { a332[c] = ruc_fmul_rn(a331[c], a150[c]); }
    if (a216[c]) { a333[c] = ruc_fsub_rn(a43[c], a332[c]); }
    if (a216[c]) { a334[c] = ruc_fmul_rn(a115[c], a333[c]); }
    if (a216[c]) { a335[c] = ruc_fadd_rn(a328[c], a334[c]); }
    if (true) { a336[c] = as_type<float>(0x3f7ae148); }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a85[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 5, memory_order_relaxed); } } }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a86[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 6, memory_order_relaxed); } } }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a129[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 7, memory_order_relaxed); } } }
    if (a216[c]) { if (!isfinite(a16[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 8, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a97[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 9, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a39[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 10, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 11, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a336[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 12, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a335[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 13, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 14, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a45[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 15, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a42[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 16, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 17, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a150[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 18, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a154[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 19, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a155[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 20, memory_order_relaxed); } }
    if (a216[c]) { a337[c] = (a86[0 * n + c] <= as_type<float>(0x00000000)); }
    if (a216[c]) { if (a337[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 21, memory_order_relaxed); } }
    if (a216[c]) { a338[c] = (a85[0 * n + c] <= as_type<float>(0x00000000)); }
    if (a216[c]) { if (a338[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 22, memory_order_relaxed); } }
    if (a216[c]) { a339[c] = (a39[c] <= as_type<float>(0x00000000)); }
    if (a216[c]) { if (a339[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 23, memory_order_relaxed); } }
    if (a216[c]) { a340[c] = (a42[c] <= as_type<float>(0x00000000)); }
    if (a216[c]) { if (a340[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 24, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a69[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 25, memory_order_relaxed); } }
    if (a216[c]) { a354[c] = (a69[c] < as_type<float>(0x00000000)); }
    if (a216[c]) { if (a354[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 26, memory_order_relaxed); } }
    if (a216[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a216[c]) { ruc_sea_ice_step(a85, a86, a129, a16, a97, a39, a40, a336, a335, a44, a45, a42, a14, a150, a154, a155, a4, delt, a69, as_type<float>(0x4a7f4f60), 0, a341, a342, a343, a344, a345, a346, a347, a348, a349, a350, a351, a352, a353, n, c); }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a355[k * n + c] = as_type<float>(0x3f800000); } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a356[k * n + c] = as_type<float>(0x00000000); } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a357[k * n + c] = as_type<float>(0x3f800000); } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a358[k * n + c] = as_type<float>(0x3f800000); } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a359[k * n + c] = as_type<float>(0x00000000); } }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a341[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 27, memory_order_relaxed); } } }
    if (a216[c]) { if (!isfinite(a342[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 28, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a343[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 29, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a344[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 30, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a345[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 12, 1u << 31, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a346[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 0, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a347[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 1, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a348[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 2, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a349[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 3, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a350[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 4, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a351[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 5, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a352[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 6, memory_order_relaxed); } }
    if (a216[c]) { if (!isfinite(a353[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 7, memory_order_relaxed); } }
    if (a216[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { a187[k * n + c] = a341[k * n + c]; } }
    if (a216[c]) { a192[c] = a342[c]; }
    if (a216[c]) { a193[c] = a343[c]; }
    if (a216[c]) { a194[c] = a344[c]; }
    if (a216[c]) { a195[c] = a345[c]; }
    if (a216[c]) { a196[c] = a346[c]; }
    if (a216[c]) { a200[c] = a347[c]; }
    if (a216[c]) { a201[c] = a348[c]; }
    if (a216[c]) { a202[c] = a349[c]; }
    if (a216[c]) { a203[c] = a350[c]; }
    if (a216[c]) { a204[c] = a351[c]; }
    if (a216[c]) { a205[c] = a352[c]; }
    if (a216[c]) { a206[c] = a353[c]; }
    if (a216[c]) { a360[c] = ruc_fmul_rn(a166[c], as_type<float>(0x3a83126f)); }
    if (a216[c]) { a163[c] = a360[c]; }
    if (a216[c]) { a164[c] = as_type<float>(0x00000000); }
    if (a216[c]) { a165[c] = as_type<float>(0x00000000); }
    if (a216[c]) { a174[c] = a16[c]; }
    if (a216[c]) { a175[c] = as_type<float>(0x00000000); }
    if (a216[c]) { a146[c] = as_type<float>(0x3f800000); }
    if (a216[c]) { a176[c] = as_type<float>(0x00000000); }
    if (a216[c]) { a148[c] = as_type<float>(0x00000000); }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { a128[k * n + c] = as_type<float>(0x3f800000); } }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { a133[k * n + c] = as_type<float>(0x00000000); } }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { a132[k * n + c] = as_type<float>(0x3f800000); } }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { a130[k * n + c] = as_type<float>(0x3f800000); } }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { a131[k * n + c] = as_type<float>(0x00000000); } }
    if (true) { a361[c] = 0; }
    if (a184[c]) { a362[c] = ruc_fsub_rn(as_type<float>(0x3f800000), a147[c]); }
    if (a184[c]) { a363[c] = ruc_fmul_rn(a111[c], a362[c]); }
    if (a184[c]) { a364[c] = ruc_fmul_rn(as_type<float>(0x33738beb), a150[c]); }
    if (a184[c]) { a365[c] = ruc_fmul_rn(a364[c], a150[c]); }
    if (a184[c]) { a366[c] = ruc_fmul_rn(a365[c], a150[c]); }
    if (a184[c]) { a367[c] = ruc_fmul_rn(a366[c], a150[c]); }
    if (a184[c]) { a368[c] = ruc_fsub_rn(a43[c], a367[c]); }
    if (a184[c]) { a369[c] = ruc_fmul_rn(a145[c], a368[c]); }
    if (a184[c]) { a370[c] = ruc_fadd_rn(a363[c], a369[c]); }
    if (a184[c]) { a361[c] = a370[c]; }
    r.raw[p.alive_base + uint(c)] = true;
}
kernel void ruc_sfctmp_stage1(
    constant RucSfctmpResources& r [[buffer(0)]],
    constant RucSfctmpParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int n = p.n;
    const float delt = p.delt;
    const float c1sn = p.c1sn;
    const float c2sn = p.c2sn;
    const float rsmax = p.rsmax;
    const int isice = p.isice;
    const int urban = p.urban;
    const int ncategory = p.ncategory;
    const int limit = p.limit;
    const uint forms = uint(p.forms);
    const int c = int(tid);
    
    if (c >= n) return;
    if (!r.raw[p.alive_base + uint(c)]) return;
    r.raw[p.alive_base + uint(c)] = false;
    if (r.raw[p.run_base + uint(c)] == 0u) { ruc_sfctmp_zero_outputs(r, p, n, c); return; }
    bool failed = false;
    if (a218[c]) { a371[c] = (a185[c] ? as_type<float>(0x3f800000) : a136[c]); }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a128[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 8, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a129[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 9, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a130[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 10, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a131[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 11, memory_order_relaxed); } } }
    if (a218[c]) { if ((a72[c] < 1 || a72[c] >= RUC_NZS)) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 12, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (a218[c]) { if (!isfinite(a38[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 13, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a138[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 14, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a16[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 15, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a97[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 16, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a92[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 17, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a135[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 18, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a134[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 19, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a371[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 20, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a137[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 21, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a39[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 22, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 23, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a41[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 24, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a145[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 25, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a361[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 26, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 27, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a45[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 28, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a46[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 29, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a148[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 30, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a108[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 13, 1u << 31, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a42[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 0, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a101[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 1, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a19[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 2, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a111[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 3, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a48[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 4, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a49[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 5, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 6, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 7, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a52[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 8, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a53[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 9, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a54[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 10, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a55[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 11, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a56[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 12, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a20[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 13, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a57[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 14, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 15, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a150[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 16, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a151[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 17, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a154[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 18, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a155[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 19, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a156[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 20, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a160[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 21, memory_order_relaxed); } }
    if (a218[c]) { a372[c] = (a50[c] <= as_type<float>(0x00000000)); }
    if (a218[c]) { if (a372[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 22, memory_order_relaxed); } }
    if (a218[c]) { a373[c] = (a54[c] >= as_type<float>(0x00000000)); }
    if (a218[c]) { if (a373[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 23, memory_order_relaxed); } }
    if (a218[c]) { a374[c] = (a55[c] <= as_type<float>(0x00000000)); }
    if (a218[c]) { if (a374[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 24, memory_order_relaxed); } }
    if (a218[c]) { a375[c] = (a20[c] <= as_type<float>(0x00000000)); }
    if (a218[c]) { if (a375[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 25, memory_order_relaxed); } }
    if (a218[c]) { a376[c] = (a42[c] <= as_type<float>(0x00000000)); }
    if (a218[c]) { a377[c] = (a39[c] <= as_type<float>(0x00000000)); }
    if (a218[c]) { a378[c] = (a376[c] || a377[c]); }
    if (a218[c]) { if (a378[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 26, memory_order_relaxed); } }
    if (a218[c]) { a379[c] = (a137[c] <= as_type<float>(0x00000000)); }
    if (a218[c]) { a380[c] = (a138[c] <= as_type<float>(0x00000000)); }
    if (a218[c]) { a381[c] = (a379[c] || a380[c]); }
    if (a218[c]) { if (a381[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 27, memory_order_relaxed); } }
    if (a218[c]) { a382[c] = (a134[c] < as_type<float>(0x00000000)); }
    if (a218[c]) { a383[c] = (a135[c] < as_type<float>(0x00000000)); }
    if (a218[c]) { a384[c] = (a382[c] || a383[c]); }
    if (a218[c]) { if (a384[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 28, memory_order_relaxed); } }
    if (a218[c]) { a385[c] = (a371[c] < as_type<float>(0x00000000)); }
    if (a218[c]) { a386[c] = (a371[c] > as_type<float>(0x3f800000)); }
    if (a218[c]) { a387[c] = (a385[c] || a386[c]); }
    if (a218[c]) { if (a387[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 29, memory_order_relaxed); } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a388[k * n + c] = a128[k * n + c]; } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a389[k * n + c] = a129[k * n + c]; } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a390[k * n + c] = a130[k * n + c]; } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a391[k * n + c] = a131[k * n + c]; } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a392[k * n + c] = a389[k * n + c]; } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a393[k * n + c] = a388[k * n + c]; } }
    if (a218[c]) { ruc_soil_phase_partition(a388, a389, a390, a391, a50, a51, a54, a55, 1, a394, a395, a396, a397, a398, a399, a400, a401, n, c); }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a401[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 30, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a400[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 14, 1u << 31, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a396[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 0, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a391[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 1, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a388[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 2, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a394[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 3, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a395[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 4, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a397[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 5, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a398[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 6, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a399[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 7, memory_order_relaxed); } } }
    if (a218[c]) { if (!isfinite(a48[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 8, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a49[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 9, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 10, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 11, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a54[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 12, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a55[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 13, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a56[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 14, memory_order_relaxed); } }
    if (a218[c]) { a402[c] = (a55[c] <= as_type<float>(0x00000000)); }
    if (a218[c]) { if (a402[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 15, memory_order_relaxed); } }
    if (a218[c]) { a403[c] = (a54[c] >= as_type<float>(0x00000000)); }
    if (a218[c]) { if (a403[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 16, memory_order_relaxed); } }
    if (a218[c]) { a404[c] = (a56[c] < as_type<float>(0x00000000)); }
    if (a218[c]) { if (a404[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 17, memory_order_relaxed); } }
    if (a218[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a218[c]) { ruc_soil_properties(a401, a400, a396, a391, a388, a394, a395, a397, a398, a399, a48, a49, a50, a51, a54, a55, a56, as_type<float>(0x3f666667), a405, a406, a407, a408, n, c, forms); }
    if (a218[c]) { if (!isfinite(a69[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 18, memory_order_relaxed); } }
    if (a218[c]) { a409[c] = (a69[c] < as_type<float>(0x00000000)); }
    if (a218[c]) { if (a409[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 19, memory_order_relaxed); } }
    if (a218[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a218[c]) { ruc_snow_soil_canopy_setup(a40, a155, a44, a42, a101, a134, a148, a20, a57, delt, a410, a411, a412, a413, a414, n, c); }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a394[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 20, memory_order_relaxed); } } }
    if (a218[c]) { if ((a72[c] < 1 || a72[c] >= RUC_NZS)) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 21, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (a218[c]) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 22, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a19[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 23, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a111[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 24, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 25, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 26, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a52[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 27, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a53[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 28, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a46[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 29, memory_order_relaxed); } }
    if (a218[c]) { a415[c] = (a181[c] < 1); }
    if (a218[c]) { a416[c] = (a181[c] > ncategory); }
    if (a218[c]) { a417[c] = (a415[c] || a416[c]); }
    if (a218[c]) { if (a417[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 30, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (a218[c]) { a418[c] = (a52[c] <= a53[c]); }
    if (a218[c]) { if (a418[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 15, 1u << 31, memory_order_relaxed); } }
    if (a218[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a218[c]) { ruc_transpiration(a394, a14, a19, a111, a50, a51, a52, a53, a46, a181, a0, a1, a5, a72, rsmax, a419, a420, n, c); }
    if (a218[c]) { ruc_snow_layer_thresholds(a137, a135, a421, a422, n, c); }
    if (true) { a423[c] = 0; }
    if (a218[c]) { ruc_snow_temperature_step(a408, a405, a419, a389, a413, a134, a135, a92, a371, a410, a421, a422, a137, a138, a38, a16, a97, a39, a14, a40, a145, a361, a44, a45, a42, a101, a412, a411, a420, a423, a150, a151, a154, a72, a182, a4, delt, a69, as_type<float>(0x4a2d08e0), as_type<float>(0x4a7f4f60), a443, a424, a425, a426, a427, a428, a429, a430, a431, a432, a433, a434, a435, a436, a437, a438, a439, a440, a441, a442, a444, n, c, forms); }
    if (a218[c]) { ruc_snow_soil_prepare_moisture(a40, a428, a44, a101, a412, a414, a419, a72, a108, a448, a445, a446, a447, n, c); }
    if (a218[c]) { ruc_soil_phase_partition(a388, a443, a390, a391, a50, a51, a54, a55, 0, a449, a450, a451, a452, a453, a454, a455, a456, n, c); }
    if (true) { a457[c] = 0; }
    if (true) { a458[c] = 1; }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a406[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 0, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a407[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 1, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a448[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 2, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a450[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 3, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a388[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 4, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a449[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 5, memory_order_relaxed); } } }
    if (a218[c]) { if (!isfinite(a428[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 6, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a427[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 7, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a429[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 8, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a41[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 9, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 10, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a447[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 11, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 12, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a457[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 13, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a457[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 14, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a435[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 15, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a101[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 16, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a371[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 17, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a458[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 18, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 19, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 20, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a52[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 21, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a56[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 22, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a414[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 23, memory_order_relaxed); } }
    if (a218[c]) { a459[c] = (a50[c] <= as_type<float>(0x00000000)); }
    if (a218[c]) { if (a459[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 24, memory_order_relaxed); } }
    if (a218[c]) { a460[c] = (a52[c] <= a51[c]); }
    if (a218[c]) { if (a460[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 25, memory_order_relaxed); } }
    if (a218[c]) { a461[c] = (a56[c] < as_type<float>(0x00000000)); }
    if (a218[c]) { if (a461[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 26, memory_order_relaxed); } }
    if (a218[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a218[c]) { ruc_soil_moisture_step(a406, a407, a448, a450, a388, a449, a428, a427, a429, a41, a40, a447, a44, a457, a457, a435, a101, a371, a458, a50, a51, a52, a56, a414, delt, a462, a463, a464, a465, a466, a467, a468, n, c); }
    if (a218[c]) { ruc_snow_soil_finalize(a450, a443, a392, a462, a393, a391, a432, a435, a160, a437, a436, a442, a424, a428, a426, a434, a411, a445, a446, a414, a148, a45, a42, a14, a39, a44, a40, a101, a361, delt, as_type<float>(0x4a2d08e0), a469, a470, a471, a472, a473, a474, a475, a476, a477, a478, a479, a480, n, c); }
    if (a218[c]) { a481[c] = a16[c]; }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a462[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 27, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a443[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 28, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a390[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 29, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a391[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 30, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a450[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 16, 1u << 31, memory_order_relaxed); } } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a463[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 0, memory_order_relaxed); } } }
    if (a218[c]) { if (!isfinite(a471[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 1, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a472[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 2, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a424[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 3, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a425[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 4, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a469[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 5, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a427[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 6, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a428[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 7, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a429[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 8, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a431[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 9, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a432[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 10, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a433[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 11, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a440[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 12, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a441[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 13, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a439[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 14, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a435[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 15, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a436[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 16, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a437[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 17, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a470[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 18, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a474[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 19, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a475[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 20, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a473[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 21, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a476[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 22, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a477[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 23, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a478[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 24, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a437[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 25, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a479[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 26, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a481[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 27, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a480[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 28, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a465[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 29, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a466[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 30, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a464[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 17, 1u << 31, memory_order_relaxed); } }
    if (a218[c]) { if (!isfinite(a467[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 0, memory_order_relaxed); } }
    if (a218[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a128[k * n + c] = a462[k * n + c]; } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a129[k * n + c] = a443[k * n + c]; } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a130[k * n + c] = a390[k * n + c]; } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a131[k * n + c] = a391[k * n + c]; } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a132[k * n + c] = a450[k * n + c]; } }
    if (a218[c]) { for (int k = 0; k < RUC_NZS; ++k) { a133[k * n + c] = a463[k * n + c]; } }
    if (a218[c]) { a148[c] = a471[c]; }
    if (a218[c]) { a153[c] = a472[c]; }
    if (a218[c]) { a150[c] = a424[c]; }
    if (a218[c]) { a151[c] = a425[c]; }
    if (a218[c]) { a152[c] = a469[c]; }
    if (a218[c]) { a154[c] = a427[c]; }
    if (a218[c]) { a155[c] = a428[c]; }
    if (a218[c]) { a156[c] = a429[c]; }
    if (a218[c]) { a134[c] = a431[c]; }
    if (a218[c]) { a135[c] = a432[c]; }
    if (a218[c]) { a137[c] = a433[c]; }
    if (a218[c]) { a179[c] = a440[c]; }
    if (a218[c]) { a180[c] = a441[c]; }
    if (a218[c]) { a178[c] = a439[c]; }
    if (a218[c]) { a157[c] = a435[c]; }
    if (a218[c]) { a158[c] = a436[c]; }
    if (a218[c]) { a159[c] = a437[c]; }
    if (a218[c]) { a160[c] = a470[c]; }
    if (a218[c]) { a163[c] = a474[c]; }
    if (a218[c]) { a164[c] = a475[c]; }
    if (a218[c]) { a165[c] = a473[c]; }
    if (a218[c]) { a166[c] = a476[c]; }
    if (a218[c]) { a167[c] = a477[c]; }
    if (a218[c]) { a168[c] = a478[c]; }
    if (a218[c]) { a169[c] = a437[c]; }
    if (a218[c]) { a170[c] = a479[c]; }
    if (a218[c]) { a172[c] = a481[c]; }
    if (a218[c]) { a173[c] = a480[c]; }
    if (a218[c]) { a174[c] = a465[c]; }
    if (a218[c]) { a175[c] = a466[c]; }
    if (a218[c]) { a146[c] = a464[c]; }
    if (a218[c]) { a176[c] = a467[c]; }
    if (a218[c]) { a182[c] = a444[c]; }
    if (a219[c]) { a482[c] = (a185[c] ? as_type<float>(0x3f800000) : a136[c]); }
    if (a219[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a85[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 1, memory_order_relaxed); } } }
    if (a219[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a86[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 2, memory_order_relaxed); } } }
    if (a219[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a129[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 3, memory_order_relaxed); } } }
    if (a219[c]) { if (!isfinite(a38[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 4, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a138[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 5, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a16[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 6, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a97[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 7, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a92[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 8, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a135[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 9, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a134[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 10, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a482[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 11, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a137[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 12, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a39[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 13, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 14, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a145[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 15, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a361[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 16, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 17, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a45[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 18, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a42[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 19, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a147[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 20, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a149[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 21, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 22, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a150[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 23, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a151[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 24, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a152[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 25, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a154[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 26, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a155[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 27, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a160[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 28, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a169[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 29, memory_order_relaxed); } }
    if (a219[c]) { a483[c] = (a86[0 * n + c] <= as_type<float>(0x00000000)); }
    if (a219[c]) { if (a483[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 30, memory_order_relaxed); } }
    if (a219[c]) { a484[c] = (a85[0 * n + c] <= as_type<float>(0x00000000)); }
    if (a219[c]) { if (a484[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 18, 1u << 31, memory_order_relaxed); } }
    if (a219[c]) { a485[c] = (a39[c] <= as_type<float>(0x00000000)); }
    if (a219[c]) { if (a485[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 0, memory_order_relaxed); } }
    if (a219[c]) { a486[c] = (a42[c] <= as_type<float>(0x00000000)); }
    if (a219[c]) { if (a486[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 1, memory_order_relaxed); } }
    if (a219[c]) { a487[c] = (a137[c] <= as_type<float>(0x00000000)); }
    if (a219[c]) { if (a487[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 2, memory_order_relaxed); } }
    if (a219[c]) { a488[c] = (a134[c] < as_type<float>(0x00000000)); }
    if (a219[c]) { if (a488[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 3, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a69[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 4, memory_order_relaxed); } }
    if (a219[c]) { a518[c] = (a69[c] < as_type<float>(0x00000000)); }
    if (a219[c]) { if (a518[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 5, memory_order_relaxed); } }
    if (a219[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a219[c]) { ruc_snow_sea_ice_step(a85, a86, a129, a38, a138, a16, a97, a92, a135, a134, a482, a137, a39, a40, a145, a361, a44, a45, a42, a147, a149, a14, a150, a151, a152, a154, a155, a160, a169, a182, a4, delt, a69, as_type<float>(0x4a7f4f60), as_type<float>(0x4a189680), 0, a489, a490, a491, a492, a493, a494, a495, a496, a497, a498, a499, a500, a501, a502, a503, a504, a505, a506, a507, a508, a509, a510, a511, a512, a513, a514, a515, a516, a517, n, c); }
    if (a219[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a489[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 6, memory_order_relaxed); } } }
    if (a219[c]) { if (!isfinite(a491[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 7, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a492[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 8, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a493[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 9, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a494[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 10, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a495[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 11, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a496[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 12, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a497[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 13, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a498[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 14, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a499[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 15, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a500[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 16, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a501[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 17, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a502[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 18, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a503[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 19, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a504[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 20, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a505[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 21, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a506[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 22, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a507[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 23, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a508[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 24, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a509[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 25, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a510[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 26, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a511[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 27, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a512[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 28, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a513[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 29, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a514[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 30, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a515[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 19, 1u << 31, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a516[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 0, memory_order_relaxed); } }
    if (a219[c]) { if (!isfinite(a517[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 1, memory_order_relaxed); } }
    if (a219[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a219[c]) { for (int k = 0; k < RUC_NZS; ++k) { a129[k * n + c] = a489[k * n + c]; } }
    if (a219[c]) { a179[c] = a491[c]; }
    if (a219[c]) { a180[c] = a492[c]; }
    if (a219[c]) { a178[c] = a493[c]; }
    if (a219[c]) { a153[c] = a494[c]; }
    if (a219[c]) { a150[c] = a495[c]; }
    if (a219[c]) { a151[c] = a496[c]; }
    if (a219[c]) { a152[c] = a497[c]; }
    if (a219[c]) { a154[c] = a498[c]; }
    if (a219[c]) { a155[c] = a499[c]; }
    if (a219[c]) { a156[c] = a500[c]; }
    if (a219[c]) { a157[c] = a501[c]; }
    if (a219[c]) { a158[c] = a502[c]; }
    if (a219[c]) { a159[c] = a503[c]; }
    if (a219[c]) { a160[c] = a504[c]; }
    if (a219[c]) { a166[c] = a505[c]; }
    if (a219[c]) { a167[c] = a506[c]; }
    if (a219[c]) { a168[c] = a507[c]; }
    if (a219[c]) { a169[c] = a508[c]; }
    if (a219[c]) { a170[c] = a509[c]; }
    if (a219[c]) { a172[c] = a510[c]; }
    if (a219[c]) { a173[c] = a511[c]; }
    if (a219[c]) { a134[c] = a512[c]; }
    if (a219[c]) { a135[c] = a513[c]; }
    if (a219[c]) { a137[c] = a514[c]; }
    if (a219[c]) { a145[c] = a515[c]; }
    if (a219[c]) { a147[c] = a516[c]; }
    if (a219[c]) { a149[c] = a517[c]; }
    if (a219[c]) { a182[c] = a490[c]; }
    if (a219[c]) { a519[c] = ruc_fmul_rn(a166[c], as_type<float>(0x3a83126f)); }
    if (a219[c]) { a163[c] = a519[c]; }
    if (a219[c]) { a164[c] = as_type<float>(0x00000000); }
    if (a219[c]) { a165[c] = as_type<float>(0x00000000); }
    if (a219[c]) { a174[c] = a157[c]; }
    if (a219[c]) { a175[c] = as_type<float>(0x00000000); }
    if (a219[c]) { a146[c] = as_type<float>(0x3f800000); }
    if (a219[c]) { a176[c] = as_type<float>(0x00000000); }
    if (a219[c]) { a148[c] = as_type<float>(0x00000000); }
    if (a219[c]) { for (int k = 0; k < RUC_NZS; ++k) { a128[k * n + c] = as_type<float>(0x3f800000); } }
    if (a219[c]) { for (int k = 0; k < RUC_NZS; ++k) { a133[k * n + c] = as_type<float>(0x00000000); } }
    if (a219[c]) { for (int k = 0; k < RUC_NZS; ++k) { a132[k * n + c] = as_type<float>(0x3f800000); } }
    if (a219[c]) { for (int k = 0; k < RUC_NZS; ++k) { a130[k * n + c] = as_type<float>(0x3f800000); } }
    if (a219[c]) { for (int k = 0; k < RUC_NZS; ++k) { a131[k * n + c] = as_type<float>(0x00000000); } }
    if (true) { a520[c] = (a135[c] == as_type<float>(0x00000000)); }
    if (true) { a521[c] = (a184[c] && a520[c]); }
    if (true) { a522[c] = !(GPUWM_RUC_SNOW_V461); }
    if (true) { a523[c] = (a521[c] && a522[c]); }
    if (true) { a524[c] = (a523[c] ? a24[c] : a147[c]); }
    if (true) { a525[c] = (a523[c] ? a70[c] : a181[c]); }
    r.raw[p.alive_base + uint(c)] = true;
}
kernel void ruc_sfctmp_stage2(
    constant RucSfctmpResources& r [[buffer(0)]],
    constant RucSfctmpParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int n = p.n;
    const float delt = p.delt;
    const float c1sn = p.c1sn;
    const float c2sn = p.c2sn;
    const float rsmax = p.rsmax;
    const int isice = p.isice;
    const int urban = p.urban;
    const int ncategory = p.ncategory;
    const int limit = p.limit;
    const uint forms = uint(p.forms);
    const int c = int(tid);
    
    if (c >= n) return;
    if (!r.raw[p.alive_base + uint(c)]) return;
    r.raw[p.alive_base + uint(c)] = false;
    if (r.raw[p.run_base + uint(c)] == 0u) { ruc_sfctmp_zero_outputs(r, p, n, c); return; }
    bool failed = false;
    if (a214[c]) { a526[c] = ruc_fsub_rn(as_type<float>(0x3f800000), a136[c]); }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a527[k * n + c] = ruc_fmul_rn(a186[k * n + c], a526[c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a528[k * n + c] = ruc_fmul_rn(a128[k * n + c], a136[c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a529[k * n + c] = ruc_fadd_rn(a527[k * n + c], a528[k * n + c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a128[k * n + c] = a529[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a530[k * n + c] = ruc_fmul_rn(a188[k * n + c], a526[c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a531[k * n + c] = ruc_fmul_rn(a130[k * n + c], a136[c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a532[k * n + c] = ruc_fadd_rn(a530[k * n + c], a531[k * n + c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a130[k * n + c] = a532[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a533[k * n + c] = ruc_fmul_rn(a190[k * n + c], a526[c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a534[k * n + c] = ruc_fmul_rn(a132[k * n + c], a136[c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a535[k * n + c] = ruc_fadd_rn(a533[k * n + c], a534[k * n + c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a132[k * n + c] = a535[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a536[k * n + c] = ruc_fmul_rn(a191[k * n + c], a526[c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a537[k * n + c] = ruc_fmul_rn(a133[k * n + c], a136[c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a538[k * n + c] = ruc_fadd_rn(a536[k * n + c], a537[k * n + c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a133[k * n + c] = a538[k * n + c]; } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a539[k * n + c] = ruc_fmul_rn(a187[k * n + c], a526[c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a540[k * n + c] = ruc_fmul_rn(a129[k * n + c], a136[c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a541[k * n + c] = ruc_fadd_rn(a539[k * n + c], a540[k * n + c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a129[k * n + c] = a541[k * n + c]; } }
    if (a214[c]) { a542[c] = (a136[c] > as_type<float>(0x3f000000)); }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a543[k * n + c] = (a542[c] ? a131[k * n + c] : a189[k * n + c]); } }
    if (a214[c]) { for (int k = 0; k < RUC_NZS; ++k) { a131[k * n + c] = a543[k * n + c]; } }
    if (a214[c]) { a544[c] = ruc_fmul_rn(a192[c], a526[c]); }
    if (a214[c]) { a545[c] = ruc_fmul_rn(a153[c], a136[c]); }
    if (a214[c]) { a546[c] = ruc_fadd_rn(a544[c], a545[c]); }
    if (a214[c]) { a153[c] = a546[c]; }
    if (a214[c]) { a547[c] = ruc_fmul_rn(a193[c], a526[c]); }
    if (a214[c]) { a548[c] = ruc_fmul_rn(a150[c], a136[c]); }
    if (a214[c]) { a549[c] = ruc_fadd_rn(a547[c], a548[c]); }
    if (a214[c]) { a150[c] = a549[c]; }
    if (a214[c]) { a550[c] = ruc_fmul_rn(a194[c], a526[c]); }
    if (a214[c]) { a551[c] = ruc_fmul_rn(a154[c], a136[c]); }
    if (a214[c]) { a552[c] = ruc_fadd_rn(a550[c], a551[c]); }
    if (a214[c]) { a154[c] = a552[c]; }
    if (a214[c]) { a553[c] = ruc_fmul_rn(a195[c], a526[c]); }
    if (a214[c]) { a554[c] = ruc_fmul_rn(a155[c], a136[c]); }
    if (a214[c]) { a555[c] = ruc_fadd_rn(a553[c], a554[c]); }
    if (a214[c]) { a155[c] = a555[c]; }
    if (a214[c]) { a556[c] = ruc_fmul_rn(a196[c], a526[c]); }
    if (a214[c]) { a557[c] = ruc_fmul_rn(a156[c], a136[c]); }
    if (a214[c]) { a558[c] = ruc_fadd_rn(a556[c], a557[c]); }
    if (a214[c]) { a156[c] = a558[c]; }
    if (a214[c]) { a559[c] = ruc_fmul_rn(a197[c], a526[c]); }
    if (a214[c]) { a560[c] = ruc_fmul_rn(a163[c], a136[c]); }
    if (a214[c]) { a561[c] = ruc_fadd_rn(a559[c], a560[c]); }
    if (a214[c]) { a163[c] = a561[c]; }
    if (a214[c]) { a562[c] = ruc_fmul_rn(a198[c], a526[c]); }
    if (a214[c]) { a563[c] = ruc_fmul_rn(a164[c], a136[c]); }
    if (a214[c]) { a564[c] = ruc_fadd_rn(a562[c], a563[c]); }
    if (a214[c]) { a164[c] = a564[c]; }
    if (a214[c]) { a565[c] = ruc_fmul_rn(a211[c], a526[c]); }
    if (a214[c]) { a566[c] = ruc_fmul_rn(a148[c], a136[c]); }
    if (a214[c]) { a567[c] = ruc_fadd_rn(a565[c], a566[c]); }
    if (a214[c]) { a148[c] = a567[c]; }
    if (a214[c]) { a568[c] = ruc_fmul_rn(a199[c], a526[c]); }
    if (a214[c]) { a569[c] = ruc_fmul_rn(a165[c], a136[c]); }
    if (a214[c]) { a570[c] = ruc_fadd_rn(a568[c], a569[c]); }
    if (a214[c]) { a165[c] = a570[c]; }
    if (a214[c]) { a571[c] = ruc_fmul_rn(a200[c], a526[c]); }
    if (a214[c]) { a572[c] = ruc_fmul_rn(a166[c], a136[c]); }
    if (a214[c]) { a573[c] = ruc_fadd_rn(a571[c], a572[c]); }
    if (a214[c]) { a166[c] = a573[c]; }
    if (a214[c]) { a574[c] = ruc_fmul_rn(a201[c], a526[c]); }
    if (a214[c]) { a575[c] = ruc_fmul_rn(a167[c], a136[c]); }
    if (a214[c]) { a576[c] = ruc_fadd_rn(a574[c], a575[c]); }
    if (a214[c]) { a167[c] = a576[c]; }
    if (a214[c]) { a577[c] = ruc_fmul_rn(a202[c], a526[c]); }
    if (a214[c]) { a578[c] = ruc_fmul_rn(a168[c], a136[c]); }
    if (a214[c]) { a579[c] = ruc_fadd_rn(a577[c], a578[c]); }
    if (a214[c]) { a168[c] = a579[c]; }
    if (a214[c]) { a580[c] = ruc_fmul_rn(a203[c], a526[c]); }
    if (a214[c]) { a581[c] = ruc_fmul_rn(a169[c], a136[c]); }
    if (a214[c]) { a582[c] = ruc_fadd_rn(a580[c], a581[c]); }
    if (a214[c]) { a169[c] = a582[c]; }
    if (a214[c]) { a583[c] = ruc_fmul_rn(a205[c], a526[c]); }
    if (a214[c]) { a584[c] = ruc_fmul_rn(a172[c], a136[c]); }
    if (a214[c]) { a585[c] = ruc_fadd_rn(a583[c], a584[c]); }
    if (a214[c]) { a172[c] = a585[c]; }
    if (a214[c]) { a586[c] = ruc_fmul_rn(a206[c], a526[c]); }
    if (a214[c]) { a587[c] = ruc_fmul_rn(a173[c], a136[c]); }
    if (a214[c]) { a588[c] = ruc_fadd_rn(a586[c], a587[c]); }
    if (a214[c]) { a173[c] = a588[c]; }
    if (a214[c]) { a589[c] = ruc_fmul_rn(a204[c], a526[c]); }
    if (a214[c]) { a171[c] = a589[c]; }
    if (a214[c]) { a590[c] = ruc_fmul_rn(a170[c], a136[c]); }
    if (a214[c]) { a170[c] = a590[c]; }
    if (a214[c]) { a591[c] = ruc_fmul_rn(a116[c], a524[c]); }
    if (a214[c]) { a592[c] = ruc_fsub_rn(a524[c], a24[c]); }
    if (a214[c]) { a593[c] = ruc_fmul_rn(a592[c], a136[c]); }
    if (a214[c]) { a594[c] = ruc_fadd_rn(a24[c], a593[c]); }
    if (a214[c]) { a595[c] = ruc_fused_min(a594[c], a524[c]); }
    if (a214[c]) { a596[c] = ruc_fused_max(a591[c], a595[c]); }
    if (a214[c]) { a524[c] = a596[c]; }
    if (a214[c]) { a597[c] = ruc_fmul_rn(a116[c], a114[c]); }
    if (a214[c]) { a598[c] = ruc_fsub_rn(a114[c], a115[c]); }
    if (a214[c]) { a599[c] = ruc_fmul_rn(a598[c], a136[c]); }
    if (a214[c]) { a600[c] = ruc_fadd_rn(a115[c], a599[c]); }
    if (a214[c]) { a601[c] = ruc_fused_min(a600[c], a114[c]); }
    if (a214[c]) { a602[c] = ruc_fused_max(a597[c], a601[c]); }
    if (a214[c]) { a145[c] = a602[c]; }
    if (a214[c]) { a603[c] = ruc_fmul_rn(a207[c], a526[c]); }
    if (a214[c]) { a604[c] = ruc_fmul_rn(a174[c], a136[c]); }
    if (a214[c]) { a605[c] = ruc_fadd_rn(a603[c], a604[c]); }
    if (a214[c]) { a174[c] = a605[c]; }
    if (a214[c]) { a606[c] = ruc_fmul_rn(a208[c], a526[c]); }
    if (a214[c]) { a607[c] = ruc_fmul_rn(a175[c], a136[c]); }
    if (a214[c]) { a608[c] = ruc_fadd_rn(a606[c], a607[c]); }
    if (a214[c]) { a175[c] = a608[c]; }
    if (a214[c]) { a609[c] = ruc_fmul_rn(a157[c], a136[c]); }
    if (a214[c]) { a610[c] = (GPUWM_RUC_SNOW_V461 ? a157[c] : a609[c]); }
    if (a214[c]) { a157[c] = a610[c]; }
    if (a214[c]) { a611[c] = ruc_fmul_rn(a158[c], a136[c]); }
    if (a214[c]) { a612[c] = (GPUWM_RUC_SNOW_V461 ? a158[c] : a611[c]); }
    if (a214[c]) { a158[c] = a612[c]; }
    if (a214[c]) { a613[c] = ruc_fmul_rn(a159[c], a136[c]); }
    if (a214[c]) { a614[c] = (GPUWM_RUC_SNOW_V461 ? a159[c] : a613[c]); }
    if (a214[c]) { a159[c] = a614[c]; }
    if (a214[c]) { a615[c] = ruc_fmul_rn(a160[c], a136[c]); }
    if (a214[c]) { a616[c] = (GPUWM_RUC_SNOW_V461 ? a160[c] : a615[c]); }
    if (a214[c]) { a160[c] = a616[c]; }
    if (a214[c]) { a617[c] = ruc_fmul_rn(a209[c], a526[c]); }
    if (a214[c]) { a618[c] = ruc_fadd_rn(a617[c], a136[c]); }
    if (a214[c]) { a146[c] = a618[c]; }
    if (a214[c]) { a619[c] = ruc_fmul_rn(a210[c], a526[c]); }
    if (a214[c]) { a620[c] = ruc_fmul_rn(a176[c], a136[c]); }
    if (a214[c]) { a621[c] = ruc_fadd_rn(a619[c], a620[c]); }
    if (a214[c]) { a176[c] = a621[c]; }
    if (a216[c]) { a622[c] = ruc_fsub_rn(as_type<float>(0x3f800000), a136[c]); }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { a623[k * n + c] = ruc_fmul_rn(a187[k * n + c], a622[c]); } }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { a624[k * n + c] = ruc_fmul_rn(a129[k * n + c], a136[c]); } }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { a625[k * n + c] = ruc_fadd_rn(a623[k * n + c], a624[k * n + c]); } }
    if (a216[c]) { for (int k = 0; k < RUC_NZS; ++k) { a129[k * n + c] = a625[k * n + c]; } }
    if (a216[c]) { a626[c] = ruc_fmul_rn(a192[c], a622[c]); }
    if (a216[c]) { a627[c] = ruc_fmul_rn(a153[c], a136[c]); }
    if (a216[c]) { a628[c] = ruc_fadd_rn(a626[c], a627[c]); }
    if (a216[c]) { a153[c] = a628[c]; }
    if (a216[c]) { a629[c] = ruc_fmul_rn(a193[c], a622[c]); }
    if (a216[c]) { a630[c] = ruc_fmul_rn(a150[c], a136[c]); }
    if (a216[c]) { a631[c] = ruc_fadd_rn(a629[c], a630[c]); }
    if (a216[c]) { a150[c] = a631[c]; }
    if (a216[c]) { a632[c] = ruc_fmul_rn(a194[c], a622[c]); }
    if (a216[c]) { a633[c] = ruc_fmul_rn(a154[c], a136[c]); }
    if (a216[c]) { a634[c] = ruc_fadd_rn(a632[c], a633[c]); }
    if (a216[c]) { a154[c] = a634[c]; }
    if (a216[c]) { a635[c] = ruc_fmul_rn(a195[c], a622[c]); }
    if (a216[c]) { a636[c] = ruc_fmul_rn(a155[c], a136[c]); }
    if (a216[c]) { a637[c] = ruc_fadd_rn(a635[c], a636[c]); }
    if (a216[c]) { a155[c] = a637[c]; }
    if (a216[c]) { a638[c] = ruc_fmul_rn(a196[c], a622[c]); }
    if (a216[c]) { a639[c] = ruc_fmul_rn(a156[c], a136[c]); }
    if (a216[c]) { a640[c] = ruc_fadd_rn(a638[c], a639[c]); }
    if (a216[c]) { a156[c] = a640[c]; }
    if (a216[c]) { a641[c] = ruc_fmul_rn(a200[c], a622[c]); }
    if (a216[c]) { a642[c] = ruc_fmul_rn(a166[c], a136[c]); }
    if (a216[c]) { a643[c] = ruc_fadd_rn(a641[c], a642[c]); }
    if (a216[c]) { a166[c] = a643[c]; }
    if (a216[c]) { a644[c] = ruc_fmul_rn(a201[c], a622[c]); }
    if (a216[c]) { a645[c] = ruc_fmul_rn(a167[c], a136[c]); }
    if (a216[c]) { a646[c] = ruc_fadd_rn(a644[c], a645[c]); }
    if (a216[c]) { a167[c] = a646[c]; }
    if (a216[c]) { a647[c] = ruc_fmul_rn(a202[c], a622[c]); }
    if (a216[c]) { a648[c] = ruc_fmul_rn(a168[c], a136[c]); }
    if (a216[c]) { a649[c] = ruc_fadd_rn(a647[c], a648[c]); }
    if (a216[c]) { a168[c] = a649[c]; }
    if (a216[c]) { a650[c] = ruc_fmul_rn(a203[c], a622[c]); }
    if (a216[c]) { a651[c] = ruc_fmul_rn(a169[c], a136[c]); }
    if (a216[c]) { a652[c] = ruc_fadd_rn(a650[c], a651[c]); }
    if (a216[c]) { a169[c] = a652[c]; }
    if (a216[c]) { a170[c] = a166[c]; }
    if (a216[c]) { a653[c] = ruc_fmul_rn(a205[c], a622[c]); }
    if (a216[c]) { a654[c] = ruc_fmul_rn(a172[c], a136[c]); }
    if (a216[c]) { a655[c] = ruc_fadd_rn(a653[c], a654[c]); }
    if (a216[c]) { a172[c] = a655[c]; }
    if (a216[c]) { a656[c] = ruc_fmul_rn(a206[c], a622[c]); }
    if (a216[c]) { a657[c] = ruc_fmul_rn(a173[c], a136[c]); }
    if (a216[c]) { a658[c] = ruc_fadd_rn(a656[c], a657[c]); }
    if (a216[c]) { a173[c] = a658[c]; }
    if (a216[c]) { a659[c] = ruc_fmul_rn(a116[c], a524[c]); }
    if (a216[c]) { a660[c] = ruc_fsub_rn(a524[c], a24[c]); }
    if (a216[c]) { a661[c] = ruc_fmul_rn(a660[c], a136[c]); }
    if (a216[c]) { a662[c] = ruc_fadd_rn(a112[c], a661[c]); }
    if (a216[c]) { a663[c] = ruc_fused_min(a662[c], a524[c]); }
    if (a216[c]) { a664[c] = ruc_fused_max(a659[c], a663[c]); }
    if (a216[c]) { a524[c] = a664[c]; }
    if (a216[c]) { a665[c] = ruc_fmul_rn(a116[c], a114[c]); }
    if (a216[c]) { a666[c] = ruc_fsub_rn(a114[c], a115[c]); }
    if (a216[c]) { a667[c] = ruc_fmul_rn(a666[c], a136[c]); }
    if (a216[c]) { a668[c] = ruc_fadd_rn(a115[c], a667[c]); }
    if (a216[c]) { a669[c] = ruc_fused_min(a668[c], a114[c]); }
    if (a216[c]) { a670[c] = ruc_fused_max(a665[c], a669[c]); }
    if (a216[c]) { a145[c] = a670[c]; }
    if (a216[c]) { a671[c] = ruc_fmul_rn(a207[c], a622[c]); }
    if (a216[c]) { a672[c] = ruc_fmul_rn(a174[c], a136[c]); }
    if (a216[c]) { a673[c] = ruc_fadd_rn(a671[c], a672[c]); }
    if (a216[c]) { a174[c] = a673[c]; }
    if (a216[c]) { a674[c] = ruc_fmul_rn(a208[c], a622[c]); }
    if (a216[c]) { a675[c] = ruc_fmul_rn(a175[c], a136[c]); }
    if (a216[c]) { a676[c] = ruc_fadd_rn(a674[c], a675[c]); }
    if (a216[c]) { a175[c] = a676[c]; }
    if (a216[c]) { a677[c] = ruc_fmul_rn(a157[c], a136[c]); }
    if (a216[c]) { a678[c] = (GPUWM_RUC_SNOW_V461 ? a157[c] : a677[c]); }
    if (a216[c]) { a157[c] = a678[c]; }
    if (a216[c]) { a679[c] = ruc_fmul_rn(a158[c], a136[c]); }
    if (a216[c]) { a680[c] = (GPUWM_RUC_SNOW_V461 ? a158[c] : a679[c]); }
    if (a216[c]) { a158[c] = a680[c]; }
    if (a216[c]) { a681[c] = ruc_fmul_rn(a159[c], a136[c]); }
    if (a216[c]) { a682[c] = (GPUWM_RUC_SNOW_V461 ? a159[c] : a681[c]); }
    if (a216[c]) { a159[c] = a682[c]; }
    if (a216[c]) { a683[c] = ruc_fmul_rn(a160[c], a136[c]); }
    if (a216[c]) { a684[c] = (GPUWM_RUC_SNOW_V461 ? a160[c] : a683[c]); }
    if (a216[c]) { a160[c] = a684[c]; }
    if (true) { a685[c] = (a135[c] == as_type<float>(0x00000000)); }
    if (true) { a686[c] = (a184[c] && a685[c]); }
    if (true) { a687[c] = (a686[c] && GPUWM_RUC_SNOW_V461); }
    if (a687[c]) { a524[c] = a24[c]; }
    if (a687[c]) { a525[c] = a70[c]; }
    if (true) { a688[c] = !(a686[c]); }
    if (true) { a689[c] = (a184[c] && a688[c]); }
    if (true) { a690[c] = (a689[c] && GPUWM_RUC_SNOW_V461); }
    if (a690[c]) { a691[c] = ruc_fmul_rn(as_type<float>(0x40000000), a87[c]); }
    if (a690[c]) { a692[c] = ruc_fdiv_rn(a135[c], a691[c]); }
    if (a690[c]) { a693[c] = ruc_fused_min(as_type<float>(0x3f800000), a692[c]); }
    if (a690[c]) { a694[c] = ruc_fused_min(as_type<float>(0x3e4ccccd), a149[c]); }
    if (a690[c]) { a695[c] = ruc_fmul_rn(as_type<float>(0x40200000), a694[c]); }
    if (a690[c]) { a696[c] = ruc_fdiv_rn(a137[c], a138[c]); }
    if (a690[c]) { a697[c] = ruc_fmul_rn(a695[c], a696[c]); }
    if (a690[c]) { a698[c] = ruc_fdiv_rn(a135[c], a697[c]); }
    if (a690[c]) { ruc_tanhf_glibc_array(a698, a699, n, c); }
    if (a690[c]) { a700[c] = ruc_fadd_rn(a693[c], a699[c]); }
    if (a690[c]) { a701[c] = ruc_fmul_rn(as_type<float>(0x3f000000), a700[c]); }
    if (a690[c]) { a136[c] = a701[c]; }
    if (true) { a702[c] = (a70[c] == urban); }
    if (true) { a703[c] = (a184[c] && a702[c]); }
    if (true) { a704[c] = (a703[c] && GPUWM_RUC_SNOW_V461); }
    if (a704[c]) { a705[c] = ruc_fused_min(as_type<float>(0x3f400000), a136[c]); }
    if (a704[c]) { a136[c] = a705[c]; }
    if (a184[c]) { a706[c] = ruc_fdiv_rn(as_type<float>(0x447a0000), a138[c]); }
    if (a184[c]) { a707[c] = ruc_fmul_rn(a706[c], a157[c]); }
    if (a184[c]) { a708[c] = ruc_fmul_rn(a707[c], delt); }
    if (a184[c]) { a709[c] = ruc_fmul_rn(a708[c], a93[c]); }
    if (a184[c]) { a710[c] = ruc_fsub_rn(a92[c], a709[c]); }
    if (a184[c]) { a711[c] = ruc_fused_max(as_type<float>(0x00000000), a710[c]); }
    if (a184[c]) { a712[c] = (GPUWM_RUC_SNOW_V461 ? a92[c] : a711[c]); }
    if (a184[c]) { a713[c] = ruc_fmul_rn(a712[c], as_type<float>(0x447a0000)); }
    if (a184[c]) { a714[c] = ruc_fadd_rn(a161[c], a713[c]); }
    if (a184[c]) { a161[c] = a714[c]; }
    if (true) { a715[c] = 0; }
    if (true) { a716[c] = 0; }
    if (a220[c]) { a180[c] = as_type<float>(0x00000000); }
    if (a220[c]) { a179[c] = as_type<float>(0x00000000); }
    if (a220[c]) { a157[c] = as_type<float>(0x00000000); }
    if (a220[c]) { a717[c] = ruc_fmul_rn(as_type<float>(0x33738beb), a150[c]); }
    if (a220[c]) { a718[c] = ruc_fmul_rn(a717[c], a150[c]); }
    if (a220[c]) { a719[c] = ruc_fmul_rn(a718[c], a150[c]); }
    if (a220[c]) { a720[c] = ruc_fmul_rn(a719[c], a150[c]); }
    if (a220[c]) { a721[c] = ruc_fsub_rn(a43[c], a720[c]); }
    if (a220[c]) { a722[c] = ruc_fmul_rn(a145[c], a721[c]); }
    if (a220[c]) { a723[c] = ruc_fadd_rn(a13[c], a722[c]); }
    if (a220[c]) { a715[c] = a723[c]; }
    if (a220[c]) { a716[c] = a722[c]; }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a128[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 2, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a129[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 3, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a130[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 4, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a131[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 5, memory_order_relaxed); } } }
    if (a223[c]) { if ((a72[c] < 1 || a72[c] >= RUC_NZS)) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 6, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (a223[c]) { if (!isfinite(a16[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 7, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a97[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 8, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a39[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 9, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 10, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a41[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 11, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a43[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 12, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a13[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 13, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a111[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 14, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a145[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 15, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a715[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 16, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 17, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a45[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 18, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a46[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 19, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a148[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 20, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a102[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 21, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a108[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 22, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a42[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 23, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a101[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 24, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a19[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 25, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a48[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 26, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a49[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 27, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 28, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 29, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a52[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 30, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a53[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 20, 1u << 31, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a54[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 0, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a55[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 1, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a56[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 2, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a20[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 3, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a57[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 4, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 5, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a150[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 6, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a154[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 7, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a155[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 8, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a156[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 9, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a47[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 10, memory_order_relaxed); } }
    if (a223[c]) { a724[c] = (a50[c] <= as_type<float>(0x00000000)); }
    if (a223[c]) { if (a724[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 11, memory_order_relaxed); } }
    if (a223[c]) { a725[c] = (a54[c] >= as_type<float>(0x00000000)); }
    if (a223[c]) { if (a725[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 12, memory_order_relaxed); } }
    if (a223[c]) { a726[c] = (a55[c] <= as_type<float>(0x00000000)); }
    if (a223[c]) { if (a726[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 13, memory_order_relaxed); } }
    if (a223[c]) { a727[c] = (a20[c] <= as_type<float>(0x00000000)); }
    if (a223[c]) { if (a727[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 14, memory_order_relaxed); } }
    if (a223[c]) { a728[c] = (a42[c] <= as_type<float>(0x00000000)); }
    if (a223[c]) { a729[c] = (a39[c] <= as_type<float>(0x00000000)); }
    if (a223[c]) { a730[c] = (a728[c] || a729[c]); }
    if (a223[c]) { if (a730[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 15, memory_order_relaxed); } }
    if (a223[c]) { a731[c] = (a47[c] < as_type<float>(0x00000000)); }
    if (a223[c]) { a732[c] = (a47[c] > as_type<float>(0x3f800000)); }
    if (a223[c]) { a733[c] = (a731[c] || a732[c]); }
    if (a223[c]) { if (a733[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 16, memory_order_relaxed); } }
    if (a223[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a734[k * n + c] = a128[k * n + c]; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a735[k * n + c] = a129[k * n + c]; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a736[k * n + c] = a130[k * n + c]; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a737[k * n + c] = a131[k * n + c]; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a738[k * n + c] = a735[k * n + c]; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a739[k * n + c] = a734[k * n + c]; } }
    if (a223[c]) { ruc_soil_phase_partition(a734, a735, a736, a737, a50, a51, a54, a55, 1, a740, a741, a742, a743, a744, a745, a746, a747, n, c); }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a747[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 17, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a746[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 18, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a742[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 19, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a737[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 20, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a734[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 21, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a740[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 22, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a741[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 23, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a743[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 24, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a744[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 25, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a745[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 26, memory_order_relaxed); } } }
    if (a223[c]) { if (!isfinite(a48[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 27, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a49[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 28, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 29, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 30, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a54[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 21, 1u << 31, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a55[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 0, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a56[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 1, memory_order_relaxed); } }
    if (a223[c]) { a748[c] = (a55[c] <= as_type<float>(0x00000000)); }
    if (a223[c]) { if (a748[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 2, memory_order_relaxed); } }
    if (a223[c]) { a749[c] = (a54[c] >= as_type<float>(0x00000000)); }
    if (a223[c]) { if (a749[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 3, memory_order_relaxed); } }
    if (a223[c]) { a750[c] = (a56[c] < as_type<float>(0x00000000)); }
    if (a223[c]) { if (a750[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 4, memory_order_relaxed); } }
    if (a223[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a223[c]) { ruc_soil_properties(a747, a746, a742, a737, a734, a740, a741, a743, a744, a745, a48, a49, a50, a51, a54, a55, a56, as_type<float>(0x3f666667), a751, a752, a753, a754, n, c, forms); }
    if (a223[c]) { ruc_soil_canopy_setup(a734, a40, a155, a154, a44, a148, a20, a57, a51, a52, a755, a756, a757, a758, n, c); }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a740[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 5, memory_order_relaxed); } } }
    if (a223[c]) { if ((a72[c] < 1 || a72[c] >= RUC_NZS)) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 6, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (a223[c]) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 7, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a19[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 8, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a111[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 9, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 10, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 11, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a52[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 12, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a53[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 13, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a46[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 14, memory_order_relaxed); } }
    if (a223[c]) { a759[c] = (a525[c] < 1); }
    if (a223[c]) { a760[c] = (a525[c] > ncategory); }
    if (a223[c]) { a761[c] = (a759[c] || a760[c]); }
    if (a223[c]) { if (a761[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 15, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (a223[c]) { a762[c] = (a52[c] <= a53[c]); }
    if (a223[c]) { if (a762[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 16, memory_order_relaxed); } }
    if (a223[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a223[c]) { ruc_transpiration(a740, a14, a19, a111, a50, a51, a52, a53, a46, a525, a0, a1, a5, a72, rsmax, a763, a764, n, c); }
    if (true) { a765[c] = 1; }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a751[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 17, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a754[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 18, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a735[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 19, memory_order_relaxed); } } }
    if (a223[c]) { if ((a72[c] < 1 || a72[c] >= RUC_NZS)) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 20, memory_order_relaxed); /* unused diagnostic payload */; } }
    if (a223[c]) { if (!isfinite(a16[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 21, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a97[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 22, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a39[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 23, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 24, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 25, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a41[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 26, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a145[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 27, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a715[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 28, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 29, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a45[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 30, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a46[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 22, 1u << 31, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a42[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 0, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a101[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 1, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a19[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 2, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a757[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 3, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a756[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 4, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a764[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 5, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a755[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 6, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a47[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 7, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a758[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 8, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a765[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 9, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 10, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 11, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a55[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 12, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a150[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 13, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a154[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 14, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a155[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 15, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a156[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 16, memory_order_relaxed); } }
    if (a223[c]) { a766[c] = (a751[0 * n + c] <= as_type<float>(0x00000000)); }
    if (a223[c]) { if (a766[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 17, memory_order_relaxed); } }
    if (a223[c]) { a767[c] = (a754[0 * n + c] <= as_type<float>(0x00000000)); }
    if (a223[c]) { if (a767[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 18, memory_order_relaxed); } }
    if (a223[c]) { a768[c] = (a39[c] <= as_type<float>(0x00000000)); }
    if (a223[c]) { if (a768[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 19, memory_order_relaxed); } }
    if (a223[c]) { a769[c] = (a42[c] <= as_type<float>(0x00000000)); }
    if (a223[c]) { if (a769[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 20, memory_order_relaxed); } }
    if (a223[c]) { a770[c] = (a47[c] < as_type<float>(0x00000000)); }
    if (a223[c]) { a771[c] = (a47[c] > as_type<float>(0x3f800000)); }
    if (a223[c]) { a772[c] = (a770[c] || a771[c]); }
    if (a223[c]) { if (a772[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 21, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a69[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 22, memory_order_relaxed); } }
    if (a223[c]) { a779[c] = (a69[c] < as_type<float>(0x00000000)); }
    if (a223[c]) { if (a779[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 23, memory_order_relaxed); } }
    if (a223[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a223[c]) { ruc_soil_temperature_step(a751, a754, a735, a16, a97, a39, a14, a40, a145, a715, a44, a45, a42, a101, a757, a756, a764, a47, a758, a150, a154, a72, a4, delt, a69, as_type<float>(0x4a7f4f60), a773, a774, a775, a776, a777, a778, n, c); }
    if (a223[c]) { ruc_soil_phase_partition(a734, a773, a736, a737, a50, a51, a54, a55, 0, a780, a781, a782, a783, a784, a785, a786, a787, n, c); }
    if (a223[c]) { ruc_soil_prepare_moisture(a40, a776, a44, a42, a101, a757, a763, a72, a108, a788, a789, a790, a791, a792, n, c); }
    if (true) { a793[c] = 0; }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a752[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 24, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a753[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 25, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a788[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 26, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a781[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 27, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a734[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 28, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a780[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 29, memory_order_relaxed); } } }
    if (a223[c]) { if (!isfinite(a776[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 30, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a775[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 23, 1u << 31, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a777[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 0, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a41[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 1, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 2, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a791[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 3, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 4, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a102[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 5, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a790[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 6, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a793[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 7, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a101[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 8, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a793[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 9, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a758[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 10, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a50[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 11, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a51[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 12, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a52[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 13, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a56[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 14, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a792[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 15, memory_order_relaxed); } }
    if (a223[c]) { a794[c] = (a50[c] <= as_type<float>(0x00000000)); }
    if (a223[c]) { if (a794[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 16, memory_order_relaxed); } }
    if (a223[c]) { a795[c] = (a52[c] <= a51[c]); }
    if (a223[c]) { if (a795[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 17, memory_order_relaxed); } }
    if (a223[c]) { a796[c] = (a56[c] < as_type<float>(0x00000000)); }
    if (a223[c]) { if (a796[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 18, memory_order_relaxed); } }
    if (a223[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a223[c]) { ruc_soil_moisture_step(a752, a753, a788, a781, a734, a780, a776, a775, a777, a41, a40, a791, a44, a102, a790, a793, a101, a793, a758, a50, a51, a52, a56, a792, delt, a797, a798, a799, a800, a801, a802, a803, n, c); }
    if (a223[c]) { ruc_soil_finalize(a781, a773, a738, a797, a739, a737, a751, a754, a148, a790, a774, a775, a776, a777, a789, a756, a758, a792, a45, a42, a14, a39, a44, a40, a101, a715, a16, a778, delt, a804, a805, a806, a807, a808, a809, a810, a811, a812, a813, a814, a815, n, c); }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a797[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 19, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a773[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 20, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a736[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 21, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a737[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 22, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a781[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 23, memory_order_relaxed); } } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a798[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 24, memory_order_relaxed); } } }
    if (a223[c]) { if (!isfinite(a804[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 25, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a790[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 26, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a774[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 27, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a775[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 28, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a776[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 29, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a777[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 30, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a805[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 24, 1u << 31, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a806[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 0, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a807[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 1, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a808[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 2, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a809[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 3, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a810[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 4, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a811[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 5, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a812[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 6, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a813[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 7, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a814[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 8, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a800[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 9, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a801[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 10, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a799[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 11, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a802[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 12, memory_order_relaxed); } }
    if (a223[c]) { if (!isfinite(a815[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 13, memory_order_relaxed); } }
    if (a223[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a128[k * n + c] = a797[k * n + c]; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a129[k * n + c] = a773[k * n + c]; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a130[k * n + c] = a736[k * n + c]; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a131[k * n + c] = a737[k * n + c]; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a132[k * n + c] = a781[k * n + c]; } }
    if (a223[c]) { for (int k = 0; k < RUC_NZS; ++k) { a133[k * n + c] = a798[k * n + c]; } }
    if (a223[c]) { a148[c] = a804[c]; }
    if (a223[c]) { a153[c] = a790[c]; }
    if (a223[c]) { a150[c] = a774[c]; }
    if (a223[c]) { a154[c] = a775[c]; }
    if (a223[c]) { a155[c] = a776[c]; }
    if (a223[c]) { a156[c] = a777[c]; }
    if (a223[c]) { a163[c] = a805[c]; }
    if (a223[c]) { a164[c] = a806[c]; }
    if (a223[c]) { a165[c] = a807[c]; }
    if (a223[c]) { a166[c] = a808[c]; }
    if (a223[c]) { a167[c] = a809[c]; }
    if (a223[c]) { a168[c] = a810[c]; }
    if (a223[c]) { a169[c] = a811[c]; }
    if (a223[c]) { a171[c] = a812[c]; }
    if (a223[c]) { a172[c] = a813[c]; }
    if (a223[c]) { a173[c] = a814[c]; }
    if (a223[c]) { a174[c] = a800[c]; }
    if (a223[c]) { a175[c] = a801[c]; }
    if (a223[c]) { a146[c] = a799[c]; }
    if (a223[c]) { a177[c] = a815[c]; }
    if (a223[c]) { a176[c] = a802[c]; }
    if (a225[c]) { a816[c] = (a524[c] != a112[c]); }
    if (a225[c]) { a817[c] = ruc_fsub_rn(as_type<float>(0x3f800000), a524[c]); }
    if (a225[c]) { a818[c] = ruc_fdiv_rn(a13[c], a817[c]); }
    if (a225[c]) { a819[c] = ruc_fsub_rn(as_type<float>(0x3f800000), a112[c]); }
    if (a225[c]) { a820[c] = ruc_fmul_rn(a818[c], a819[c]); }
    if (a225[c]) { a821[c] = (a816[c] ? a820[c] : a13[c]); }
    if (a225[c]) { a524[c] = a112[c]; }
    if (a225[c]) { a822[c] = ruc_fadd_rn(a821[c], a716[c]); }
    if (a225[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a85[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 14, memory_order_relaxed); } } }
    if (a225[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a86[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 15, memory_order_relaxed); } } }
    if (a225[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a129[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 16, memory_order_relaxed); } } }
    if (a225[c]) { if (!isfinite(a16[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 17, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a97[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 18, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a39[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 19, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a40[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 20, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a145[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 21, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a822[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 22, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a44[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 23, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a45[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 24, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a42[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 25, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a14[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 26, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a150[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 27, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a154[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 28, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a155[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 29, memory_order_relaxed); } }
    if (a225[c]) { a823[c] = (a86[0 * n + c] <= as_type<float>(0x00000000)); }
    if (a225[c]) { if (a823[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 30, memory_order_relaxed); } }
    if (a225[c]) { a824[c] = (a85[0 * n + c] <= as_type<float>(0x00000000)); }
    if (a225[c]) { if (a824[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 25, 1u << 31, memory_order_relaxed); } }
    if (a225[c]) { a825[c] = (a39[c] <= as_type<float>(0x00000000)); }
    if (a225[c]) { if (a825[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 0, memory_order_relaxed); } }
    if (a225[c]) { a826[c] = (a42[c] <= as_type<float>(0x00000000)); }
    if (a225[c]) { if (a826[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 1, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a69[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 2, memory_order_relaxed); } }
    if (a225[c]) { a840[c] = (a69[c] < as_type<float>(0x00000000)); }
    if (a225[c]) { if (a840[c]) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 3, memory_order_relaxed); } }
    if (a225[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a225[c]) { ruc_sea_ice_step(a85, a86, a129, a16, a97, a39, a40, a145, a822, a44, a45, a42, a14, a150, a154, a155, a4, delt, a69, as_type<float>(0x4a7f4f60), 0, a827, a828, a829, a830, a831, a832, a833, a834, a835, a836, a837, a838, a839, n, c); }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a841[k * n + c] = as_type<float>(0x3f800000); } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a842[k * n + c] = as_type<float>(0x00000000); } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a843[k * n + c] = as_type<float>(0x3f800000); } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a844[k * n + c] = as_type<float>(0x3f800000); } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { a845[k * n + c] = as_type<float>(0x00000000); } }
    if (a225[c]) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a827[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 4, memory_order_relaxed); } } }
    if (a225[c]) { if (!isfinite(a828[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 5, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a829[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 6, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a830[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 7, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a831[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 8, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a832[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 9, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a833[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 10, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a834[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 11, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a835[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 12, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a836[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 13, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a837[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 14, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a838[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 15, memory_order_relaxed); } }
    if (a225[c]) { if (!isfinite(a839[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 16, memory_order_relaxed); } }
    if (a225[c]) { if (failed) { ruc_sfctmp_zero_outputs(r, p, n, c); return; } }
    if (a225[c]) { for (int k = 0; k < RUC_NZS; ++k) { a129[k * n + c] = a827[k * n + c]; } }
    if (a225[c]) { a153[c] = a828[c]; }
    if (a225[c]) { a150[c] = a829[c]; }
    if (a225[c]) { a154[c] = a830[c]; }
    if (a225[c]) { a155[c] = a831[c]; }
    if (a225[c]) { a156[c] = a832[c]; }
    if (a225[c]) { a166[c] = a833[c]; }
    if (a225[c]) { a167[c] = a834[c]; }
    if (a225[c]) { a168[c] = a835[c]; }
    if (a225[c]) { a169[c] = a836[c]; }
    if (a225[c]) { a171[c] = a837[c]; }
    if (a225[c]) { a172[c] = a838[c]; }
    if (a225[c]) { a173[c] = a839[c]; }
    if (a225[c]) { a846[c] = ruc_fmul_rn(a166[c], as_type<float>(0x3a83126f)); }
    if (a225[c]) { a163[c] = a846[c]; }
    if (a225[c]) { a164[c] = as_type<float>(0x00000000); }
    if (a225[c]) { a165[c] = as_type<float>(0x00000000); }
    if (a225[c]) { a174[c] = a16[c]; }
    if (a225[c]) { a175[c] = as_type<float>(0x00000000); }
    if (a225[c]) { a146[c] = as_type<float>(0x3f800000); }
    if (a225[c]) { a176[c] = as_type<float>(0x00000000); }
    if (a225[c]) { a148[c] = as_type<float>(0x00000000); }
    if (a225[c]) { for (int k = 0; k < RUC_NZS; ++k) { a128[k * n + c] = as_type<float>(0x3f800000); } }
    if (a225[c]) { for (int k = 0; k < RUC_NZS; ++k) { a133[k * n + c] = as_type<float>(0x00000000); } }
    if (a225[c]) { for (int k = 0; k < RUC_NZS; ++k) { a132[k * n + c] = as_type<float>(0x3f800000); } }
    if (a225[c]) { for (int k = 0; k < RUC_NZS; ++k) { a130[k * n + c] = as_type<float>(0x3f800000); } }
    if (a225[c]) { for (int k = 0; k < RUC_NZS; ++k) { a131[k * n + c] = as_type<float>(0x00000000); } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a128[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 17, memory_order_relaxed); } } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a129[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 18, memory_order_relaxed); } } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a130[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 19, memory_order_relaxed); } } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a131[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 20, memory_order_relaxed); } } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a132[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 21, memory_order_relaxed); } } }
    if (true) { for (int k = 0; k < RUC_NZS; ++k) { if (!isfinite(a133[k * n + c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 22, memory_order_relaxed); } } }
    if (true) { if (!isfinite(a134[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 23, memory_order_relaxed); } }
    if (true) { if (!isfinite(a135[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 24, memory_order_relaxed); } }
    if (true) { if (!isfinite(a136[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 25, memory_order_relaxed); } }
    if (true) { if (!isfinite(a137[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 26, memory_order_relaxed); } }
    if (true) { if (!isfinite(a138[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 27, memory_order_relaxed); } }
    if (true) { if (!isfinite(a139[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 28, memory_order_relaxed); } }
    if (true) { if (!isfinite(a140[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 29, memory_order_relaxed); } }
    if (true) { if (!isfinite(a141[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 30, memory_order_relaxed); } }
    if (true) { if (!isfinite(a142[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 26, 1u << 31, memory_order_relaxed); } }
    if (true) { if (!isfinite(a143[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 0, memory_order_relaxed); } }
    if (true) { if (!isfinite(a144[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 1, memory_order_relaxed); } }
    if (true) { if (!isfinite(a145[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 2, memory_order_relaxed); } }
    if (true) { if (!isfinite(a146[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 3, memory_order_relaxed); } }
    if (true) { if (!isfinite(a524[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 4, memory_order_relaxed); } }
    if (true) { if (!isfinite(a148[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 5, memory_order_relaxed); } }
    if (true) { if (!isfinite(a149[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 6, memory_order_relaxed); } }
    if (true) { if (!isfinite(a150[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 7, memory_order_relaxed); } }
    if (true) { if (!isfinite(a151[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 8, memory_order_relaxed); } }
    if (true) { if (!isfinite(a152[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 9, memory_order_relaxed); } }
    if (true) { if (!isfinite(a153[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 10, memory_order_relaxed); } }
    if (true) { if (!isfinite(a154[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 11, memory_order_relaxed); } }
    if (true) { if (!isfinite(a155[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 12, memory_order_relaxed); } }
    if (true) { if (!isfinite(a156[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 13, memory_order_relaxed); } }
    if (true) { if (!isfinite(a157[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 14, memory_order_relaxed); } }
    if (true) { if (!isfinite(a158[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 15, memory_order_relaxed); } }
    if (true) { if (!isfinite(a159[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 16, memory_order_relaxed); } }
    if (true) { if (!isfinite(a160[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 17, memory_order_relaxed); } }
    if (true) { if (!isfinite(a161[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 18, memory_order_relaxed); } }
    if (true) { if (!isfinite(a162[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 19, memory_order_relaxed); } }
    if (true) { if (!isfinite(a163[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 20, memory_order_relaxed); } }
    if (true) { if (!isfinite(a164[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 21, memory_order_relaxed); } }
    if (true) { if (!isfinite(a165[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 22, memory_order_relaxed); } }
    if (true) { if (!isfinite(a166[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 23, memory_order_relaxed); } }
    if (true) { if (!isfinite(a167[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 24, memory_order_relaxed); } }
    if (true) { if (!isfinite(a168[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 25, memory_order_relaxed); } }
    if (true) { if (!isfinite(a169[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 26, memory_order_relaxed); } }
    if (true) { if (!isfinite(a170[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 27, memory_order_relaxed); } }
    if (true) { if (!isfinite(a171[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 28, memory_order_relaxed); } }
    if (true) { if (!isfinite(a172[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 29, memory_order_relaxed); } }
    if (true) { if (!isfinite(a173[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 30, memory_order_relaxed); } }
    if (true) { if (!isfinite(a174[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 27, 1u << 31, memory_order_relaxed); } }
    if (true) { if (!isfinite(a175[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 28, 1u << 0, memory_order_relaxed); } }
    if (true) { if (!isfinite(a176[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 28, 1u << 1, memory_order_relaxed); } }
    if (true) { if (!isfinite(a177[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 28, 1u << 2, memory_order_relaxed); } }
    if (true) { if (!isfinite(a178[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 28, 1u << 3, memory_order_relaxed); } }
    if (true) { if (!isfinite(a179[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 28, 1u << 4, memory_order_relaxed); } }
    if (true) { if (!isfinite(a180[c])) { failed = true; atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) + 28, 1u << 5, memory_order_relaxed); } }
    r.raw[p.alive_base + uint(c)] = true;
}
#undef a0
#undef a1
#undef a2
#undef a3
#undef a4
#undef a5
#undef a6
#undef a7
#undef a8
#undef a9
#undef a10
#undef a11
#undef a12
#undef a13
#undef a14
#undef a15
#undef a16
#undef a17
#undef a18
#undef a19
#undef a20
#undef a21
#undef a22
#undef a23
#undef a24
#undef a25
#undef a26
#undef a27
#undef a28
#undef a29
#undef a30
#undef a31
#undef a32
#undef a33
#undef a34
#undef a35
#undef a36
#undef a37
#undef a38
#undef a39
#undef a40
#undef a41
#undef a42
#undef a43
#undef a44
#undef a45
#undef a46
#undef a47
#undef a48
#undef a49
#undef a50
#undef a51
#undef a52
#undef a53
#undef a54
#undef a55
#undef a56
#undef a57
#undef a58
#undef a59
#undef a60
#undef a61
#undef a62
#undef a63
#undef a64
#undef a65
#undef a66
#undef a67
#undef a68
#undef a69
#undef a70
#undef a71
#undef a72
#undef a73
#undef a74
#undef a75
#undef a76
#undef a77
#undef a78
#undef a79
#undef a80
#undef a81
#undef a82
#undef a83
#undef a84
#undef a85
#undef a86
#undef a87
#undef a88
#undef a89
#undef a90
#undef a91
#undef a92
#undef a93
#undef a94
#undef a95
#undef a96
#undef a97
#undef a98
#undef a99
#undef a100
#undef a101
#undef a102
#undef a103
#undef a104
#undef a105
#undef a106
#undef a107
#undef a108
#undef a109
#undef a110
#undef a111
#undef a112
#undef a113
#undef a114
#undef a115
#undef a116
#undef a117
#undef a118
#undef a119
#undef a120
#undef a121
#undef a122
#undef a123
#undef a124
#undef a125
#undef a126
#undef a127
#undef a128
#undef a129
#undef a130
#undef a131
#undef a132
#undef a133
#undef a134
#undef a135
#undef a136
#undef a137
#undef a138
#undef a139
#undef a140
#undef a141
#undef a142
#undef a143
#undef a144
#undef a145
#undef a146
#undef a147
#undef a148
#undef a149
#undef a150
#undef a151
#undef a152
#undef a153
#undef a154
#undef a155
#undef a156
#undef a157
#undef a158
#undef a159
#undef a160
#undef a161
#undef a162
#undef a163
#undef a164
#undef a165
#undef a166
#undef a167
#undef a168
#undef a169
#undef a170
#undef a171
#undef a172
#undef a173
#undef a174
#undef a175
#undef a176
#undef a177
#undef a178
#undef a179
#undef a180
#undef a181
#undef a182
#undef a183
#undef a184
#undef a185
#undef a186
#undef a187
#undef a188
#undef a189
#undef a190
#undef a191
#undef a192
#undef a193
#undef a194
#undef a195
#undef a196
#undef a197
#undef a198
#undef a199
#undef a200
#undef a201
#undef a202
#undef a203
#undef a204
#undef a205
#undef a206
#undef a207
#undef a208
#undef a209
#undef a210
#undef a211
#undef a212
#undef a213
#undef a214
#undef a215
#undef a216
#undef a217
#undef a218
#undef a219
#undef a220
#undef a221
#undef a222
#undef a223
#undef a224
#undef a225
#undef a226
#undef a227
#undef a228
#undef a229
#undef a230
#undef a231
#undef a232
#undef a233
#undef a234
#undef a235
#undef a236
#undef a237
#undef a238
#undef a239
#undef a240
#undef a241
#undef a242
#undef a243
#undef a244
#undef a245
#undef a246
#undef a247
#undef a248
#undef a249
#undef a250
#undef a251
#undef a252
#undef a253
#undef a254
#undef a255
#undef a256
#undef a257
#undef a258
#undef a259
#undef a260
#undef a261
#undef a262
#undef a263
#undef a264
#undef a265
#undef a266
#undef a267
#undef a268
#undef a269
#undef a270
#undef a271
#undef a272
#undef a273
#undef a274
#undef a275
#undef a276
#undef a277
#undef a278
#undef a279
#undef a280
#undef a281
#undef a282
#undef a283
#undef a284
#undef a285
#undef a286
#undef a287
#undef a288
#undef a289
#undef a290
#undef a291
#undef a292
#undef a293
#undef a294
#undef a295
#undef a296
#undef a297
#undef a298
#undef a299
#undef a300
#undef a301
#undef a302
#undef a303
#undef a304
#undef a305
#undef a306
#undef a307
#undef a308
#undef a309
#undef a310
#undef a311
#undef a312
#undef a313
#undef a314
#undef a315
#undef a316
#undef a317
#undef a318
#undef a319
#undef a320
#undef a321
#undef a322
#undef a323
#undef a324
#undef a325
#undef a326
#undef a327
#undef a328
#undef a329
#undef a330
#undef a331
#undef a332
#undef a333
#undef a334
#undef a335
#undef a336
#undef a337
#undef a338
#undef a339
#undef a340
#undef a341
#undef a342
#undef a343
#undef a344
#undef a345
#undef a346
#undef a347
#undef a348
#undef a349
#undef a350
#undef a351
#undef a352
#undef a353
#undef a354
#undef a355
#undef a356
#undef a357
#undef a358
#undef a359
#undef a360
#undef a361
#undef a362
#undef a363
#undef a364
#undef a365
#undef a366
#undef a367
#undef a368
#undef a369
#undef a370
#undef a371
#undef a372
#undef a373
#undef a374
#undef a375
#undef a376
#undef a377
#undef a378
#undef a379
#undef a380
#undef a381
#undef a382
#undef a383
#undef a384
#undef a385
#undef a386
#undef a387
#undef a388
#undef a389
#undef a390
#undef a391
#undef a392
#undef a393
#undef a394
#undef a395
#undef a396
#undef a397
#undef a398
#undef a399
#undef a400
#undef a401
#undef a402
#undef a403
#undef a404
#undef a405
#undef a406
#undef a407
#undef a408
#undef a409
#undef a410
#undef a411
#undef a412
#undef a413
#undef a414
#undef a415
#undef a416
#undef a417
#undef a418
#undef a419
#undef a420
#undef a421
#undef a422
#undef a423
#undef a424
#undef a425
#undef a426
#undef a427
#undef a428
#undef a429
#undef a430
#undef a431
#undef a432
#undef a433
#undef a434
#undef a435
#undef a436
#undef a437
#undef a438
#undef a439
#undef a440
#undef a441
#undef a442
#undef a443
#undef a444
#undef a445
#undef a446
#undef a447
#undef a448
#undef a449
#undef a450
#undef a451
#undef a452
#undef a453
#undef a454
#undef a455
#undef a456
#undef a457
#undef a458
#undef a459
#undef a460
#undef a461
#undef a462
#undef a463
#undef a464
#undef a465
#undef a466
#undef a467
#undef a468
#undef a469
#undef a470
#undef a471
#undef a472
#undef a473
#undef a474
#undef a475
#undef a476
#undef a477
#undef a478
#undef a479
#undef a480
#undef a481
#undef a482
#undef a483
#undef a484
#undef a485
#undef a486
#undef a487
#undef a488
#undef a489
#undef a490
#undef a491
#undef a492
#undef a493
#undef a494
#undef a495
#undef a496
#undef a497
#undef a498
#undef a499
#undef a500
#undef a501
#undef a502
#undef a503
#undef a504
#undef a505
#undef a506
#undef a507
#undef a508
#undef a509
#undef a510
#undef a511
#undef a512
#undef a513
#undef a514
#undef a515
#undef a516
#undef a517
#undef a518
#undef a519
#undef a520
#undef a521
#undef a522
#undef a523
#undef a524
#undef a525
#undef a526
#undef a527
#undef a528
#undef a529
#undef a530
#undef a531
#undef a532
#undef a533
#undef a534
#undef a535
#undef a536
#undef a537
#undef a538
#undef a539
#undef a540
#undef a541
#undef a542
#undef a543
#undef a544
#undef a545
#undef a546
#undef a547
#undef a548
#undef a549
#undef a550
#undef a551
#undef a552
#undef a553
#undef a554
#undef a555
#undef a556
#undef a557
#undef a558
#undef a559
#undef a560
#undef a561
#undef a562
#undef a563
#undef a564
#undef a565
#undef a566
#undef a567
#undef a568
#undef a569
#undef a570
#undef a571
#undef a572
#undef a573
#undef a574
#undef a575
#undef a576
#undef a577
#undef a578
#undef a579
#undef a580
#undef a581
#undef a582
#undef a583
#undef a584
#undef a585
#undef a586
#undef a587
#undef a588
#undef a589
#undef a590
#undef a591
#undef a592
#undef a593
#undef a594
#undef a595
#undef a596
#undef a597
#undef a598
#undef a599
#undef a600
#undef a601
#undef a602
#undef a603
#undef a604
#undef a605
#undef a606
#undef a607
#undef a608
#undef a609
#undef a610
#undef a611
#undef a612
#undef a613
#undef a614
#undef a615
#undef a616
#undef a617
#undef a618
#undef a619
#undef a620
#undef a621
#undef a622
#undef a623
#undef a624
#undef a625
#undef a626
#undef a627
#undef a628
#undef a629
#undef a630
#undef a631
#undef a632
#undef a633
#undef a634
#undef a635
#undef a636
#undef a637
#undef a638
#undef a639
#undef a640
#undef a641
#undef a642
#undef a643
#undef a644
#undef a645
#undef a646
#undef a647
#undef a648
#undef a649
#undef a650
#undef a651
#undef a652
#undef a653
#undef a654
#undef a655
#undef a656
#undef a657
#undef a658
#undef a659
#undef a660
#undef a661
#undef a662
#undef a663
#undef a664
#undef a665
#undef a666
#undef a667
#undef a668
#undef a669
#undef a670
#undef a671
#undef a672
#undef a673
#undef a674
#undef a675
#undef a676
#undef a677
#undef a678
#undef a679
#undef a680
#undef a681
#undef a682
#undef a683
#undef a684
#undef a685
#undef a686
#undef a687
#undef a688
#undef a689
#undef a690
#undef a691
#undef a692
#undef a693
#undef a694
#undef a695
#undef a696
#undef a697
#undef a698
#undef a699
#undef a700
#undef a701
#undef a702
#undef a703
#undef a704
#undef a705
#undef a706
#undef a707
#undef a708
#undef a709
#undef a710
#undef a711
#undef a712
#undef a713
#undef a714
#undef a715
#undef a716
#undef a717
#undef a718
#undef a719
#undef a720
#undef a721
#undef a722
#undef a723
#undef a724
#undef a725
#undef a726
#undef a727
#undef a728
#undef a729
#undef a730
#undef a731
#undef a732
#undef a733
#undef a734
#undef a735
#undef a736
#undef a737
#undef a738
#undef a739
#undef a740
#undef a741
#undef a742
#undef a743
#undef a744
#undef a745
#undef a746
#undef a747
#undef a748
#undef a749
#undef a750
#undef a751
#undef a752
#undef a753
#undef a754
#undef a755
#undef a756
#undef a757
#undef a758
#undef a759
#undef a760
#undef a761
#undef a762
#undef a763
#undef a764
#undef a765
#undef a766
#undef a767
#undef a768
#undef a769
#undef a770
#undef a771
#undef a772
#undef a773
#undef a774
#undef a775
#undef a776
#undef a777
#undef a778
#undef a779
#undef a780
#undef a781
#undef a782
#undef a783
#undef a784
#undef a785
#undef a786
#undef a787
#undef a788
#undef a789
#undef a790
#undef a791
#undef a792
#undef a793
#undef a794
#undef a795
#undef a796
#undef a797
#undef a798
#undef a799
#undef a800
#undef a801
#undef a802
#undef a803
#undef a804
#undef a805
#undef a806
#undef a807
#undef a808
#undef a809
#undef a810
#undef a811
#undef a812
#undef a813
#undef a814
#undef a815
#undef a816
#undef a817
#undef a818
#undef a819
#undef a820
#undef a821
#undef a822
#undef a823
#undef a824
#undef a825
#undef a826
#undef a827
#undef a828
#undef a829
#undef a830
#undef a831
#undef a832
#undef a833
#undef a834
#undef a835
#undef a836
#undef a837
#undef a838
#undef a839
#undef a840
#undef a841
#undef a842
#undef a843
#undef a844
#undef a845
#undef a846
