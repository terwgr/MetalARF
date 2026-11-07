#include <metal_stdlib>
using namespace metal;
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_gfk_log.metalh"
struct GFReal { uint bits; GFReal()=default; GFReal(float f):bits(as_type<uint>(f)){} static GFReal word(uint x){GFReal a;a.bits=x;return a;} };
static inline GFReal gf_nan(uint r,uint a,uint b){if(!soft32_is_nan(r))return GFReal::word(r);if(soft32_is_nan(a))return GFReal::word(soft32_quiet_nan(a));if(soft32_is_nan(b))return GFReal::word(soft32_quiet_nan(b));return GFReal::word(0xffc00000u);}
static inline GFReal operator+(GFReal a,GFReal b){return gf_nan(soft32_add(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator-(GFReal a,GFReal b){return gf_nan(soft32_sub(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator*(GFReal a,GFReal b){return gf_nan(soft32_mul(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator/(GFReal a,GFReal b){return gf_nan(soft32_div(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator-(GFReal a){return GFReal::word(a.bits^0x80000000u);}
static inline bool operator<(GFReal a,GFReal b){return soft32_less(a.bits,b.bits);}
static inline GFReal gf_log(GFReal a){return gf_nan(soft_gfk_log(a.bits),a.bits,0u);}
static inline GFReal gf_pow(GFReal a,GFReal b){return gf_nan(soft_gfk_pow_bits(a.bits,b.bits),a.bits,0u);}
static inline GFReal satvap(GFReal temp2)
{

  GFReal eilog;
  GFReal ewlog;
  GFReal ewlog2;
  GFReal ewlog3;
  GFReal ewlog4;
  GFReal temp;
  GFReal toot;
  GFReal toto;
  GFReal tsot;
  GFReal __result_satvap;

  temp = temp2 - 2.73154998779296875e+2f;
  if (temp < -2.0e+1f)
    {
      toot = 2.73160003662109375e+2f / temp2;
      toto = 1.0e+0f / toot;
      eilog = ((-(((toot - 1.0e+0f)) * 9.09718036651611328125e+0f) - ((gf_log (toot) / 2.302585124969482421875e+0f)) * 3.5665400028228759765625e+0f) + ((1.0e+0f - toto)) * 8.7679302692413330078125e-1f) + 7.858350276947021484375e-1f;
      __result_satvap = gf_pow (1.0e+1f, eilog);
    }
  else
    {
      tsot = 3.73160003662109375e+2f / temp2;
      ewlog = ((gf_log (tsot) / 2.302585124969482421875e+0f)) * 5.028079986572265625e+0f - ((tsot - 1.0e+0f)) * 7.90297985076904296875e+0f;
      ewlog2 = ewlog - ((gf_pow (1.0e+1f, ((((1.0e+0f - ((1.0e+0f / tsot)))) * 1.13439998626708984375e+1f))) - 1.0e+0f)) * 1.38160004325982299633324146270751953125e-7f;
      ewlog3 = ((gf_pow (1.0e+1f, ((-(((tsot - 1.0e+0f)) * 3.491489887237548828125e+0f)))) - 1.0e+0f)) * 8.13280045986175537109375e-3f + ewlog2;
      ewlog4 = ewlog3 + 3.005714893341064453125e+0f;
      __result_satvap = gf_pow (1.0e+1f, ewlog4);
    }

  return __result_satvap;
  return __result_satvap;
}
#include "hexv842_mpas_gf_satvap_f32_bindings.metalh"
kernel void hexv842_mpas_gf_satvap_f32(constant Hexv842MpasGfSatvapF32Resources &r [[buffer(0)]],constant Hexv842MpasGfSatvapF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]){
 if(gid>=uint(p.n))return; r.output[gid]=as_type<float>(satvap(GFReal::word(as_type<uint>(r.temperature[gid]))).bits);
}
