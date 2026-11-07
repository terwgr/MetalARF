#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "soft_binary64.metalh"
using namespace metal;
#pragma clang fp contract(off)
// Source arithmetic with gradual underflow and software binary64 clocks.
static inline bool operator==(Exact32 a, Exact32 b) { return soft32_equal(a.bits,b.bits); }
static inline bool operator!=(Exact32 a, Exact32 b) { return !soft32_equal(a.bits,b.bits); }
static inline bool operator<(Exact32 a, Exact32 b) { return soft32_less(a.bits,b.bits); }
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_less_equal(a.bits,b.bits); }
static inline Exact32 fmaxf(Exact32 a, Exact32 b) { return exact_max(a,b); }
static inline Exact32 fminf(Exact32 a, Exact32 b) { return exact_min(a,b); }
struct NudgeComplex { Exact32 x,y; };
static inline NudgeComplex make_cuFloatComplex(Exact32 a,Exact32 b) { return {a,b}; }
static inline Exact32 cuCrealf(NudgeComplex a) { return a.x; }
struct Nudge64 {
    ulong bits;
    Nudge64() : bits(0ul) {}
    Nudge64(float value)
        : bits(soft64_from_f32_bits(as_type<uint>(value))) {}
    Nudge64(int value) : bits(soft64_from_i32(value)) {}
    static Nudge64 from_bits(ulong value) {
        Nudge64 result;
        result.bits = value;
        return result;
    }
    operator Exact32() const { return Exact32::from_bits(soft64_to_f32_bits(bits)); }
};

static inline Nudge64 operator+(Nudge64 a, Nudge64 b) {
    Nudge64 result; result.bits = soft64_add(a.bits, b.bits); return result;
}
static inline Nudge64 operator-(Nudge64 a, Nudge64 b) {
    Nudge64 result; result.bits = soft64_sub(a.bits, b.bits); return result;
}
static inline Nudge64 operator*(Nudge64 a, Nudge64 b) {
    Nudge64 result; result.bits = soft64_mul(a.bits, b.bits); return result;
}
static inline Nudge64 operator/(Nudge64 a, Nudge64 b) {
    Nudge64 result; result.bits = soft64_div(a.bits, b.bits); return result;
}
static inline thread Nudge64 &operator+=(thread Nudge64 &a, Nudge64 b) {
    a.bits = soft64_add(a.bits, b.bits); return a;
}
static inline bool operator<(Nudge64 a, Nudge64 b) { return soft64_less(a.bits, b.bits); }
static inline bool operator>(Nudge64 a, Nudge64 b) { return soft64_greater(a.bits, b.bits); }
static inline bool operator<=(Nudge64 a, Nudge64 b) { return soft64_less_equal(a.bits, b.bits); }
static inline bool operator>=(Nudge64 a, Nudge64 b) { return soft64_greater_equal(a.bits, b.bits); }
static inline Nudge64 fmin(Nudge64 a, Nudge64 b) { return a > b ? b : a; }
static inline Nudge64 fmax(Nudge64 a, Nudge64 b) { return a < b ? b : a; }
static inline bool operator!=(Nudge64 a,Nudge64 b) { return !soft64_equal(a.bits,b.bits); }
static inline Nudge64 fabs(Nudge64 a) { return Nudge64::from_bits(soft64_abs_bits(a.bits)); }
static inline long llround(Nudge64 a) {
    bool negative=soft64_sign(a.bits);
    ulong magnitude=soft64_add(soft64_abs_bits(a.bits),0x3fe0000000000000ul);
    int v=soft64_to_i32_rz(magnitude).value;
    return negative ? -long(v) : long(v);
}
static inline bool due(thread const Nudge64 *clock, int cadence) {
    long step = llround(clock[0] / clock[1]) + 1;
    return step == 1 || step % cadence == 0;
}
static inline bool clear_between_calls(thread const Nudge64 *clock, int cadence) {
    // WRF clears at mod(itimestep-1,STEPFG)==0, then calculates at
    // step 1 or multiples of STEPFG. The first extra call is held until
    // the first regular call even when fgdtzero is enabled.
    return llround(clock[0]/clock[1]) % cadence == 0;
}
static inline Exact32 time_weight(thread const Nudge64 *clock, Nudge64 end, Nudge64 ramp) {
    if (end < 0.) return 1.f;
    Nudge64 stop = end + fmax(0., ramp);
    if (ramp != 0.) return (Exact32)fmin(1., fmax(0., (stop-clock[0])/fabs(ramp)));
    return clock[0] <= end ? 1.f : 0.f;
}

static inline Exact32 phi_at(device const Exact32 *a,device const Exact32 *b,int k,int j,int i,int ny,int nx,int terrain) {
 int q=(k*ny+j)*nx+i; return a[q]+b[terrain?q:k];
}
static inline Exact32 zm(device const Exact32 *a,device const Exact32 *b,device const Exact32 *ht,int k,int j,int i,int ny,int nx,int terrain) {
 return .5f*(phi_at(a,b,k,j,i,ny,nx,terrain)+phi_at(a,b,k+1,j,i,ny,nx,terrain))/9.81f-ht[j*nx+i];
}
#include "nudging_difference_bindings.metalh"
kernel void nudging_difference(constant NudgingDifferenceResources& res [[buffer(0)]], constant NudgingDifferenceParams& params [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const Exact32 *value=(device const Exact32*)res.value;
    device const Exact32 *a=(device const Exact32*)res.a;
    device const Exact32 *b=(device const Exact32*)res.b;
    device NudgeComplex *out=(device NudgeComplex*)res.out;
    Nudge64 clock[4]; for(int c=0;c<4;c++) clock[c]=Nudge64::from_bits(res.clock[c]);
    device uint *error=(device uint*)res.error;
    int n=params.n;
    int wrf=params.wrf;
    int cadence=params.cadence;
    Nudge64 end=Nudge64::from_bits(res.timing[0]);
    Nudge64 ramp=Nudge64::from_bits(res.timing[1]);
    int grid_nudging=params.grid_nudging;
    if (gid >= uint(n)) return;
    int i=int(gid);


    if (!due(clock, cadence) || time_weight(clock, end, ramp) == 0.f) {
        out[i] = make_cuFloatComplex(0.f, 0.f); return;
    }
    // Midpoint forcing. clock = elapsed, dt, interval_start, interval_end.
    bool valid = clock[1] > 0 && clock[0] >= clock[2]
        && (clock[0] + clock[1] <= clock[3] || (end >= 0. && clock[0] >= end));
    if (!valid) {
        atomic_fetch_or_explicit((device atomic_uint*)error, 1u, memory_order_relaxed);
        out[i] = make_cuFloatComplex(0.f, 0.f);
        return;
    }
    Exact32 alpha = (Exact32)((clock[0] + (wrf ? 0. : 0.5 * clock[1]) - clock[2])
                          / (clock[3] - clock[2]));
    // Grid FDDA holds the last analysis during a positive end ramp.
    // Spectral FDDA extrapolates the last two analyses instead.
    if (wrf && grid_nudging && end >= 0. && ramp > 0. && clock[0] >= end)
        alpha = (Exact32)((end-clock[2])/(clock[3]-clock[2]));
    out[i] = make_cuFloatComplex(a[i] + alpha * (b[i] - a[i]) - value[i], 0.f);
}

#include "nudging_truncate_modes_bindings.metalh"
kernel void nudging_truncate_modes(constant NudgingTruncateModesResources& res [[buffer(0)]], constant NudgingTruncateModesParams& params [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device NudgeComplex *spectrum=(device NudgeComplex*)res.spectrum;
    int n=params.n;
    int nx=params.nx;
    int ny=params.ny;
    int cutoff_x=params.cutoff_x;
    int cutoff_y=params.cutoff_y;
    if (gid >= uint(n)) return;
    int i=int(gid);


    int x = i % nx, y = (i / nx) % ny;
    int kx = min(x, nx - x), ky = min(y, ny - y);
    if (kx > cutoff_x || ky > cutoff_y)
        spectrum[i] = make_cuFloatComplex(0.f, 0.f);
}

#include "nudging_relax_bindings.metalh"
kernel void nudging_relax(constant NudgingRelaxResources& res [[buffer(0)]], constant NudgingRelaxParams& params [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device Exact32 *value=(device Exact32*)res.value;
    device const NudgeComplex *filtered=(device const NudgeComplex*)res.filtered;
    device const Exact32 *height_agl=(device const Exact32*)res.height_agl;
    device const Exact32 *pblh=(device const Exact32*)res.pblh;
    Nudge64 clock[4]; for(int c=0;c<4;c++) clock[c]=Nudge64::from_bits(res.clock[c]);
    int n=params.n;
    int nx=params.nx;
    int ny=params.ny;
    int boundary=params.boundary;
    Exact32 floor_m=params.floor_m;
    Exact32 ramp_m=params.ramp_m;
    Exact32 tau=params.tau;
    device const Exact32 *vertical_weight=(device const Exact32*)res.vertical_weight;
    device Exact32 *held=(device Exact32*)res.held;
    int wrf=params.wrf;
    int cadence=params.cadence;
    int fgdtzero=params.fgdtzero;
    Nudge64 end=Nudge64::from_bits(res.timing[0]);
    Nudge64 ramp=Nudge64::from_bits(res.timing[1]);
    if (gid >= uint(n)) return;
    int i=int(gid);


    if (!(clock[1] > 0 && clock[0] >= clock[2] &&
          (clock[0] + clock[1] <= clock[3] || (end >= 0. && clock[0] >= end)))) return;
    int x = i % nx, y = (i / nx) % ny;
    if (x < boundary || x >= nx - boundary || y < boundary || y >= ny - boundary) return;
    if (wrf) {
        if (due(clock, cadence)) {
            held[i] = vertical_weight[i] * time_weight(clock, end, ramp) * cuCrealf(filtered[i]) / tau;
        } else if (fgdtzero && clear_between_calls(clock,cadence)) {
            held[i] = 0.f;
        }
        if (time_weight(clock, end, ramp) > 0.f && held[i] != 0.f)
            value[i] += (Exact32)clock[1] * held[i];
        return;
    }
    Exact32 z0 = fmaxf(floor_m, pblh[i % (nx * ny)]);
    Exact32 r = fminf(1.f, fmaxf(0.f, (height_agl[i] - z0) / ramp_m));
    // Post-filter masking preserves the PBL exactly. Spatially varying masks
    // mix Fourier modes; do not claim high-mode invariance in that case.
    Exact32 weight = r * r * (3.f - 2.f * r);
    if (cadence > 1 || end >= 0.) {
        if (due(clock,cadence)) held[i] = weight*time_weight(clock,end,ramp)*cuCrealf(filtered[i])/tau;
        else if (fgdtzero && clear_between_calls(clock,cadence)) held[i] = 0.f;
        if (time_weight(clock,end,ramp) > 0.f && held[i] != 0.f) value[i] += (Exact32)clock[1]*held[i];
        return;
    }
    if (weight > 0.f) value[i] += (Exact32)(clock[1] / tau) * weight * cuCrealf(filtered[i]);
}

#include "nudging_tick_bindings.metalh"
kernel void nudging_tick(constant NudgingTickResources& res [[buffer(0)]], constant NudgingTickParams& params [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device ulong *clock=(device ulong*)res.clock;
    device uint *error=(device uint*)res.error;
    device ulong *count=(device ulong*)res.count;
    if (gid >= uint(1)) return;

    if (*error == 0) {clock[0] = soft64_add(clock[0],clock[1]); *count += 1;}
}

#include "nudging_pack_fields_bindings.metalh"
kernel void nudging_pack_fields(constant NudgingPackFieldsResources& res [[buffer(0)]], constant NudgingPackFieldsParams& params [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const Exact32 *thp=(device const Exact32*)res.thp;
    device const Exact32 *thb=(device const Exact32*)res.thb;
    device const Exact32 *exner=(device const Exact32*)res.exner;
    device const Exact32 *php=(device const Exact32*)res.php;
    device const Exact32 *phb=(device const Exact32*)res.phb;
    device const Exact32 *ht=(device const Exact32*)res.ht;
    device const Exact32 *pbl=(device const Exact32*)res.pbl;
    device Exact32 *temp=(device Exact32*)res.temp;
    device Exact32 *oldt=(device Exact32*)res.oldt;
    device Exact32 *phi=(device Exact32*)res.phi;
    device Exact32 *oldphi=(device Exact32*)res.oldphi;
    device Exact32 *zu=(device Exact32*)res.zu;
    device Exact32 *zv=(device Exact32*)res.zv;
    device Exact32 *zt=(device Exact32*)res.zt;
    device Exact32 *zf=(device Exact32*)res.zf;
    device Exact32 *bu=(device Exact32*)res.bu;
    device Exact32 *bv=(device Exact32*)res.bv;
    device Exact32 *bt=(device Exact32*)res.bt;
    device Exact32 *bf=(device Exact32*)res.bf;
    int nz=params.nz;
    int ny=params.ny;
    int nx=params.nx;
    int terrain=params.terrain;
    int potential=params.potential;
    if (gid >= uint(max(max((nz+1)*ny*nx,nz*ny*(nx+1)),nz*(ny+1)*nx))) return;
    int q=int(gid);


 if(q<nz*ny*nx){int k=q/(nx*ny), j=q/nx%ny, i=q%nx;
  oldt[q]=temp[q]=(thp[q]+thb[terrain?q:k])*(potential ? 1.f : exner[q]);
  zt[q]=zm(php,phb,ht,k,j,i,ny,nx,terrain);
 }
 if(q<(nz+1)*ny*nx){int k=q/(nx*ny), j=q/nx%ny, i=q%nx;
  oldphi[q]=phi[q]=phi_at(php,phb,k,j,i,ny,nx,terrain); zf[q]=phi[q]/9.81f-ht[j*nx+i];}
 if(q<nz*ny*(nx+1)){int i=q%(nx+1),j=q/(nx+1)%ny,k=q/((nx+1)*ny);
  int a=max(0,i-1),b=min(nx-1,i); zu[q]=.5f*(zm(php,phb,ht,k,j,a,ny,nx,terrain)+zm(php,phb,ht,k,j,b,ny,nx,terrain));}
 if(q<nz*(ny+1)*nx){int i=q%nx,j=q/nx%(ny+1),k=q/(nx*(ny+1));
  int a=max(0,j-1),b=min(ny-1,j); zv[q]=.5f*(zm(php,phb,ht,k,a,i,ny,nx,terrain)+zm(php,phb,ht,k,b,i,ny,nx,terrain));}
 if(q<ny*nx)bt[q]=bf[q]=pbl[q];
 if(q<ny*(nx+1)){int i=q%(nx+1),j=q/(nx+1);bu[q]=fmaxf(pbl[j*nx+max(0,i-1)],pbl[j*nx+min(nx-1,i)]);}
 if(q<(ny+1)*nx){int i=q%nx,j=q/nx;bv[q]=fmaxf(pbl[max(0,j-1)*nx+i],pbl[min(ny-1,j)*nx+i]);}
}

#include "nudging_unpack_fields_bindings.metalh"
kernel void nudging_unpack_fields(constant NudgingUnpackFieldsResources& res [[buffer(0)]], constant NudgingUnpackFieldsParams& params [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device Exact32 *thp=(device Exact32*)res.thp;
    device Exact32 *php=(device Exact32*)res.php;
    device const Exact32 *exner=(device const Exact32*)res.exner;
    device const Exact32 *temp=(device const Exact32*)res.temp;
    device const Exact32 *oldt=(device const Exact32*)res.oldt;
    device const Exact32 *phi=(device const Exact32*)res.phi;
    device const Exact32 *oldphi=(device const Exact32*)res.oldphi;
    int nz=params.nz;
    int ny=params.ny;
    int nx=params.nx;
    int potential=params.potential;
    if (gid >= uint((nz+1)*ny*nx)) return;
    int q=int(gid);


 if(q<nz*ny*nx){Exact32 d=temp[q]-oldt[q];if(d!=0.f) thp[q]+=d/(potential ? 1.f : exner[q]);}
 if(q<(nz+1)*ny*nx){Exact32 d=phi[q]-oldphi[q];if(d!=0.f) php[q]+=d;}
}

#include "nudging_wrf_weights_bindings.metalh"
kernel void nudging_wrf_weights(constant NudgingWrfWeightsResources& res [[buffer(0)]], constant NudgingWrfWeightsParams& params [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const Exact32 *php=(device const Exact32*)res.php;
    device const Exact32 *phb=(device const Exact32*)res.phb;
    device const Exact32 *ht=(device const Exact32*)res.ht;
    device const Exact32 *pbl=(device const Exact32*)res.pbl;
    device Exact32 *weight=(device Exact32*)res.weight;
    int nz=params.nz;
    int ny=params.ny;
    int nx=params.nx;
    int terrain=params.terrain;
    int field=params.field;
    int grid_nudging=params.grid_nudging;
    int no_pbl=params.no_pbl;
    int zfac=params.zfac;
    int kz=params.kz;
    int dk=params.dk;
    if (gid >= uint((nx+(field==0))*(ny+(field==1)))) return;
    int q=int(gid);

 int sx=nx+(field==0), sy=ny+(field==1);

 int i=q%sx,j=q/sx;
 int ia=field==0?max(0,i-1):i, ib=min(nx-1,i);
 int ja=field==1?max(0,j-1):j, jb=min(ny-1,j);
 Exact32 ground=.5f*(ht[ja*nx+ia]+ht[jb*nx+ib]);
 Exact32 zpbl=.5f*(pbl[ja*nx+ia]+pbl[jb*nx+ib]);
 int kpbl=1;
 if(no_pbl)for(int k=0;k<nz;k++){
  Exact32 lo=.5f*(phi_at(php,phb,k,ja,ia,ny,nx,terrain)+phi_at(php,phb,k,jb,ib,ny,nx,terrain))/9.81f-ground;
  Exact32 hi=.5f*(phi_at(php,phb,k+1,ja,ia,ny,nx,terrain)+phi_at(php,phb,k+1,jb,ib,ny,nx,terrain))/9.81f-ground;
  if(field==3){if(zpbl>=lo)kpbl=k+1;}
  else if(zpbl>=lo&&zpbl<hi){kpbl=k+1;break;}
 }
 for(int k=1;k<=nz+(field==3);k++){
  Exact32 w=1.f;
  if(grid_nudging){
   if(no_pbl)w=k<=kpbl?0.f:(k==kpbl+1?.1f:1.f);
   if(zfac)w*=k<=kz?0.f:(k==kz+1?.1f:1.f);
  }else{
   int base=no_pbl?kpbl:(zfac?kz:0);
   if(no_pbl||zfac)w=fminf(1.f,fmaxf(0.f,(Exact32)(k-base)/(Exact32)dk));
  }
  weight[((k-1)*sy+j)*sx+i]=w;
 }
}
