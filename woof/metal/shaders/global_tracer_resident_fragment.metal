// Resident FP64 kernels for GridTracerTransport.advance.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_tracer_resident_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline ulong tr_min(ulong a, ulong b) { return soft64_less(a,b) ? a : b; }
static inline ulong tr_max(ulong a, ulong b) { return soft64_greater(a,b) ? a : b; }
static inline ulong tr_harmonic(ulong a, ulong b) {
    const ulong product=soft64_mul(a,b);
    return soft64_greater(product,0ul)
        ? soft64_div(soft64_mul(soft64_from_u32(2u),product),soft64_add(a,b)) : 0ul;
}
static inline uint tr_count(ulong courant, ulong limit) {
    uint n=soft64_to_u32_rz(soft64_div(courant,limit)).value;
    if (n == 0u) n=1u;
    else if (soft64_less(soft64_mul(soft64_from_u32(n),limit),courant)) ++n;
    return n;
}

kernel void global_tracer_pack_fragment(
    constant GlobalTracerPackFragmentResources &r [[buffer(0)]],
    constant GlobalTracerPackFragmentParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i<uint(p.count)) r.mass[i]=soft64_mul(r.tracer[i],r.density[i]);
}

kernel void global_tracer_flux_prepare_fragment(
    constant GlobalTracerFluxPrepareFragmentResources &r [[buffer(0)]],
    constant GlobalTracerFluxPrepareFragmentParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    const uint nl=uint(p.nlevel), ny=uint(p.nlatitude), nx=uint(p.nlongitude);
    const uint horizontal=ny*nx, count=nl*horizontal;
    if (i>=count) return;
    const uint j=(i/nx)%ny, x=i%nx, k=i/horizontal;
    const uint east=k*horizontal+j*nx+(x+1u)%nx;
    const ulong half_bits=0x3fe0000000000000ul;
    r.psi_x[i]=soft64_mul(soft64_mul(
        soft64_add(r.flux_u[i],r.flux_u[east]),half_bits),r.x_scale[j]);
    if (j+1u<ny) {
        const uint north=i+nx;
        const uint face=k*(ny-1u)*nx+j*nx+x;
        r.psi_y[face]=soft64_mul(soft64_mul(
            soft64_add(r.flux_v[i],r.flux_v[north]),half_bits),r.y_face[j]);
    }
    if (k+1u<nl) r.psi_z[i]=r.omega[i+horizontal];
}

kernel void global_tracer_flux_prepare_resident_fragment(
    constant GlobalTracerFluxPrepareResidentFragmentResources &r [[buffer(0)]],
    constant GlobalTracerFluxPrepareResidentFragmentParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    const uint nl=uint(p.nlevel), ny=uint(p.nlatitude), nx=uint(p.nlongitude);
    const uint horizontal=ny*nx, count=nl*horizontal;
    if (i>=count) return;
    const uint j=(i/nx)%ny, x=i%nx, k=i/horizontal;
    const uint east=k*horizontal+j*nx+(x+1u)%nx;
    const ulong half_bits=0x3fe0000000000000ul;
    r.psi_x[i]=soft64_mul(soft64_mul(
        soft64_add(r.flux_u[i],r.flux_u[east]),half_bits),r.x_scale[j]);
    if (j+1u<ny) {
        const uint north=i+nx;
        const uint face=k*(ny-1u)*nx+j*nx+x;
        r.psi_y[face]=soft64_mul(soft64_mul(
            soft64_add(r.flux_v[i],r.flux_v[north]),half_bits),r.y_face[j]);
    }
    if (k+1u<nl) r.psi_z[i]=r.omega[i+horizontal];
}

kernel void global_tracer_courant_row_fragment(
    constant GlobalTracerCourantRowFragmentResources &r [[buffer(0)]],
    constant GlobalTracerCourantRowFragmentParams &p [[buffer(1)]], uint row [[thread_position_in_grid]]) {
    if (row>=uint(p.nlatitude)) return;
    const uint nx=uint(p.nlongitude), nl=uint(p.nlevel), ny=uint(p.nlatitude);
    ulong largest=0ul;
    for(uint k=0;k<nl;++k) for(uint x=0;x<nx;++x) {
        const uint i=k*ny*nx+row*nx+x, next=k*ny*nx+row*nx+(x+1u)%nx;
        const ulong f=soft64_mul(r.psi[i],r.dt[0]);
        const ulong c0=soft64_div(f,r.density[i]);
        const ulong c1=soft64_div(soft64_negate_bits(f),r.density[next]);
        largest=tr_max(largest,tr_max(c0,c1));
    }
    r.row_max[row]=largest;
}

kernel void global_tracer_courant_row_finalize_fragment(
    constant GlobalTracerCourantRowFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalTracerCourantRowFinalizeFragmentParams &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if(tid!=0u) return;
    const ulong limit=r.limit[0];
    const uint nlat=uint(p.nlatitude), half_lat=nlat/2u;
    ulong maximum=0ul; uint south_end=0u, north_start=nlat;
    for(uint j=0;j<uint(p.nlatitude);++j) {
        const ulong value=r.row_max[j]; const uint n=tr_count(value,limit);
        maximum=tr_max(maximum,value);
        if(n>1u) { if(j<half_lat) south_end=j+1u; else north_start=min(north_start,j); }
    }
    uint south_steps=1u,north_steps=1u;
    for(uint j=0;j<south_end;++j) south_steps=max(south_steps,tr_count(r.row_max[j],limit));
    for(uint j=north_start;j<nlat;++j) north_steps=max(north_steps,tr_count(r.row_max[j],limit));
    uint maxsteps=max(south_steps,north_steps);
    for(uint j=0;j<nlat;++j) {
        const uint steps=(j<south_end)?south_steps:((j>=north_start)?north_steps:1u);
        r.row_steps[j]=int(steps);
    }
    const uint polar=south_end+(north_start<nlat?nlat-north_start:0u);
    r.summary[0]=maximum; r.summary[1]=soft64_from_u32(maxsteps);
    r.summary[2]=soft64_from_u32(polar);
}

kernel void global_tracer_courant_scalar_fragment(
    constant GlobalTracerCourantScalarFragmentResources &r [[buffer(0)]],
    constant GlobalTracerCourantScalarFragmentParams &p [[buffer(1)]], uint block [[thread_position_in_grid]]) {
    const uint start=block*uint(p.chunk), stop=min(uint(p.count),start+uint(p.chunk));
    ulong largest=0ul;
    const uint nx=uint(p.nlongitude), ny=uint(p.nlatitude);
    for(uint f=start;f<stop;++f) {
        uint lo,hi,axis_index;
        if(p.axis==1) {
            const uint row=f/(nx*(ny-1u)), rem=f%(nx*(ny-1u));
            const uint j=rem/nx, x=rem%nx; lo=row*ny*nx+j*nx+x; hi=lo+nx; axis_index=j;
        } else {
            const uint col=f%(ny*nx), k=f/(ny*nx); lo=k*ny*nx+col; hi=lo+ny*nx; axis_index=k;
        }
        const ulong fdt=soft64_mul(r.psi[f],r.dt[0]);
        ulong wlo=0x3ff0000000000000ul, whi=wlo;
        if(p.axis==1) { wlo=r.inv_width[axis_index]; whi=r.inv_width[axis_index+1u]; }
        const ulong c0=soft64_div(soft64_mul(fdt,wlo),r.density[lo]);
        const ulong c1=soft64_div(soft64_mul(soft64_negate_bits(fdt),whi),r.density[hi]);
        largest=tr_max(largest,tr_max(c0,c1));
    }
    r.partial[block]=largest;
}

kernel void global_tracer_courant_scalar_finalize_fragment(
    constant GlobalTracerCourantScalarFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalTracerCourantScalarFinalizeFragmentParams &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if(tid!=0u) return;
    ulong maximum=0ul;
    for(uint i=0;i<uint(p.npartial);++i) maximum=tr_max(maximum,r.partial[i]);
    r.summary[0]=maximum;
    r.summary[1]=soft64_from_u32(tr_count(maximum,r.limit[0]));
}

static inline ulong tr_zonal_flux(device const ulong *mass, device const ulong *density,
    device const ulong *psi,uint tracer,uint cells,uint base,uint lon,uint nx,ulong dt) {
    const uint prev=base+(lon+nx-1u)%nx, next=base+(lon+1u)%nx, next2=base+(lon+2u)%nx;
    const uint t=tracer*cells;
    const ulong qm2=soft64_div(mass[t+prev],density[prev]);
    const ulong q0=soft64_div(mass[t+base+lon],density[base+lon]);
    const ulong qp1=soft64_div(mass[t+next],density[next]);
    const ulong qp2=soft64_div(mass[t+next2],density[next2]);
    const ulong slope=tr_harmonic(soft64_sub(q0,qm2),soft64_sub(qp1,q0));
    const ulong slope1=tr_harmonic(soft64_sub(qp1,q0),soft64_sub(qp2,qp1));
    const ulong fdt=soft64_mul(psi[base+lon],dt);
    const ulong clo=soft64_div(fdt,density[base+lon]);
    const ulong chi=soft64_div(soft64_negate_bits(fdt),density[next]);
    const ulong half_bits=0x3fe0000000000000ul, one=0x3ff0000000000000ul;
    const bool positive=soft64_greater_equal(psi[base+lon],0ul);
    ulong face=positive?soft64_add(q0,soft64_mul(slope,soft64_mul(half_bits,soft64_sub(one,clo))))
                       :soft64_add(qp1,soft64_mul(slope1,soft64_mul(soft64_negate_bits(half_bits),soft64_sub(one,chi))));
    const ulong up=positive?q0:qp1;
    face=tr_min(tr_max(face,tr_min(q0,qp1)),tr_min(tr_max(q0,qp1),soft64_mul(soft64_from_u32(2u),up)));
    return soft64_mul(fdt,face);
}

static inline ulong tr_zonal_density_flux(device const ulong *psi,
    uint base,uint lon,uint nx,ulong dt) {
    const uint face=base+lon;
    return soft64_mul(psi[face],dt);
}

kernel void global_tracer_zonal_rows_fragment(
    constant GlobalTracerZonalRowsFragmentResources &r [[buffer(0)]],
    constant GlobalTracerZonalRowsFragmentParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    const uint cells=uint(p.nlevel)*uint(p.nlatitude)*uint(p.nlongitude);
    if(i>=cells)return;
    const uint nx=uint(p.nlongitude), ny=uint(p.nlatitude), lon=i%nx;
    const uint row=(i/nx)%ny, base=i-lon, n=max(1u,uint(r.row_steps[row]));
    if(uint(p.iteration)>=n) { r.new_density[i]=r.density[i];
        for(uint t=0;t<uint(p.ntracer);++t)r.new_mass[t*cells+i]=r.mass[t*cells+i]; return; }
    const ulong sub=soft64_div(r.dt[0],soft64_from_u32(n));
    const ulong here=tr_zonal_density_flux(r.psi,base,lon,nx,sub);
    const ulong west=tr_zonal_density_flux(r.psi,base,(lon+nx-1u)%nx,nx,sub);
    r.new_density[i]=soft64_sub(r.density[i],soft64_sub(here,west));
    for(uint t=0;t<uint(p.ntracer);++t) {
        const ulong fh=tr_zonal_flux(r.mass,r.density,r.psi,t,cells,base,lon,nx,sub);
        const ulong fw=tr_zonal_flux(r.mass,r.density,r.psi,t,cells,base,(lon+nx-1u)%nx,nx,sub);
        r.new_mass[t*cells+i]=soft64_sub(r.mass[t*cells+i],soft64_sub(fh,fw));
    }
}

static inline ulong tr_walled_flux(device const ulong *mass,
    device const ulong *density,device const ulong *psi,
    device const ulong *centre,device const ulong *face,
    device const ulong *inv_width,uint tracer,uint cells,uint nlev,
    uint nlat,uint nlon,uint level,uint face_row,uint lon,ulong sub) {
    const uint layer=nlat*nlon;
    const uint lo=level*layer+face_row*nlon+lon, hi=lo+nlon;
    const uint pface=(level*(nlat-1u)+face_row)*nlon+lon;
    const uint t=tracer*cells;
    const ulong qlo=soft64_div(mass[t+lo],density[lo]);
    const ulong qhi=soft64_div(mass[t+hi],density[hi]);
    ulong slope_lo=0ul,slope_hi=0ul;
    const ulong dface=soft64_sub(centre[face_row+1u],centre[face_row]);
    const ulong secant=soft64_div(soft64_sub(qhi,qlo),dface);
    if(face_row>0u) {
        const uint prev=lo-nlon;
        const ulong qprev=soft64_div(mass[t+prev],density[prev]);
        const ulong dprev=soft64_sub(centre[face_row],centre[face_row-1u]);
        slope_lo=tr_harmonic(soft64_div(soft64_sub(qlo,qprev),dprev),secant);
    }
    if(face_row+2u<nlat) {
        const uint next=hi+nlon;
        const ulong qnext=soft64_div(mass[t+next],density[next]);
        const ulong dnext=soft64_sub(centre[face_row+2u],centre[face_row+1u]);
        slope_hi=tr_harmonic(secant,soft64_div(soft64_sub(qnext,qhi),dnext));
    }
    const ulong fdt=soft64_mul(psi[pface],sub);
    const ulong one=0x3ff0000000000000ul;
    const ulong clo=soft64_div(soft64_mul(fdt,inv_width[face_row]),density[lo]);
    const ulong chi=soft64_div(soft64_mul(soft64_negate_bits(fdt),
                                           inv_width[face_row+1u]),density[hi]);
    const ulong wlo=soft64_mul(soft64_sub(face[face_row],centre[face_row]),
                                soft64_sub(one,clo));
    const ulong whi=soft64_mul(soft64_sub(face[face_row],centre[face_row+1u]),
                                soft64_sub(one,chi));
    const bool positive=soft64_greater_equal(psi[pface],0ul);
    ulong value=positive?soft64_add(qlo,soft64_mul(slope_lo,wlo))
                        :soft64_add(qhi,soft64_mul(slope_hi,whi));
    const ulong up=positive?qlo:qhi;
    value=tr_min(tr_max(value,tr_min(qlo,qhi)),
                 tr_min(tr_max(qlo,qhi),soft64_mul(soft64_from_u32(2u),up)));
    return soft64_mul(fdt,value);
}

kernel void global_tracer_walled_resident_fragment(
    constant GlobalTracerWalledResidentFragmentResources &r [[buffer(0)]],
    constant GlobalTracerWalledResidentFragmentParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    const uint nlev=uint(p.nouter), nlat=uint(p.axis_length), nlon=uint(p.stride);
    const uint cells=nlev*nlat*nlon;
    if(i>=cells)return;
    const uint lon=i%nlon,row=(i/nlon)%nlat,level=i/(nlat*nlon);
    ulong rho=r.density[i];
    if(row+1u<nlat) {
        const uint f=(level*(nlat-1u)+row)*nlon+lon;
        rho=soft64_sub(rho,soft64_mul(soft64_mul(r.psi[f],r.sub[0]),r.inv_width[row]));
    }
    if(row>0u) {
        const uint f=(level*(nlat-1u)+row-1u)*nlon+lon;
        rho=soft64_add(rho,soft64_mul(soft64_mul(r.psi[f],r.sub[0]),r.inv_width[row]));
    }
    r.new_density[i]=rho;
    for(uint t=0;t<uint(p.ntracer);++t) {
        ulong value=r.mass[t*cells+i];
        if(row+1u<nlat) {
            const ulong flux=tr_walled_flux(r.mass,r.density,r.psi,r.centre,r.face,
                r.inv_width,t,cells,nlev,nlat,nlon,level,row,lon,r.sub[0]);
            value=soft64_sub(value,soft64_mul(flux,r.inv_width[row]));
        }
        if(row>0u) {
            const ulong flux=tr_walled_flux(r.mass,r.density,r.psi,r.centre,r.face,
                r.inv_width,t,cells,nlev,nlat,nlon,level,row-1u,lon,r.sub[0]);
            value=soft64_add(value,soft64_mul(flux,r.inv_width[row]));
        }
        r.new_mass[t*cells+i]=value;
    }
}

kernel void global_tracer_pressure_coords_fragment(
    constant GlobalTracerPressureCoordsFragmentResources &r [[buffer(0)]],
    constant GlobalTracerPressureCoordsFragmentParams &p [[buffer(1)]], uint col [[thread_position_in_grid]]) {
    if(col>=uint(p.ncolumn))return;
    ulong lower=r.p_top[0];
    r.p_half[col]=lower;
    for(uint k=0;k<uint(p.nlevel);++k) {
        const uint i=k*uint(p.ncolumn)+col; const ulong upper=soft64_add(lower,r.density[i]);
        r.p_half[(k+1u)*uint(p.ncolumn)+col]=upper;
        r.p_full[i]=soft64_sqrt(soft64_mul(lower,upper)); lower=upper;
    }
}

static inline ulong tr_vertical_flux(device const ulong *mass,device const ulong *density,
    device const ulong *psi,device const ulong *p_full,device const ulong *p_half,
    uint tracer,uint cells,uint nlev,uint col,uint k,ulong sub) {
    const uint lo=k*cells/nlev+col, hi=lo+cells/nlev, t=tracer*cells;
    const ulong qlo=soft64_div(mass[t+lo],density[lo]), qhi=soft64_div(mass[t+hi],density[hi]);
    ulong slo=0ul,shi=0ul;
    if(k>0u) { const ulong qprev=soft64_div(mass[t+lo-cells/nlev],density[lo-cells/nlev]);
        slo=tr_harmonic(soft64_div(soft64_sub(qlo,qprev),soft64_sub(p_full[lo],p_full[lo-cells/nlev])),
                        soft64_div(soft64_sub(qhi,qlo),soft64_sub(p_full[hi],p_full[lo]))); }
    if(k+1u<uint(nlev)-1u) { const ulong qnext=soft64_div(mass[t+hi+cells/nlev],density[hi+cells/nlev]);
        shi=tr_harmonic(soft64_div(soft64_sub(qhi,qlo),soft64_sub(p_full[hi],p_full[lo])),
                        soft64_div(soft64_sub(qnext,qhi),soft64_sub(p_full[hi+cells/nlev],p_full[hi]))); }
    const uint f=k*cells/nlev+col; const ulong fdt=soft64_mul(psi[f],sub);
    const ulong clo=soft64_div(fdt,density[lo]), chi=soft64_div(soft64_negate_bits(fdt),density[hi]);
    const ulong one=0x3ff0000000000000ul, pos=p_half[(k+1u)*uint(cells/nlev)+col];
    const ulong wlo=soft64_mul(soft64_sub(pos,p_full[lo]),soft64_sub(one,clo));
    const ulong whi=soft64_mul(soft64_sub(pos,p_full[hi]),soft64_sub(one,chi));
    const bool positive=soft64_greater_equal(psi[f],0ul);
    ulong face=positive?soft64_add(qlo,soft64_mul(slo,wlo)):soft64_add(qhi,soft64_mul(shi,whi));
    const ulong up=positive?qlo:qhi;
    face=tr_min(tr_max(face,tr_min(qlo,qhi)),tr_min(tr_max(qlo,qhi),soft64_mul(soft64_from_u32(2u),up)));
    return soft64_mul(fdt,face);
}

kernel void global_tracer_vertical_fragment(
    constant GlobalTracerVerticalFragmentResources &r [[buffer(0)]],
    constant GlobalTracerVerticalFragmentParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    const uint nlev=uint(p.nlevel), ncol=uint(p.ncolumn), cells=nlev*ncol;
    if(i>=cells)return;
    const uint k=i/ncol,col=i%ncol; ulong updated=r.density[i];
    if(k+1u<nlev) updated=soft64_sub(updated,soft64_mul(r.psi[k*ncol+col],r.sub[0]));
    if(k>0u) updated=soft64_add(updated,soft64_mul(r.psi[(k-1u)*ncol+col],r.sub[0]));
    r.new_density[i]=updated;
    for(uint t=0;t<uint(p.ntracer);++t) {
        ulong m=r.mass[t*cells+i];
        if(k+1u<nlev)m=soft64_sub(m,tr_vertical_flux(r.mass,r.density,r.psi,r.p_full,r.p_half,t,cells,nlev,col,k,r.sub[0]));
        if(k>0u)m=soft64_add(m,tr_vertical_flux(r.mass,r.density,r.psi,r.p_full,r.p_half,t,cells,nlev,col,k-1u,r.sub[0]));
        r.new_mass[t*cells+i]=m;
    }
}

kernel void global_tracer_floor_fragment(
    constant GlobalTracerFloorFragmentResources &r [[buffer(0)]],
    constant GlobalTracerFloorFragmentParams &p [[buffer(1)]], uint block [[thread_position_in_grid]]) {
    const uint ncell=uint(p.nlevel)*uint(p.nlatitude)*uint(p.nlongitude);
    const uint tracer=block/uint(p.nblock), part=block%uint(p.nblock);
    const uint start=part*uint(p.chunk), stop=min(ncell,start+uint(p.chunk));
    ulong neg=0ul,pos=0ul;
    for(uint i=start;i<stop;++i) {
        const uint index=tracer*ncell+i; const ulong q=soft64_div(r.mass[index],r.density[i]);
        const uint row=(i/uint(p.nlongitude))%uint(p.nlatitude);
        ulong value=q;
        if(soft64_less(q,0ul)) { neg=soft64_add(neg,soft64_mul(soft64_mul(q,r.density[i]),r.row_weight[row*uint(p.nlongitude)+(i%uint(p.nlongitude))])); value=0ul; }
        else pos=soft64_add(pos,soft64_mul(soft64_mul(q,r.density[i]),r.row_weight[row*uint(p.nlongitude)+(i%uint(p.nlongitude))]));
        r.mass[index]=value;
    }
    r.negative[block]=neg;r.positive[block]=pos;
}

kernel void global_tracer_floor_finalize_fragment(
    constant GlobalTracerFloorFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalTracerFloorFinalizeFragmentParams &p [[buffer(1)]], uint tracer [[thread_position_in_grid]]) {
    if(tracer>=uint(p.ntracer))return;
    ulong neg=0ul,pos=0ul; const uint base=tracer*uint(p.nblock);
    for(uint i=0;i<uint(p.nblock);++i) { neg=soft64_add(neg,r.negative[base+i]); pos=soft64_add(pos,r.positive[base+i]); }
    r.neg_total[tracer]=neg;r.pos_total[tracer]=pos;
}
