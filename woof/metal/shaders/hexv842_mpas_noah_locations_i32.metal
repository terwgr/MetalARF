#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_noah_locations_i32_bindings.metalh"
kernel void hexv842_mpas_noah_locations_i32(constant Hexv842MpasNoahLocationsI32Resources &r [[buffer(0)]],constant Hexv842MpasNoahLocationsI32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
if(c)return; int n=p.n;

    for(int c=0;c<n;++c){
        if(r.routes[c]){
            int at=2*(r.routes[c]-1);
            r.state[at]=r.locations[2*c];r.state[at+1]=r.locations[2*c+1];
        }
    }
}
