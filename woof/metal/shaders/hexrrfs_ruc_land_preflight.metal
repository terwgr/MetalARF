#include <metal_stdlib>
using namespace metal;
#include "hexrrfs_ruc_land_preflight_bindings.metalh"
// RRFS ruc_land_tile_preflight. First masked lake terminates the native tile;
// its prefix executes only preparation, no full column runs for that tile.
kernel void hexrrfs_ruc_land_preflight(constant HexrrfsRucLandPreflightResources& r [[buffer(0)]],constant HexrrfsRucLandPreflightParams& p [[buffer(1)]],uint gid [[thread_position_in_grid]]){
 if(gid>=1u)return;
 r.stop[0]=p.solve;r.stop[1]=0;
 if(p.lakemodel!=1)return;
 for(int c=0;c<p.solve;++c)if(r.lakemask[c]==1.f){r.stop[0]=c;r.stop[1]=1;return;}
}
