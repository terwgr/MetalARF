// Source-ordered WSM6 preparation with physics_exner.cu.
// No CUDA fused stencil exists for these caller operations.
extern "C" __global__ void strict_wsm6_prep(
 const float* thb,const float* thp,const float* alt,const float* pressure,
 const float* phb,const float* php,float* t,float* rho,float* pii,
 float* dz,float* zi,float* saved,int nz,int nc,int thb3d,int phb3d) {
 int col=blockIdx.x*blockDim.x+threadIdx.x;
 if(col>=nc)return;
 for(int k=0;k<nz;k++){
  int idx=k*nc+col;float theta=__fadd_rn(thb[thb3d?idx:k],thp[idx]);
  t[idx]=theta;saved[idx]=theta;rho[idx]=__fdiv_rn(1.0f,alt[idx]);
  float ratio=__fdiv_rn(pressure[idx],P0);pii[idx]=gfk_pow(ratio,RCP);
 }
 for(int k=0;k<nz+1;k++){
  int idx=k*nc+col;zi[idx]=__fdiv_rn(__fadd_rn(phb[phb3d?idx:k],php[idx]),G);
 }
 for(int k=0;k<nz;k++){
  int idx=k*nc+col;float lo=zi[idx],hi=zi[idx+nc];
  dz[idx]=__fsub_rn(hi,lo);
 }
}
