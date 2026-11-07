"""Emit the source descriptor in Metal C++14 and a small instantiation kernel."""
from itertools import product
from pathlib import Path
p=Path(__file__).resolve().parent
s='''#include <metal_stdlib>
using namespace metal;
template<int R> struct MynnIndices { int value[R]; };
struct MynnRange { int lo, hi; };
struct MynnArena {
    device uchar *data;
    ulong offset;
    device void *allocate(ulong bytes, int itemsize) {
        offset = (offset+7ul)&~7ul;
        device void *result = data+offset;
        offset += bytes;
        return result;
    }
};
template<class T, int R> struct MynnArray {
    device T *data = nullptr;
    int lo[R],hi[R],stride[R];
    bool present() const { return data != nullptr; }
    int extent(int d) const { return hi[d]-lo[d]+1; }
    int size() const { int n=1; for (int d=0;d<R;++d) n*=extent(d); return n; }
    void rebase(MynnIndices<R> lower) { for (int d=0;d<R;++d) { int n=extent(d); lo[d]=lower.value[d]; hi[d]=lo[d]+n-1; } }
    device T &linear(int index) const {
        int offset=0;
        for (int d=0;d<R;++d) { offset+=(index%extent(d))*stride[d]; index/=extent(d); }
        return data[offset];
    }
'''
for rank in range(1,5):
    args=', '.join('int i%d'%i for i in range(rank))
    index='+'.join('(i%d-lo[%d])*stride[%d]'%(i,i,i) for i in range(rank))
    s+=f'    device T &operator()({args}) const {{ return data[{index}]; }}\n'
for rank in range(1,5):
    for mask in product([False,True],repeat=rank):
        n=sum(mask)
        if not n: continue
        args=', '.join(('MynnRange' if k else 'int')+f' i{d}' for d,k in enumerate(mask))
        offsets='+'.join(f'(i{d}.lo-lo[{d}])*stride[{d}]' if k else f'(i{d}-lo[{d}])*stride[{d}]' for d,k in enumerate(mask))
        s+=f'    MynnArray<T,{n}> slice({args}) const {{ MynnArray<T,{n}> out; out.data=data+{offsets}; '
        od=0
        for d,k in enumerate(mask):
            if k:
                s+=f'out.lo[{od}]=1; out.hi[{od}]=i{d}.hi-i{d}.lo+1; out.stride[{od}]=stride[{d}]; '
                od+=1
        s+='return out; }\n'
s+='''    template<int N> MynnArray<T,N> sequence(MynnIndices<N> lower,MynnIndices<N> upper) const {
        MynnArray<T,N> out; out.data=data; int step=1;
        for(int d=0;d<N;++d) { out.lo[d]=lower.value[d];out.hi[d]=upper.value[d];out.stride[d]=step;step*=out.extent(d); }
        return out;
    }
};
template<class T,int R,int TILE> MynnArray<T,R> mynn_array(thread MynnArena &arena,MynnIndices<R> lo,MynnIndices<R> hi) {
    static_assert(sizeof(T)==4,"MYNN word size");
    static_assert(TILE==1,"contiguous serial column");
    MynnArray<T,R> a; int count=1;
    for(int d=0;d<R;++d) { a.lo[d]=lo.value[d];a.hi[d]=hi.value[d];a.stride[d]=count;count*=a.extent(d); }
    a.data=static_cast<device T *>(arena.allocate(sizeof(T)*count,sizeof(T)));
    return a;
}
template<class T> struct MynnOptional {
    T value{}; bool supplied=false;
    MynnOptional()=default;
    MynnOptional(T v):value(v),supplied(true) {}
    operator T() const { return value; }
    bool present() const { return supplied; }
};
'''
(p/'array_support.metalh').write_text(s)
probe='''#include "array_support.metalh"
kernel void array_probe(device uchar *scratch [[buffer(0)]],device int *out [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
    if(gid>=1) return;
    MynnArena arena{scratch,0};
    ulong saved=arena.offset;
    auto a=mynn_array<int,2,1>(arena,{2,3},{4,5});
    a(3,4)=43;
    auto b=a.slice(MynnRange{2,4},4);
    auto c=a.slice(3,MynnRange{3,5});
    auto d=a.slice(MynnRange{2,4},MynnRange{3,5});
    b.rebase({2});
    MynnOptional<bool> missing, yes(true);
    out[0]=a.size();out[1]=a(3,4);out[2]=b(3);out[3]=c(2);out[4]=d(2,2);
    out[5]=missing.present();out[6]=yes.present();out[7]=bool(yes);
    arena.offset=saved;out[8]=int(arena.offset);
}
'''
(p/'array_probe.metal').write_text(probe)
