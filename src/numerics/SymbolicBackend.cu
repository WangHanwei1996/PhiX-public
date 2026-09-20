#include "SymbolicInternal.h"
#include "core/CudaCheck.h"
#include <cuda.h>
#include <nvrtc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace PhiX::scheme::detail {
namespace {
template<DerivativeMethod M> DerivativeMethod selectDerivative(){return M;}
}
const std::vector<Entry<DerivativeFactory>>& secondDerivativeFactories(){
    static const std::vector<Entry<DerivativeFactory>> entries={
        {{"CD2",2,1,true,Grid2D::Rectangular,"CD2","direct diagonal or mixed second derivative"},&selectDerivative<DerivativeMethod::CD2>},
        {{"CD4",4,2,true,Grid2D::Rectangular,"CD4","direct diagonal or tensor-product mixed second derivative"},&selectDerivative<DerivativeMethod::CD4>},
        {{"CD6",6,3,true,Grid2D::Rectangular,"CD6","direct diagonal or tensor-product mixed second derivative"},&selectDerivative<DerivativeMethod::CD6>}};
    return entries;
}
}
namespace PhiX::numerics::symbolic::detail {
namespace {
std::string literal(double v){
    std::ostringstream s;s<<std::setprecision(17)<<std::scientific<<v;return "R("+s.str()+")";
}
std::vector<std::pair<int,double>> weights1D(int order,int derivative){
    if(derivative==1){
        if(order==2)return {{-1,-.5},{1,.5}};
        if(order==4)return {{-2,1./12},{-1,-2./3},{1,2./3},{2,-1./12}};
        return {{-3,-1./60},{-2,3./20},{-1,-3./4},{1,3./4},{2,-3./20},{3,1./60}};
    }
    if(order==2)return {{0,-2},{-1,1},{1,1}};
    if(order==4)return {{0,-2.5},{-1,4./3},{1,4./3},{-2,-1./12},{2,-1./12}};
    return {{0,-49./18},{-1,1.5},{1,1.5},{-2,-3./20},{2,-3./20},{-3,1./90},{3,1./90}};
}
std::vector<Weight> stencil(const Node& n,const std::string& name){
    double hx=n.field->mesh.d[0],hy=n.field->mesh.d[1];
    std::vector<Weight> out;
    if(name=="Iso9"){
        if(n.x<0){
            if(std::abs(hx-hy)>1e-12*std::max(hx,hy))throw std::invalid_argument("symbolic lap(Iso9) requires square cells");
            out.push_back({0,0,-20/(6*hx*hx)});
            for(auto o:std::vector<std::pair<int,int>>{{-1,0},{1,0},{0,-1},{0,1}})out.push_back({o.first,o.second,4/(6*hx*hx)});
            for(int x:{-1,1})for(int y:{-1,1})out.push_back({x,y,1/(6*hx*hx)});
        }else{
            if(n.x+n.y!=1)throw std::invalid_argument("Iso9 is not a direct Hessian scheme");
            for(int normal:{-1,1})for(int t:{-1,0,1})
                out.push_back({n.x?normal:t,n.x?t:normal,normal*(t==0?4.:1.)/(12*(n.x?hx:hy))});
        }
        return out;
    }
    int order=2;
    if(n.x+n.y==2 && n.x>=0){
        auto method=scheme::detail::lookup(scheme::detail::secondDerivativeFactories(),name,"symbolic second derivative").factory();
        order=method==scheme::detail::DerivativeMethod::CD2?2:method==scheme::detail::DerivativeMethod::CD4?4:6;
    }else if(name=="CD4")order=4;
    else if(name=="CD6")order=6;
    else if(name!="CD2" && !(n.x<0 && name=="Iso27"))throw std::invalid_argument("unsupported expanded stencil: "+name);
    if(n.x==1 && n.y==1){
        for(auto x:weights1D(order,1))for(auto y:weights1D(order,1))
            out.push_back({x.first,y.first,x.second*y.second/(hx*hy)});
    }else{
        auto append=[&](int axis,int d){double h=axis?hy:hx;
            for(auto w:weights1D(order,d))out.push_back({axis?0:w.first,axis?w.first:0,w.second/(d==1?h:h*h)});};
        if(n.x<0){append(0,2);append(1,2);}else append(n.x?0:1,n.x+n.y);
    }
    return out;
}
Real evaluateStencil(const Primitive& p,const Real* data,int c,int sx){
    Real v=0;for(const auto& w:p.weights)v+=Real(w.value)*data[c+w.x+sx*w.y];return v;
}
// Resolve translated expressions after their composite templates are selected.
// In particular, shifting a derivative shifts its complete stencil, not an
// uninitialized derivative auxiliary. This also exposes the exact halo union.
Expr resolveSamples(Expr expression,const Schemes& schemes,const ScalarField& layout,OperationInfo& info){
    std::map<std::tuple<const Node*,int,int>,Expr> memo;
    std::function<Expr(Expr,int,int)> visit=[&](Expr e,int x,int y){
        auto cache=std::make_tuple(e.node().get(),x,y);
        if(auto it=memo.find(cache);it!=memo.end())return it->second;
        const auto& n=*e.node();Expr r;
        if(n.op==Op::Shift)r=visit(n.args[0],x+n.x,y+n.y);
        else if(n.op==Op::Field || n.op==Op::Parameter || n.op==Op::Sample){
            int dx=x+(n.op==Op::Sample?n.x:0),dy=y+(n.op==Op::Sample?n.y:0);
            r=(dx || dy)?make(Op::Sample,{},0,n.field,dx,dy):make(Op::Field,{},0,n.field);
        }else if(n.op==Op::Coordinate)
            r=e+((n.x==0?x:y)*layout.mesh.d[n.x]);
        else if(n.op==Op::Jet && (x || y)){
            auto choice=n.x+n.y==1 && n.x>=0?schemes.gradient(n.field->name,n.x?0:1):schemes.select(section(n),key(n));
            info.schemes.push_back(choice.describe());r=0;
            for(auto w:stencil(n,choice.name))
                r=r+w.value*make(Op::Sample,{},0,n.field,x+w.x,y+w.y);
        }else{
            std::vector<Expr> args;for(auto a:n.args)args.push_back(visit(a,x,y));
            r=make(n.op,args,n.value,n.field,n.x,n.y,n.name);
        }
        memo.emplace(cache,r);return r;
    };
    return canonicalize(visit(expression,0,0));
}
std::string stencilSource(const Primitive& p,const std::string& input,int sx){
    std::string s="R(0)";
    for(const auto& w:p.weights)s+=" + "+literal(w.value)+"*"+input+"[c+("+std::to_string(w.x+sx*w.y)+")]";
    return "("+s+")";
}
void driver(CUresult r,const char* where){
    if(r==CUDA_SUCCESS)return;
    const char* text=nullptr;cuGetErrorString(r,&text);
    throw std::runtime_error(std::string(where)+": "+(text?text:"CUDA driver error"));
}
void rtc(nvrtcResult r,const char* where){
    if(r!=NVRTC_SUCCESS)throw std::runtime_error(std::string(where)+": "+nvrtcGetErrorString(r));
}
std::string cellIndex(const Plan& p){
    return "int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>="+std::to_string(p.nx*p.ny)+") return;\n"
        "int c=(tid%"+std::to_string(p.nx)+"+"+std::to_string(p.ghost)+")+"+std::to_string(p.sx)+
        "*((tid/"+std::to_string(p.nx)+"+"+std::to_string(p.ghost)+")+"+std::to_string(p.sy*p.ghost)+");\n";
}
class Emitter {
    const Plan& p_;
    Fusion mode_;
    std::map<const Node*,std::string> values_;
    std::map<const ScalarField*,int> fields_;
    std::map<const Node*,int> primitives_;
    int next_=0;
    std::ostringstream out_;
public:
    Emitter(const Plan& p,Fusion mode):p_(p),mode_(mode){
        for(std::size_t i=0;i<p.fields.size();++i)fields_[p.fields[i]]=int(i);
        for(std::size_t i=0;i<p.primitives.size();++i)primitives_[p.primitives[i].node]=int(i);
    }
    std::string emit(Expr e){
        const auto& n=*e.node();auto it=values_.find(&n);if(it!=values_.end())return it->second;
        if(n.op==Op::Constant)return literal(n.value);
        std::string expression;
        if(n.op==Op::Where || n.op==Op::And || n.op==Op::Or){
            auto condition=emit(n.args[0]);std::string name="v"+std::to_string(next_++);
            out_<<"R "<<name<<"; if ("<<condition<<" != R(0)) {\n";
            auto outer=values_;
            std::string yes=n.op==Op::Where?emit(n.args[1]):n.op==Op::And?"("+emit(n.args[1])+" != R(0))":"R(1)";
            out_<<name<<"="<<yes<<";\n} else {\n";values_=outer;
            std::string no=n.op==Op::Where?emit(n.args[2]):n.op==Op::Or?"("+emit(n.args[1])+" != R(0))":"R(0)";
            out_<<name<<"="<<no<<";\n}\n";values_=outer;values_[&n]=name;return name;
        }
        if(n.op==Op::Jet){
            int i=primitives_.at(&n);
            expression=mode_==Fusion::Off?"d"+std::to_string(i)+"[c]":stencilSource(p_.primitives[i],"f"+std::to_string(fields_.at(n.field)),p_.sx);
        }else if(n.op==Op::Field || n.op==Op::Parameter || n.op==Op::Sample)
            expression="f"+std::to_string(fields_.at(n.field))+"[c+("+std::to_string(n.op==Op::Sample?n.x+n.y*p_.sx:0)+")]";
        else if(n.op==Op::Coordinate){
            expression=literal(p_.layout->mesh.coord(n.x,0))+"+"+literal(p_.layout->mesh.d[n.x])+"*R(tid"+
                (n.x==0?"%":"/")+std::to_string(p_.nx)+")";
        }else{
            std::vector<std::string>a;for(auto child:n.args)a.push_back(emit(child));
            const std::map<Op,std::string> binary={{Op::Add,"+"},{Op::Mul,"*"},{Op::Divide,"/"},
                {Op::Less,"<"},{Op::LessEqual,"<="},{Op::Greater,">"},{Op::GreaterEqual,">="}};
            const std::map<Op,std::string> unary={{Op::Sin,"sin"},{Op::Cos,"cos"},{Op::Tanh,"tanh"},
                {Op::Exp,"exp"},{Op::Log,"log"},{Op::Sqrt,"sqrt"},{Op::Abs,"fabs"}};
            if(auto b=binary.find(n.op);b!=binary.end())expression="("+a[0]+b->second+a[1]+")";
            else if(auto u=unary.find(n.op);u!=unary.end())expression=u->second+"("+a[0]+")";
            else if(n.op==Op::Negate)expression="(-"+a[0]+")";
            else if(n.op==Op::Atan2)expression="atan2("+a[0]+","+a[1]+")";
            else if(n.op==Op::Hypot)expression="hypot("+a[0]+","+a[1]+")";
            else if(n.op==Op::Pow)expression=n.value==2?"("+a[0]+"*"+a[0]+")":"pow("+a[0]+","+literal(n.value)+")";
            else throw std::logic_error("unexpanded node reached CUDA emitter");
        }
        auto name="v"+std::to_string(next_++);out_<<"const R "<<name<<"="<<expression<<";\n";values_[&n]=name;return name;
    }
    std::string body(){auto result=emit(p_.expression);return out_.str()+"rhs[c] += coeff*"+result+";\n";}
};
}
std::shared_ptr<Plan> bind(Expr expression,const std::string& name,const Schemes& schemes,
                         const ScalarField& layout,ScalarField* state,bool checkHalos){
    if(layout.mesh.dim!=2)throw std::invalid_argument("symbolic discretization supports 2D only");
    check::checkCartesian(layout.mesh,"symbolic discretization");
    auto p=std::make_shared<Plan>();p->expression=expression;p->name=name;p->layout=&layout;p->state=state;
    p->nx=layout.mesh.n[0];p->ny=layout.mesh.n[1];p->sx=layout.storedDims[0];p->sy=layout.storedDims[1];
    p->ghost=layout.ghost;p->storedSize=layout.storedSize;p->info.complete=true;
    expression=resolveSamples(lowerComposites(expression,schemes,layout,p->info),schemes,layout,p->info);
    p->expression=expression;
    std::map<std::string,const ScalarField*> names;
    for(auto e:nodes(expression)){
        const auto& n=*e.node();
        if(n.field){
            check::checkSameMesh(layout,*n.field,"symbolic field layout");
            if(n.field->ghost!=layout.ghost)throw std::invalid_argument("symbolic fields need matching ghost layouts");
            if(auto it=names.find(n.field->name);it!=names.end() && it->second!=n.field)
                throw std::invalid_argument("symbolic inventory: distinct fields have the same name: "+n.field->name);
            names[n.field->name]=n.field;
            if(std::find(p->fields.begin(),p->fields.end(),n.field)==p->fields.end())p->fields.push_back(n.field);
            p->info.read(n.field);
        }
        if(n.op==Op::Jet){
            auto selection=n.x+n.y==1 && n.x>=0?schemes.gradient(n.field->name,n.x?0:1):schemes.select(section(n),key(n));
            Primitive primitive{&n,n.field,selection,stencil(n,selection.name)};
            for(const auto& w:primitive.weights){primitive.halo=std::max({primitive.halo,std::abs(w.x),std::abs(w.y)});primitive.corners|=w.x!=0 && w.y!=0;}
            if(checkHalos)check::checkGhost(*n.field,primitive.halo,("symbolic "+key(n)).c_str());
            p->info.read(n.field,primitive.halo,primitive.corners);p->info.schemes.push_back(selection.describe());
            p->primitives.push_back(std::move(primitive));
        }
        if(n.op==Op::Sample){
            const int halo=std::max(std::abs(n.x),std::abs(n.y));
            if(checkHalos)check::checkGhost(*n.field,halo,"composite sample");
            p->info.read(n.field,halo,n.x!=0 && n.y!=0);
        }
        if(n.op==Op::Derivative || n.op==Op::Laplacian || n.op==Op::Divergence || n.op==Op::Partial || n.op==Op::Alias)
            throw std::logic_error("symbolic discretization requires a fully expanded expression");
    }
    if(state){
        check::checkSameMesh(*state,layout,"symbolic equation output");
        auto s=schemes.select("ddt","ddt("+state->name+")");
        if(s.name!="EULER")throw std::invalid_argument("symbolic equations currently support Euler only");p->info.schemes.push_back(s.describe());
    }
    auto all=p->fields;if(std::find(all.begin(),all.end(),&layout)==all.end())all.push_back(&layout);
    for(auto f:all){
        const auto mesh=f->mesh;int ghost=f->ghost;auto size=f->storedSize;
        std::array<int,3> dims{f->storedDims[0],f->storedDims[1],f->storedDims[2]};
        p->layoutChecks.push_back([f,mesh,ghost,size,dims]{
            if(!check::sameMeshGeometry(mesh,f->mesh) || ghost!=f->ghost || size!=f->storedSize || !std::equal(dims.begin(),dims.end(),f->storedDims))
                throw std::logic_error("symbolic plan layout changed; discretize again");
        });
    }
    std::sort(p->info.schemes.begin(),p->info.schemes.end());
    p->info.schemes.erase(std::unique(p->info.schemes.begin(),p->info.schemes.end()),p->info.schemes.end());
    return p;
}
std::string source(const Plan& p,Fusion mode){
    std::ostringstream s;s<<"// Expanded direct derivatives; no intermediate flux differentiation.\n";
    s<<"typedef "<<(sizeof(Real)==sizeof(double)?"double":"float")<<" R;\n";
    if(mode==Fusion::Off)for(std::size_t i=0;i<p.primitives.size();++i){
        s<<"extern \"C\" __global__ void phix_d"<<i<<"(const R* f,R* out){\n"<<cellIndex(p)
         <<"out[c]="<<stencilSource(p.primitives[i],"f",p.sx)<<";\n}\n";
    }
    s<<"extern \"C\" __global__ void phix_symbolic(R* rhs,R coeff";
    for(std::size_t i=0;i<p.fields.size();++i)s<<",const R* f"<<i;
    if(mode==Fusion::Off)for(std::size_t i=0;i<p.primitives.size();++i)s<<",const R* d"<<i;
    s<<"){\n"<<cellIndex(p)<<Emitter(p,mode).body()<<"}\n";return s.str();
}
struct GpuProgram {
    CUmodule module=nullptr;CUcontext context=nullptr;CUfunction evaluate=nullptr;
    std::vector<CUfunction> derivatives;
    ~GpuProgram(){
        if(module){CUcontext current=nullptr;cuCtxGetCurrent(&current);
            if(current==context)cuModuleUnload(module);
            else if(cuCtxPushCurrent(context)==CUDA_SUCCESS){cuModuleUnload(module);CUcontext ignored;cuCtxPopCurrent(&ignored);}}
    }
};
Plan::~Plan()=default;
namespace {
std::shared_ptr<GpuProgram> program(const std::shared_ptr<Plan>& p,Fusion mode){
    std::lock_guard<std::mutex> lock(p->mutex);
    int device;PHIX_CUDA_CHECK(cudaGetDevice(&device));
    auto cacheKey=std::make_pair(device,mode==Fusion::Off?0:1);
    CUcontext context;driver(cuCtxGetCurrent(&context),"current CUDA context");
    if(auto i=p->gpuPrograms.find(cacheKey);i!=p->gpuPrograms.end()){
        if(i->second->context!=context)throw std::runtime_error("symbolic plan used in a different CUDA context");return i->second;
    }
    cudaDeviceProp prop;PHIX_CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    auto text=source(*p,mode);nvrtcProgram rtcProgram=nullptr;
    rtc(nvrtcCreateProgram(&rtcProgram,text.c_str(),"phix_expanded.cu",0,nullptr,nullptr),"create expanded CUDA program");
    std::string arch="--gpu-architecture=compute_"+std::to_string(prop.major)+std::to_string(prop.minor);
    const char* options[]={"--std=c++17",arch.c_str(),"--fmad=false"};
    auto status=nvrtcCompileProgram(rtcProgram,3,options);
    if(status!=NVRTC_SUCCESS){
        std::size_t size=0;nvrtcGetProgramLogSize(rtcProgram,&size);std::string log(size,'\0');nvrtcGetProgramLog(rtcProgram,log.data());nvrtcDestroyProgram(&rtcProgram);
        throw std::runtime_error("expanded CUDA compilation failed for "+p->name+":\n"+log);
    }
    std::size_t size;rtc(nvrtcGetPTXSize(rtcProgram,&size),"expanded PTX size");
    std::string ptx(size,'\0');rtc(nvrtcGetPTX(rtcProgram,ptx.data()),"expanded PTX");rtc(nvrtcDestroyProgram(&rtcProgram),"destroy CUDA compiler");
    auto result=std::make_shared<GpuProgram>();result->context=context;
    driver(cuModuleLoadData(&result->module,ptx.c_str()),"load expanded CUDA module");
    driver(cuModuleGetFunction(&result->evaluate,result->module,"phix_symbolic"),"expanded CUDA entry");
    if(mode==Fusion::Off)for(std::size_t i=0;i<p->primitives.size();++i){
        CUfunction fn;driver(cuModuleGetFunction(&fn,result->module,("phix_d"+std::to_string(i)).c_str()),"expanded derivative entry");result->derivatives.push_back(fn);
    }
    p->gpuPrograms.emplace(cacheKey,result);return result;
}
void runGPU(const std::shared_ptr<Plan>& p,Fusion mode,Real* rhs,double coefficient,ScratchPool& pool){
    for(const auto& check:p->layoutChecks)check();
    for(auto f:p->fields)check::checkOnDevice(*f,"symbolic input");
    auto compiled=program(p,mode);unsigned int blocks=(p->nx*p->ny+127)/128;
    std::vector<Real*> derivatives;
    if(mode==Fusion::Off)for(std::size_t i=0;i<p->primitives.size();++i){
        const Real* input=p->primitives[i].field->d_curr;Real* output=pool.acquireDevice(p->storedSize);derivatives.push_back(output);
        void* args[]={&input,&output};
        driver(cuLaunchKernel(compiled->derivatives[i],blocks,1,1,128,1,1,0,pool.stream,args,nullptr),"expanded derivative launch");
    }
    Real coeff=Real(coefficient);std::vector<const Real*> fields;for(auto f:p->fields)fields.push_back(f->d_curr);
    std::vector<void*> args{&rhs,&coeff};for(auto& f:fields)args.push_back(&f);for(auto& d:derivatives)args.push_back(&d);
    driver(cuLaunchKernel(compiled->evaluate,blocks,1,1,128,1,1,0,pool.stream,args.data(),nullptr),"expanded expression launch");
}
void runCPU(const std::shared_ptr<Plan>& p,Fusion mode,Real* rhs,double coefficient,ScratchPool& pool){
    for(const auto& check:p->layoutChecks)check();
    for(auto f:p->fields)if(f->curr.size()!=f->storedSize)throw std::invalid_argument("symbolic input lacks host storage");
    auto list=nodes(p->expression);std::map<const Node*,std::size_t> indices;
    for(std::size_t i=0;i<list.size();++i)indices[list[i].node().get()]=i;
    std::map<const Node*,const Primitive*> primitive;
    std::map<const Node*,Real*> values;
    for(const auto& d:p->primitives){
        primitive[d.node]=&d;
        if(mode==Fusion::Off){auto out=pool.acquireHost(p->storedSize);values[d.node]=out;
            for(int j=0;j<p->ny;++j)for(int i=0;i<p->nx;++i){int c=(i+p->ghost)+p->sx*((j+p->ghost)+p->sy*p->ghost);out[c]=evaluateStencil(d,d.field->curr.data(),c,p->sx);}}
    }
    std::vector<Real> cache(list.size());std::vector<unsigned char> ready(list.size());
    for(int j=0;j<p->ny;++j)for(int i=0;i<p->nx;++i){
        int c=(i+p->ghost)+p->sx*((j+p->ghost)+p->sy*p->ghost);std::fill(ready.begin(),ready.end(),0);
        std::function<Real(Expr)> eval=[&](Expr e)->Real{
            const auto& n=*e.node();auto index=indices.at(&n);if(ready[index])return cache[index];
            auto a=[&](int k){return eval(n.args[k]);};Real v=0;
            switch(n.op){
            case Op::Constant:v=Real(n.value);break;
            case Op::Field:case Op::Parameter:v=n.field->curr[c];break;
            case Op::Sample:v=n.field->curr[c+n.x+p->sx*n.y];break;
            case Op::Coordinate:v=Real(p->layout->mesh.coord(n.x,n.x==0?i:j));break;
            case Op::Jet:v=mode==Fusion::Off?values.at(&n)[c]:evaluateStencil(*primitive.at(&n),n.field->curr.data(),c,p->sx);break;
            case Op::Add:v=a(0)+a(1);break;case Op::Mul:v=a(0)*a(1);break;case Op::Divide:v=a(0)/a(1);break;
            case Op::Negate:v=-a(0);break;
            case Op::Sin:v=std::sin(a(0));break;case Op::Cos:v=std::cos(a(0));break;case Op::Tanh:v=std::tanh(a(0));break;
            case Op::Exp:v=std::exp(a(0));break;case Op::Log:v=std::log(a(0));break;case Op::Sqrt:v=std::sqrt(a(0));break;
            case Op::Abs:v=std::abs(a(0));break;case Op::Atan2:v=std::atan2(a(0),a(1));break;
            case Op::Hypot:v=std::hypot(a(0),a(1));break;
            case Op::Pow:{Real x=a(0);v=n.value==2?x*x:std::pow(x,Real(n.value));break;}
            case Op::Less:v=a(0)<a(1);break;case Op::LessEqual:v=a(0)<=a(1);break;
            case Op::Greater:v=a(0)>a(1);break;case Op::GreaterEqual:v=a(0)>=a(1);break;
            case Op::And:v=a(0)!=0 && a(1)!=0;break;case Op::Or:v=a(0)!=0 || a(1)!=0;break;
            case Op::Where:v=a(0)!=0?a(1):a(2);break;
            default:throw std::logic_error("unexpanded expression in CPU evaluator");
            }
            ready[index]=1;cache[index]=v;return v;
        };
        rhs[c]+=Real(coefficient)*eval(p->expression);
    }
}
}
Term lower(const std::shared_ptr<Plan>& p,Fusion mode){
    Term t;t.type=TermType::COMPOSITE;t.field=p->layout;t.inputs=p->fields;t.info=p->info;
    t.rhsGhost=p->ghost;for(const auto& r:t.info.reads)t.ghostRequired=std::max(t.ghostRequired,r.halo);
    t.info.execution="expanded direct derivatives: "+std::string(mode==Fusion::Off?"materialized primitive reference":"fused scalar DAG")+
        "; CUDA kernels="+std::to_string(1+(mode==Fusion::Off?p->primitives.size():0));
    t.cpu_kernel=[p,mode](Real* r,double c,ScratchPool& pool){runCPU(p,mode,r,c,pool);};
    t.gpu_launcher=[p,mode](Real* r,double c,ScratchPool& pool){runGPU(p,mode,r,c,pool);};return t;
}
} // namespace PhiX::numerics::symbolic::detail
