// Expanded anisotropy: compare the same bound discrete graph, excluding JIT setup.
#include "numerics/Symbolic.h"
#include "core/CudaCheck.h"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace PhiX;
namespace num=PhiX::numerics;
namespace sym=num::symbolic;
int main(){
    try{
        nlohmann::json results=nlohmann::json::array();
        for(int n:{128,1024}){
            auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,n,.7,0,n,.7,0);
            ScalarField psi(mesh,"psi",1),theta(mesh,"theta",1),out(mesh,"out",1);
            psi.fill(0);theta.fill(.25);out.fill(0);
            for(int j=-1;j<=n;++j)for(int i=-1;i<=n;++i){double x=mesh.coord(0,i),y=mesh.coord(1,j);
                psi.curr[psi.index(i,j)]=Real(.2*x+.1*y+std::sin(.1*x)*std::cos(.12*y));}
            for(auto* f:{&psi,&theta,&out}){f->allocDevice();f->uploadAllToDevice();}
            auto s=sym::field(psi),orientation=sym::parameter(theta);auto g=sym::grad(s);
            auto angle=sym::atan2(g.y,g.x),a=1+.007*sym::cos(4*(angle-orientation));
            auto expanded=sym::div(a*a*g+a*sym::partial(a,angle)*sym::perpendicular(g)).expand("benchmark anisotropy");
            auto plan=expanded.discretize(Schemes{});
            nlohmann::json row={{"n",n},{"primitive_count",expanded.requiredOperators().entries.size()}};
            std::vector<Real> reference;
            for(auto mode:{num::Fusion::Off,num::Fusion::Auto}){
                Equation equation(out);equation.setRHS(plan.lower(mode));
                for(int k=0;k<5;++k)equation.computeRHS(out);
                PHIX_CUDA_CHECK(cudaDeviceSynchronize());
                cudaEvent_t begin,end;PHIX_CUDA_CHECK(cudaEventCreate(&begin));PHIX_CUDA_CHECK(cudaEventCreate(&end));
                constexpr int repeats=100;PHIX_CUDA_CHECK(cudaEventRecord(begin));
                for(int k=0;k<repeats;++k)equation.computeRHS(out);
                PHIX_CUDA_CHECK(cudaEventRecord(end));PHIX_CUDA_CHECK(cudaEventSynchronize(end));
                float ms;PHIX_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
                PHIX_CUDA_CHECK(cudaEventDestroy(begin));PHIX_CUDA_CHECK(cudaEventDestroy(end));
                out.downloadCurrFromDevice();
                const std::string name=mode==num::Fusion::Off?"off":"auto";
                row[name]={{"milliseconds_per_rhs",ms/repeats},{"expression_kernel_launches",plan.kernelLaunches(mode)}};
                if(mode==num::Fusion::Off)reference=out.curr;
                else{
                    double error=0;for(int j=0;j<n;++j)for(int i=0;i<n;++i){auto c=out.index(i,j);
                        if(!std::isfinite(out.curr[c]))throw std::runtime_error("nonfinite benchmark result");
                        error=std::max(error,std::abs(double(out.curr[c]-reference[c])));}
                    row["maximum_off_auto_error"]=error;
                    if(error>(sizeof(Real)==sizeof(float)?1e-4:1e-10))throw std::runtime_error("benchmark discrete equivalence failed");
                }
            }
            row["speedup"]=double(row["off"]["milliseconds_per_rhs"])/double(row["auto"]["milliseconds_per_rhs"]);
            results.push_back(std::move(row));
        }
        std::cout<<results.dump(2)<<'\n';
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
