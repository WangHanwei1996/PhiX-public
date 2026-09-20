#include "numerics/Face.h"
#include "operators/DivFaceAssembled.h"
#include "boundary/PeriodicBC.h"
#include "boundary/Apply2D.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>
using namespace PhiX;
namespace num=PhiX::numerics;
namespace {
constexpr double tol=std::is_same<Real,float>::value ? 3e-4 : 2e-11;
void require(bool b,const char *s) { if (!b) throw std::runtime_error(s); }
void device(ScalarField &f) { f.allocDevice(); f.uploadAllToDevice(); }
void evaluate(Term t, ScalarField &f, bool gpu) {
    Equation eq(f); eq.setRHS(t);
    if (gpu) { eq.computeRHS(f); f.downloadCurrFromDevice(); } else eq.computeRHSCPU(f);
}
void close(const ScalarField &a,const ScalarField &b) {
    for (int j=0;j<a.mesh.n[1];++j) for (int i=0;i<a.mesh.n[0];++i) {
        const auto n=a.index(i,j);
        require(std::isfinite(a.curr[n]) && std::abs(a.curr[n]-b.curr[n])<tol,"extension result differs from reference");
    }
}
struct Anti {
    int axis;
    __host__ __device__ Real operator()(Real gx,Real gy,Real a) const {
        const Real n2=gx*gx+gy*gy;
        return n2>Real(1e-12) ? a*(axis==0?gx:gy)/sqrt(n2) : Real(0);
    }
};
}
int main() {
    try {
        auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,19,.7,0,13,.7,0);
        ScalarField a(mesh,"a",1),b(mesh,"b",1),c(mesh,"c",1),d(mesh,"d",1),e(mesh,"e",1);
        ScalarField out(mesh,"out",1),ref(mesh,"reference",1);
        int fno=0;
        for (auto *f:{&a,&b,&c,&d,&e}) {
            const double k=++fno;
            f->initialize([=](double x,double y,double) { return .3*k+.1*cos(x*.37)+.07*sin(y*.43); });
            device(*f);
        }
        device(out);device(ref);
        Schemes schemes;
        num::Spatial op(schemes);num::Faces faces(schemes);
        const auto five=num::pure(PHIX_FN(Real a,Real b,Real c,Real d,Real e) { return a*b+c*d-e; });
        const auto eight=num::pure(PHIX_FN(Real a,Real b,Real c,Real d,Real e,Real f,Real g,Real h) {
            return a*b+c*d-e*f+g/h;
        });
        auto p5=op.pw(five,a,b,c,d,e);
        auto p8=op.pw(eight,a,b,c,d,e,a,b,c); // repeated input must retain one coherent dependency
        for (bool gpu:{false,true}) for (auto mode:{num::Fusion::Off,num::Fusion::Auto,num::Fusion::Required}) {
            evaluate(PhiX::pw(a,b,c,d,e,five.fn),ref,gpu);evaluate(p5.lower(mode),out,gpu);close(out,ref);
            evaluate(PhiX::pw(a,b,c,d,e,a,b,c,eight.fn),ref,gpu);evaluate(p8.lower(mode),out,gpu);close(out,ref);
        }
        // Expressions must bind current storage again after a host/device replacement.
        a.curr.swap(b.curr);std::swap(a.d_curr,b.d_curr);
        for (bool gpu:{false,true}) {
            evaluate(PhiX::pw(a,b,c,d,e,five.fn),ref,gpu);evaluate(p5.lower(num::Fusion::Auto),out,gpu);close(out,ref);
        }
        PeriodicBC x(mesh.patch("xmin")),y(mesh.patch("ymin"));
        std::vector<BoundaryCondition*> bc{&x,&y};
        applyBCsCPU2D(a,bc);applyBCsCPU2D(b,bc);
        a.uploadAllToDevice();b.uploadAllToDevice();
        auto flux=faces.flux("anti",[&](auto axis) {
            return num::pw(faces.gradComponent(a,axis,0),faces.gradComponent(a,axis,1),
                           faces.interp(b,axis),num::pure(Anti{decltype(axis)::value}));
        });
        auto old=divFaceAssembled(a,b,PHIX_FN(int ax,double gx,double gy,double,double,double af) {
            const double n2=gx*gx+gy*gy;return n2>1e-12 ? af*(ax==0?gx:gy)/sqrt(n2) : 0.;
        });
        for (bool gpu:{false,true}) for (auto mode:{num::Fusion::Off,num::Fusion::Auto}) {
            evaluate(old,ref,gpu);evaluate(faces.div(flux).lower(mode),out,gpu);close(out,ref);
            double total=0;
            for (int j=0;j<mesh.n[1];++j) for (int i=0;i<mesh.n[0];++i) total+=out.curr[out.index(i,j)];
            require(std::abs(total)<tol*20,"periodic antitrapping divergence must telescope");
            auto composed=num::sum(num::sum(faces.div(flux),2.0*faces.div(flux)),p5);
            evaluate(num::sum(num::sum(old,old*2),PhiX::pw(a,b,c,d,e,five.fn)),ref,gpu);
            evaluate(composed.lower(mode),out,gpu);close(out,ref);
            require(composed.lower(mode).info.execution.find("anti")!=std::string::npos,"sum lost execution report");
        }
        std::cout << "Pointwise 5/8 inputs, rebinding, face nonlinear reference/conservation, region sums: passed\n";
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
}
