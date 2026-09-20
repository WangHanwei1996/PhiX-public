#include "numerics/Preparation.h"
#include "numerics/Stencil.h"
#include "boundary/Apply2D.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include <cmath>
#include <iostream>
using namespace PhiX;
namespace num=PhiX::numerics;
namespace sym=num::symbolic;
namespace {
constexpr double tol=sizeof(Real)==4?3e-4:5e-11;
void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
void close(double a,double b,const char* what){
    if(!std::isfinite(a) || std::abs(a-b)>tol*std::max(1.,std::abs(b))){
        std::cerr<<what<<": "<<a<<" vs "<<b<<'\n';throw std::runtime_error(what);
    }
}
template<class F>void fill(ScalarField& f,F value){
    f.fill(0);
    for(int j=-2;j<f.mesh.n[1]+2;++j)for(int i=-2;i<f.mesh.n[0]+2;++i)
        f.curr[f.index(i,j)]=value(f.mesh.coord(0,i),f.mesh.coord(1,j));
}
void upload(ScalarField& f){f.allocDevice();f.uploadAllToDevice();}
void evaluate(sym::Expr expr,ScalarField& out,const Schemes& schemes,bool gpu,num::Fusion fusion){
    Equation equation(out);equation.setRHS(expr.expand().discretize(schemes).lower(fusion));
    if(gpu){equation.computeRHS(out);out.downloadCurrFromDevice();}else equation.computeRHSCPU(out);
}
Schemes choices(){auto d=scheme::preset("Ji2022S21").defaults;d["divNormal"]="Iso21Vector";return Schemes::builtin(d);}

// 1. Near-cancelled off-lattice derivative: the published mixed reconstruction
// can grow arbitrarily, whereas projecting the same normalized vector is bounded.
void boundedLink(){
    auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,1,1.,-.5,1,1.,-.5);
    ScalarField b(mesh,"b",2),out(mesh,"out",2);out.fill(0);
    auto recipe=sym::FluxStencil::isotropic(true);recipe.links.resize(1);
    recipe.projection=sym::FluxProjection::ReconstructedGradient;
    upload(b);upload(out);
    for(double small:{.02,1e-4,1e-6}){
        fill(b,[&](double x,double y){
            int i=int(std::round(x));double value=i<0?-13.:(i==0?0.:(i==1?1.:14.-24*small));
            return value+small*y;
        });b.uploadAllToDevice();
        auto value=recipe.apply(2,sym::field(b),1,1,true);
        for(auto fusion:{num::Fusion::Off,num::Fusion::Auto,num::Fusion::Required})for(bool gpu:{false,true}){
            evaluate(value,out,Schemes{},gpu,fusion);
            require(std::isfinite(out.curr[out.index(0,0)]) && std::abs(out.curr[out.index(0,0)])<=4./3+tol,
                    "normalized vector component exceeded coefficient/link bound");
        }
    }
    for(double constant:{0.,.37,137.}){
        b.fill(constant);b.uploadAllToDevice();
        for(bool gpu:{false,true}){
            evaluate(recipe.apply(2,sym::field(b),1,1,true),out,Schemes{},gpu,num::Fusion::Auto);
            require(out.curr[out.index(0,0)]==0,"constant must have exactly zero reconstructed gradient");
        }
    }
    bool rejected=false;
    try{sym::weightedDifferences(sym::field(b),{{0,0,1},{1,0,1}});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"nonzero-sum difference stencil accepted");
}

// 2. Shared links conserve arbitrary signed fluxes, including stationary points.
// CPU/CUDA and all fusion policies must evaluate the same discrete operator.
void periodicConservation(){
    constexpr int n=24;const double h=2*M_PI/n;
    auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,n,h,0,n,h,0);
    ScalarField a(mesh,"a",2),b(mesh,"b",2),out(mesh,"out",2);out.fill(0);
    fill(a,[](double x,double y){return .2+std::sin(2*x+y);});
    fill(b,[](double x,double y){return std::sin(x)+.4*std::cos(3*y)+.2*std::sin(5*x+2*y);});
    PeriodicBC px(mesh.facePatch(Axis::X,Side::LOW)),py(mesh.facePatch(Axis::Y,Side::LOW));
    applyBCsCPU2D(a,{&px,&py});applyBCsCPU2D(b,{&px,&py});
    upload(a);upload(b);upload(out);
    auto expression=sym::divNormal(sym::field(a),sym::field(b));auto schemes=choices();
    auto info=expression.expand().inspect(schemes,out);
    require(info.reads.size()==2,"unexpected input inventory");
    for(auto r:info.reads)require(r.halo==(r.cell==&a?1:2),"wrong composed halo");
    evaluate(expression,out,schemes,false,num::Fusion::Auto);auto expected=out.curr;
    for(auto fusion:{num::Fusion::Off,num::Fusion::Auto,num::Fusion::Required})for(bool gpu:{false,true}){
        evaluate(expression,out,schemes,gpu,fusion);double sum=0,scale=0;
        for(int j=0;j<n;++j)for(int i=0;i<n;++i){auto q=out.index(i,j);
            close(out.curr[q],expected[q],"vector CPU/CUDA agreement");sum+=out.curr[q];scale+=std::abs(out.curr[q]);
        }
        require(std::abs(sum)<tol*std::max(1.,scale),"vector flux conservation");
    }
    // The production case reflects x and is periodic in y. Paired diagonal
    // links must cancel at the wall as well as across periodic seams.
    NoFluxBC xl(mesh.facePatch(Axis::X,Side::LOW),NoFluxBC::Closure::Reflect);
    NoFluxBC xh(mesh.facePatch(Axis::X,Side::HIGH),NoFluxBC::Closure::Reflect);
    applyBCsCPU2D(a,{&xl,&xh,&py});applyBCsCPU2D(b,{&xl,&xh,&py});
    a.uploadAllToDevice();b.uploadAllToDevice();
    evaluate(expression,out,schemes,false,num::Fusion::Auto);expected=out.curr;
    for(auto fusion:{num::Fusion::Off,num::Fusion::Auto,num::Fusion::Required})for(bool gpu:{false,true}){
        evaluate(expression,out,schemes,gpu,fusion);double sum=0,scale=0;
        for(int j=0;j<n;++j)for(int i=0;i<n;++i){auto q=out.index(i,j);
            close(out.curr[q],expected[q],"reflected vector CPU/CUDA agreement");
            sum+=out.curr[q];scale+=std::abs(out.curr[q]);
        }
        require(std::abs(sum)<tol*std::max(1.,scale),"reflected vector flux conservation");
    }
    auto draft=sym::prepare(expression.expand(),out,{scheme::preset("Ji2022S21"),{{"divNormal",{{"default","Iso21Vector"}}}}});
    require(draft.report.at("ready").get<bool>() && draft.report.at("presetModified").get<bool>(),"vector preparation round trip");
}

// 3. Smooth 1D profile rotated relative to the grid: exact normal points along
// the profile, so div(a*n)=a'. Check h^2 error and h^4 angular spread separately.
void convergence(){
    if(sizeof(Real)==4)return; // fine-grid angular spread falls below float precision
    auto schemes=choices();double previousError=0,previousSpread=0;
    for(double h:{.4,.2,.1}){
        auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,1,h,-.5*h,1,h,-.5*h);
        ScalarField a(mesh,"a",2),b(mesh,"b",2),out(mesh,"out",2);out.fill(0);
        double error=0,lo=1e30,hi=-1e30;
        for(int q=0;q<=12;++q){double angle=q*M_PI/48,c=std::cos(angle),s=std::sin(angle);
            fill(a,[&](double x,double y){return 1+.3*std::cos(c*x+s*y+.37);});
            fill(b,[&](double x,double y){double z=c*x+s*y+.37;return z+.2*std::sin(z);});
            evaluate(sym::divNormal(sym::field(a),sym::field(b)),out,schemes,false,num::Fusion::Auto);
            double value=out.curr[out.index(0,0)];lo=std::min(lo,value);hi=std::max(hi,value);
            error=std::max(error,std::abs(value+.3*std::sin(.37)));
        }
        double spread=hi-lo;
        if(previousError)require(previousError/error>3.6,"vector flux is not second-order consistent");
        if(previousSpread)require(previousSpread/spread>12,"vector flux leading error is not isotropic");
        std::cout<<"Iso21Vector h="<<h<<" max_error="<<error<<" angular_spread="<<spread<<'\n';
        previousError=error;previousSpread=spread;
    }
}
// Curved normals as well as straight profiles: the rotation property must
// survive nonlinear normalization, not only reduce to a derivative of a.
void curvedConvergence(){
    if(sizeof(Real)==4)return;
    auto schemes=choices();double previousError=0,previousSpread=0;
    const double X=.37,Y=.29,A=1+.3*std::cos(X)+.1*std::sin(Y);
    const double gx=1+.2*std::cos(X),gy=-.1*std::sin(Y),n=std::hypot(gx,gy);
    const double xx=-.2*std::sin(X),yy=-.1*std::cos(Y);
    const double exact=(-.3*std::sin(X)*gx+.1*std::cos(Y)*gy)/n+
        A*((xx+yy)/n-(gx*gx*xx+gy*gy*yy)/(n*n*n));
    for(double h:{.4,.2,.1}){
        auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,1,h,-.5*h,1,h,-.5*h);
        ScalarField a(mesh,"a",2),b(mesh,"b",2),out(mesh,"out",2);out.fill(0);
        double error=0,lo=1e30,hi=-1e30;
        for(int q=0;q<=12;++q){double angle=q*M_PI/48,c=std::cos(angle),s=std::sin(angle);
            fill(a,[&](double x,double y){return 1+.3*std::cos(c*x+s*y+X)+.1*std::sin(-s*x+c*y+Y);});
            fill(b,[&](double x,double y){double z=c*x+s*y+X,w=-s*x+c*y+Y;return z+.2*std::sin(z)+.1*std::cos(w);});
            evaluate(sym::divNormal(sym::field(a),sym::field(b)),out,schemes,false,num::Fusion::Auto);
            double value=out.curr[out.index(0,0)];lo=std::min(lo,value);hi=std::max(hi,value);
            error=std::max(error,std::abs(value-exact));
        }
        double spread=hi-lo;
        if(previousError)require(previousError/error>3.6,"curved-normal second order");
        if(previousSpread)require(previousSpread/spread>12,"curved-normal leading-error isotropy");
        std::cout<<"Iso21Vector curved h="<<h<<" max_error="<<error<<" angular_spread="<<spread<<'\n';
        previousError=error;previousSpread=spread;
    }
}
}
int main(){try{boundedLink();periodicConservation();convergence();curvedConvergence();std::cout<<"Iso21Vector bounds, conservation, CPU/CUDA and isotropy passed\n";}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
