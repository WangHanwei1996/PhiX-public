#include "numerics/Symbolic.h"
#include "boundary/PeriodicBC.h"
#include "core/CudaCheck.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <set>
using namespace PhiX;
namespace num=PhiX::numerics;
namespace sym=num::symbolic;
namespace {
constexpr double tol=sizeof(Real)==sizeof(float)?3e-4:3e-11;
void require(bool yes,const std::string& message){if(!yes)throw std::runtime_error(message);}
void close(double actual,double expected,const std::string& message,double tolerance=tol){
    if(!std::isfinite(actual) || std::abs(actual-expected)>tolerance*std::max(1.,std::abs(expected)))
        throw std::runtime_error(message+": "+std::to_string(actual)+" vs "+std::to_string(expected));
}
template<class Fn>void rejects(Fn fn,const std::string& message){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}require(caught,message);}
template<class Fn>void fill(ScalarField& f,Fn fn){
    for(int j=-f.ghost;j<f.mesh.n[1]+f.ghost;++j)for(int i=-f.ghost;i<f.mesh.n[0]+f.ghost;++i)
        f.curr[f.index(i,j)]=Real(fn(f.mesh.coord(0,i),f.mesh.coord(1,j)));
}
void upload(ScalarField& f){f.allocDevice();f.uploadAllToDevice();}
template<class Fn>void compare(sym::Discrete expression,ScalarField& out,Fn exact){
    for(auto mode:{num::Fusion::Off,num::Fusion::Auto,num::Fusion::Required}){
        Equation equation(out);equation.setRHS(expression.lower(mode));
        for(bool gpu:{false,true}){
            if(gpu){equation.computeRHS(out);out.downloadCurrFromDevice();}else equation.computeRHSCPU(out);
            for(int j=0;j<out.mesh.n[1];++j)for(int i=0;i<out.mesh.n[0];++i)
                close(out.curr[out.index(i,j)],exact(i,j),"expanded reference "+std::to_string(int(mode))+(gpu?" GPU":" CPU"));
        }
    }
}
void productAndChain(){
    auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,12,.2,0,9,.3,0);
    ScalarField f(mesh,"f",1),a(mesh,"a",1),out(mesh,"out",1);
    fill(f,[](double x,double y){return x*x+x*y+3*y*y;});
    fill(a,[](double x,double y){return 1+x*x+y;});out.fill(0);
    auto u=sym::field(f),coefficient=sym::field(a);
    auto original=sym::div(coefficient*sym::grad(u));auto expanded=original.expand("variable diffusion");
    auto inventory=expanded.requiredOperators();require(inventory.entries.size()==5,"variable diffusion operator inventory");
    inventory.merge(original.expand("second equation").requiredOperators());
    require(inventory.entries.size()==5 && inventory.entries[0].sources.size()==2,"merged operator inventory lost provenance");
    auto config=inventory.schemeTemplate();auto schemes=Schemes::fromJson(config);
    auto discrete=expanded.discretize(schemes); // before device allocation
    std::ostringstream report;expanded.print(report);
    require(report.str().find("spatial product/chain")!=std::string::npos,"expansion trace missing");
    require(discrete.cudaSource().find("phix_d0")==std::string::npos,"Auto materialized a derivative");
    require(discrete.kernelLaunches(num::Fusion::Auto)==1 && discrete.kernelLaunches(num::Fusion::Off)==6,"execution counts");
    for(auto q:{&f,&a,&out})upload(*q);
    compare(discrete,out,[&](int i,int j){double x=mesh.coord(0,i),y=mesh.coord(1,j);return 2*x*(2*x+y)+(x+6*y)+8*(1+x*x+y);});
    // Chain rule is applied to tanh before differencing. A direct difference of
    // tanh(f) would have a different answer on this deliberately coarse mesh.
    auto phi=sym::named("phi",sym::tanh(u/std::sqrt(2.)));
    auto chain=sym::lap(phi).expand().discretize(Schemes{});
    compare(chain,out,[&](int i,int j){double x=mesh.coord(0,i),y=mesh.coord(1,j),p=std::tanh((x*x+x*y+3*y*y)/std::sqrt(2.));
        double norm=(2*x+y)*(2*x+y)+(x+6*y)*(x+6*y);return (1-p*p)*(8/std::sqrt(2.)-p*norm);});
    // Explicit branch semantics: the inactive branch contains 0/0.
    fill(f,[](double,double){return 0.;});f.uploadAllToDevice();
    auto guarded=sym::where(u>0,1/u,sym::where(u<0,sym::sqrt(u),7)).expand().discretize(Schemes{});
    compare(guarded,out,[](int,int){return 7.;});
    // Algebraic partial derivative is a symbolic operation, not a stencil.
    auto polynomial=sym::partial(sym::pow(u,3)+2*u,u).expand().discretize(Schemes{});
    compare(polynomial,out,[](int,int){return 2.;});
    config["gradient"].erase("grad(a,x)");rejects([&]{expanded.discretize(Schemes::fromJson(config));},"strict missing selection accepted");
}
void hessianAndRotation(){
    auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,11,.8,0,9,.6,0);
    ScalarField f(mesh,"f",3),theta(mesh,"theta",3),out(mesh,"out",3);
    fill(f,[](double x,double y){return .2*x*x+.3*x*y+.4*y*y;});
    fill(theta,[](double x,double){return x<4?.2:.7;});out.fill(0);for(auto q:{&f,&theta,&out})upload(*q);
    auto h=sym::rotate(sym::hessian(sym::field(f)),sym::parameter(theta));
    auto expression=(h.xx+3*h.xy+2*h.yy).expand("rotated Hessian");
    auto config=expression.requiredOperators().schemeTemplate();
    config["secondDerivative"]["dxx(f)"]="CD4";config["secondDerivative"]["dxy(f)"]="CD6";
    auto d=expression.discretize(Schemes::fromJson(config));
    compare(d,out,[&](int i,int j){double angle=theta.curr[theta.index(i,j)],c=std::cos(angle),s=std::sin(angle);
        return (.4*c*c+.6*c*s+.8*s*s)+3*(.4*c*s+.3*(c*c-s*s))+2*(.4*s*s-.6*c*s+.8*c*c);});
    // A checkerboard has zero centered first derivative but nonzero direct dxx.
    for(int j=-3;j<12;++j)for(int i=-3;i<14;++i)f.curr[f.index(i,j)]=(i%2)?-1:1;
    f.uploadAllToDevice();auto second=sym::dx(sym::dx(sym::field(f))).expand();
    require(second.requiredOperators().entries.size()==1 && second.requiredOperators().entries[0].key=="dxx(f)","nested derivative not reduced to dxx");
    compare(second.discretize(Schemes{}),out,[&](int i,int j){return -4*double(f.curr[f.index(i,j)])/(.8*.8);});
    auto originalGhost=f.ghost;f.ghost=2;
    rejects([&]{Equation e(out);e.setRHS(d.lower(num::Fusion::Auto));e.computeRHSCPU(out);},"stale symbolic layout accepted");f.ghost=originalGhost;
}
sym::Expr anisotropy(sym::Expr u,sym::Expr orientation,double eps){
    auto g=sym::grad(u); auto angle=sym::atan2(g.y,g.x);
    auto a=1+eps*sym::cos(4*(angle-orientation));
    return sym::div(a*a*g+a*sym::partial(a,angle)*sym::perpendicular(g));
}
void appendixB(){
    auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,17,.7,0,15,.7,0);
    ScalarField f(mesh,"psi",1),theta(mesh,"theta",1),out(mesh,"out",1);
    fill(f,[](double x,double y){return x+.2*y+.1*std::sin(.7*x)*std::cos(.6*y);});
    fill(theta,[](double x,double){return x<5?0.:25*M_PI/180;});out.fill(0);
    auto u=sym::field(f);auto original=anisotropy(u,sym::parameter(theta),.007);
    auto expanded=original.expand("TK2015 Appendix B");auto list=expanded.requiredOperators();
    require(list.entries.size()==6,"anisotropy must use dx dy dxx dxy dyy and lap");
    for(const auto& e:list.entries)require(e.field=="psi" && e.section!="anisoDiv","unexpected intermediate/orientation derivative");
    auto config=list.schemeTemplate();config["laplacian"]["lap(psi)"]="Iso9";
    auto discrete=expanded.discretize(Schemes::fromJson(config));
    for(auto q:{&f,&theta,&out})upload(*q);
    auto value=[&](int i,int j){return double(f.curr[f.index(i,j)]);};
    // Independent Appendix B formula, with direct compact derivatives and PK9 lap.
    compare(discrete,out,[&](int i,int j){
        double h=.7,p=value(i,j),px=(value(i+1,j)-value(i-1,j))/(2*h),py=(value(i,j+1)-value(i,j-1))/(2*h);
        double xx=(value(i+1,j)-2*p+value(i-1,j))/(h*h),yy=(value(i,j+1)-2*p+value(i,j-1))/(h*h);
        double xy=(value(i+1,j+1)-value(i+1,j-1)-value(i-1,j+1)+value(i-1,j-1))/(4*h*h);
        double lap=(4*(value(i-1,j)+value(i+1,j)+value(i,j-1)+value(i,j+1))+
            value(i-1,j-1)+value(i-1,j+1)+value(i+1,j-1)+value(i+1,j+1)-20*p)/(6*h*h);
        double angle=std::atan2(py,px)-theta.curr[theta.index(i,j)],a=1+.007*std::cos(4*angle),b=-.028*std::sin(4*angle),c=-.112*std::cos(4*angle);
        double tx=(px*xy-py*xx)/(px*px+py*py),ty=(px*yy-py*xy)/(px*px+py*py);
        return a*a*lap+2*a*b*(px*tx+py*ty)+(b*b+a*c)*(px*ty-py*tx);
    });
    // Rebinding must change a nonzero anisotropic result, not just stay finite.
    Equation e(out);e.setRHS(discrete.lower(num::Fusion::Auto));e.computeRHS(out);out.downloadCurrFromDevice();
    auto before=out.curr;
    ScalarField other(mesh,"replacement",1);fill(other,[](double,double){return .4;});upload(other);
    std::swap(theta.d_curr,other.d_curr);std::swap(theta.curr,other.curr);
    e.computeRHSCPU(out);auto reference=out.curr;e.computeRHS(out);out.downloadCurrFromDevice();
    double difference=0;
    for(int j=0;j<15;++j)for(int i=0;i<17;++i){auto c=out.index(i,j);
        close(out.curr[c],reference[c],"current parameter binding");difference=std::max(difference,std::abs(double(out.curr[c]-before[c])));}
    require(difference>1e-4,"parameter rebinding check is insensitive to orientation");
    std::swap(theta.d_curr,other.d_curr);std::swap(theta.curr,other.curr);
    // Constant gradient with a grain jump: no face-orientation coupling.
    fill(f,[](double x,double y){return x+.25*y;});f.uploadAllToDevice();
    compare(discrete,out,[](int,int){return 0.;});

}
void convergence(){
    #ifndef PHIX_REAL_FLOAT // high-order error reaches float roundoff
    for(const auto& name:{"CD2","CD4","CD6"}){
        double previous=0;
        for(int n:{8,16,32}){
            double h=1./n;auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,n,h,0,n,h,0);
            ScalarField f(mesh,"f",3),out(mesh,"out",3);
            fill(f,[](double x,double y){return std::sin(2*x+.3)*std::cos(3*y-.1);});out.fill(0);
            auto u=sym::field(f);auto expression=(sym::dxx(u)+2*sym::dxy(u)+3*sym::dyy(u)).expand()
                .discretize(Schemes::builtin({{"secondDerivative",name}}));
            Equation e(out);e.setRHS(expression.lower(num::Fusion::Auto));e.computeRHSCPU(out);
            double error=0;
            for(int j=0;j<n;++j)for(int i=0;i<n;++i){double x=mesh.coord(0,i),y=mesh.coord(1,j);
                double expected=-31*std::sin(2*x+.3)*std::cos(3*y-.1)-12*std::cos(2*x+.3)*std::sin(3*y-.1);
                error=std::max(error,std::abs(out.curr[out.index(i,j)]-expected));}
            if(previous){double order=std::log(previous/error)/std::log(2.);
                require(order>(std::string(name)=="CD2"?1.8:std::string(name)=="CD4"?3.7:5.4),"direct Hessian convergence order "+std::string(name));
                std::cout<<name<<" N="<<n<<" Linf="<<error<<" order="<<order<<'\n';}
            previous=error;
        }
    }
    #endif
}
void boundaryEvolution(){
    const int n=16;const double h=2*M_PI/n;
    auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,n,h,0,n,h,0);
    ScalarField f(mesh,"f",1);PeriodicBC bx(mesh.facePatch(Axis::X,Side::LOW)),by(mesh.facePatch(Axis::Y,Side::LOW));
    Schemes schemes;
    for(auto backend:{num::Backend::CPU,num::Backend::CUDA}){
        f.fill(123);for(int j=0;j<n;++j)for(int i=0;i<n;++i)f.curr[f.index(i,j)]=Real(std::sin(mesh.coord(0,i))*std::cos(mesh.coord(1,j)));
        auto initial=f.curr;upload(f);auto u=sym::field(f);
        num::System system(schemes);system.bc(f,{&bx,&by});
        system.add((sym::ddt(f)==sym::dxx(u)+.3*sym::dxy(u)).expand().discretize(schemes));
        system.compile({backend,num::Fusion::Auto,num::Coupling::SameLevel});system.advance(.01);
        if(backend==num::Backend::CUDA)f.downloadCurrFromDevice();
        for(int j=0;j<n;++j)for(int i=0;i<n;++i){double x=mesh.coord(0,i),y=mesh.coord(1,j);
            double rate=-4*std::pow(std::sin(h/2),2)/(h*h)*std::sin(x)*std::cos(y)
                        -.3*std::pow(std::sin(h)/h,2)*std::cos(x)*std::sin(y);
            close(f.curr[f.index(i,j)],initial[f.index(i,j)]+.01*rate,"periodic mixed derivative halo including corners");}
    }
    auto x=sym::coordinate(0);
    auto expression=sym::dx(sym::exp(sym::sin(x))+sym::log(2+x)+sym::sqrt(2+x)+sym::abs(x-.4));
    ScalarField out(mesh,"out",1);out.fill(0);upload(out);
    compare(expression.expand().discretize(schemes,f),out,[&](int i,int){double x=mesh.coord(0,i);
        return std::exp(std::sin(x))*std::cos(x)+1/(2+x)+.5/std::sqrt(2+x)+(x>.4?1:-1);});
}
void evolutionAndErrors(){
    auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,12,.4,0,10,.4,0);
    ScalarField f(mesh,"f",1),g(mesh,"g",1);fill(f,[](double x,double y){return .2*std::sin(x)+.1*std::cos(y);});
    fill(g,[](double x,double y){return .1*std::cos(x+y);});auto initial=f.curr,initialG=g.curr;
    Schemes schemes;PeriodicBC bx(mesh.facePatch(Axis::X,Side::LOW)),by(mesh.facePatch(Axis::Y,Side::LOW));
    for(auto backend:{num::Backend::CPU,num::Backend::CUDA}){
        f.curr=initial;g.curr=initialG;upload(f);upload(g);
        num::System system(schemes);system.bc(f,{&bx,&by});system.bc(g,{&bx,&by});
        system.add((2*sym::ddt(f)==2*sym::field(g)).expand().discretize(schemes));
        system.add((sym::ddt(g)==-sym::field(f)).expand().discretize(schemes));
        system.compile({backend,num::Fusion::Auto,num::Coupling::SameLevel});system.advance(.01);
        if(backend==num::Backend::CUDA){f.downloadCurrFromDevice();g.downloadCurrFromDevice();}
        for(int j=0;j<10;++j)for(int i=0;i<12;++i){auto c=f.index(i,j);close(f.curr[c],initial[c]+.01*initialG[c],"same-level Euler f");close(g.curr[c],initialG[c]-.01*initial[c],"same-level Euler g");}
    }
    rejects([&]{sym::dx(sym::dxx(sym::field(f))).expand();},"unsupported third derivative accepted");
    rejects([&]{sym::dxx(sym::field(f)).expand().discretize(Schemes::builtin({{"secondDerivative","CD4"}}));},"insufficient halo accepted");
    rejects([&]{(sym::ddt(f)==sym::field(f)).expand().discretize(Schemes::builtin({{"ddt","RK4"}}));},"non-Euler accepted");
    ScalarField duplicate(mesh,"f",1);rejects([&]{(sym::field(f)+sym::field(duplicate)).expand().discretize(schemes);},"ambiguous field names accepted");
    auto coordinate=(sym::dx(sym::pow(sym::coordinate(0),2))+sym::dy(sym::coordinate(1))).expand().discretize(schemes,f);
    ScalarField out(mesh,"out",1);out.fill(0);upload(out);compare(coordinate,out,[&](int i,int){return 2*mesh.coord(0,i)+1;});
}
}
int main(){try{productAndChain();hessianAndRotation();appendixB();convergence();boundaryEvolution();evolutionAndErrors();
    std::cout<<"symbolic expansion, direct stencils, Appendix B, branches, CPU/CUDA and Euler passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
