#include "numerics/Preparation.h"
#include "numerics/Stencil.h"
#include "boundary/Apply2D.h"
#include "boundary/BCFactory.h"
#include "core/CudaCheck.h"
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

using namespace PhiX;
namespace num=PhiX::numerics;
namespace sym=num::symbolic;
namespace {
constexpr double tol=sizeof(Real)==sizeof(float)?8e-5:4e-11;
void require(bool yes,const std::string& why){if(!yes)throw std::runtime_error(why);}
void close(double a,double b,const std::string& why,double t=tol){
    require(std::isfinite(a) && std::abs(a-b)<=t*std::max(1.,std::abs(b)),
            why+": "+std::to_string(a)+" vs "+std::to_string(b));
}
template<class F>void rejects(F f,const std::string& why){
    bool failed=false;try{f();}catch(const std::exception&){failed=true;}require(failed,why);
}
template<class F>void fill(ScalarField& f,F fun){
    for(int j=-f.ghost;j<f.mesh.n[1]+f.ghost;++j)
        for(int i=-f.ghost;i<f.mesh.n[0]+f.ghost;++i)
            f.curr[f.index(i,j)]=Real(fun(f.mesh.coord(0,i),f.mesh.coord(1,j)));
}
void upload(ScalarField& f){f.allocDevice();f.uploadAllToDevice();}
std::vector<Real> evaluate(const sym::Discrete& d,ScalarField& out,bool gpu,
                           num::Fusion fusion=num::Fusion::Auto){
    Equation e(out);e.setRHS(d.lower(fusion));
    if(gpu){e.computeRHS(out);out.downloadCurrFromDevice();}else e.computeRHSCPU(out);
    return out.curr;
}
template<class F>void compare(const sym::Discrete& d,ScalarField& out,F reference){
    for(auto fusion:{num::Fusion::Off,num::Fusion::Auto,num::Fusion::Required})
        for(bool gpu:{false,true}){
            evaluate(d,out,gpu,fusion);
            for(int j=0;j<out.mesh.n[1];++j)for(int i=0;i<out.mesh.n[0];++i)
                close(out.curr[out.index(i,j)],reference(i,j),"composite CPU/CUDA reference");
        }
}

// 1. Verify the published off-lattice gradients independently through moments.
void reconstructionMoments(){
    auto recipe=sym::FluxStencil::isotropic(true);
    std::set<std::pair<int,int>> footprint{{0,0}};
    for(const auto& link:recipe.links){
        for(const auto* weights:{&link.coefficient,&link.gradientX,&link.gradientY})
            for(auto w:*weights)footprint.insert({w.x,w.y});
        for(int degree=0;degree<=4;++degree)for(int px=0;px<=degree;++px){
            int py=degree-px;
            for(int axis=0;axis<2;++axis){
                double value=0;
                for(auto w:axis==0?link.gradientX:link.gradientY)
                    value+=w.weight*std::pow(double(w.x),px)*std::pow(double(w.y),py);
                double x=.5*link.x,y=.5*link.y;
                double exact=axis==0?(px?px*std::pow(x,px-1)*std::pow(y,py):0):
                                       (py?py*std::pow(x,px)*std::pow(y,py-1):0);
                close(value,exact,"Appendix B polynomial moment",2e-13);
            }
        }
    }
    require(footprint.size()==21,"full normal-flux footprint must have 21 points");
}

// 2. Exact references for variable diffusion, squared gradient and arbitrary recipes.
void algebraAndRecipes(){
    auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,9,.4,0,7,.4,0);
    ScalarField a(mesh,"a",3),b(mesh,"b",3),out(mesh,"out",3);
    fill(a,[](double x,double y){return 1+.2*x+.1*y;});
    fill(b,[](double x,double y){return x*x+x*y+2*y*y;});out.fill(0);
    auto A=sym::field(a),B=sym::field(b);
    for(auto f:{&a,&b,&out})upload(*f);
    auto ji=Schemes::builtin(scheme::preset("Ji2022S21").defaults);
    auto axial=Schemes::builtin(scheme::preset("AxialCD2").defaults);
    for(const auto* schemes:{&ji,&axial}){
        auto d=sym::div(A*sym::grad(B)).expand("diffusion",sym::ExpansionMode::Structured).discretize(*schemes);
        require(d.kernelLaunches(num::Fusion::Auto)==1,"composite must fuse without a flux field");
        compare(d,out,[&](int i,int j){
            double x=mesh.coord(0,i),y=mesh.coord(1,j);
            return 6*(1+.2*x+.1*y)+.2*(2*x+y)+.1*(x+4*y);
        });
        auto g=sym::normSquared(sym::grad(B)).expand("gradient energy",sym::ExpansionMode::Structured).discretize(*schemes);
        compare(g,out,[&](int i,int j){
            double x=mesh.coord(0,i),y=mesh.coord(1,j);return std::pow(2*x+y,2)+std::pow(x+4*y,2);
        });
    }
    fill(b,[](double x,double y){return std::sin(.8*x)+.4*std::cos(.5*y)+.1*std::sin(x+y);});b.uploadAllToDevice();
    auto lap=sym::lap(B).expand().discretize(ji);
    auto expected=evaluate(lap,out,false);
    compare(sym::divGrad(1,B).expand().discretize(ji),out,[&](int i,int j){return expected[out.index(i,j)];});
    Equation gradientReference(out);gradientReference.setRHS(PhiX::gradSq(b,"Iso9"));
    gradientReference.computeRHSCPU(out);expected=out.curr;
    compare(sym::gradSq(B).expand().discretize(ji),out,[&](int i,int j){return expected[out.index(i,j)];});
    // A public recipe can be assembled without registering a model-specific class.
    auto custom=sym::weightedSum(A,{{-2,1,.25},{2,-1,.75}});
    compare(custom.expand().discretize(Schemes{}),out,[&](int i,int j){
        return .25*a.curr[a.index(i-2,j+1)]+.75*a.curr[a.index(i+2,j-1)];
    });
    auto nested=sym::divNormal(1+sym::dx(B)*sym::dx(B),B).expand();
    auto nestedDefaults=scheme::preset("Ji2022S21").defaults;
    nestedDefaults["gradient"]="CD4";
    auto nestedSchemes=Schemes::builtin(nestedDefaults);
    auto info=nested.inspect(nestedSchemes,out);
    require(info.reads.size()==1 && info.reads[0].halo==3 && info.reads[0].corners,
            "shifted coefficient derivative needs composed halo 3");
    auto cpu=evaluate(nested.discretize(nestedSchemes),out,false);
    compare(nested.discretize(nestedSchemes),out,[&](int i,int j){return cpu[out.index(i,j)];});
    auto original=sym::div(A*sym::grad(B));
    require(original.expand().requiredOperators().entries.size()==5,"legacy Direct inventory changed");
    require(original.expand("",sym::ExpansionMode::Structured).requiredOperators().entries.size()==1,
            "Structured should retain coefficient diffusion");
    require(sym::analytic(original).expand("",sym::ExpansionMode::Structured).requiredOperators().entries.size()==5,
            "analytic subtree must preserve product/chain expansion");
    auto normalOriginal=sym::div(A*sym::normalized(sym::grad(B),1e-6));
    for(const auto& op:sym::analytic(normalOriginal).expand("",sym::ExpansionMode::Structured).requiredOperators().entries)
        require(op.section!="divNormal","analytic normalized flux was incorrectly retained");
}

// 3. Local link cancellation, including the full normalized flux and flat fields.
void conservationAndNormals(){
    constexpr int n=12;const double h=2*M_PI/n;
    auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,n,h,0,n,h,0);
    ScalarField a(mesh,"a",2),b(mesh,"b",2),out(mesh,"out",2);
    fill(a,[](double x,double y){return 1+.2*std::sin(x)*std::cos(y);});
    fill(b,[](double x,double y){return std::sin(x+.17)+.3*std::cos(y+.29)+.2*std::sin(x+y+.43);});
    out.fill(0);for(auto f:{&a,&b,&out})upload(*f);
    auto A=sym::field(a),B=sym::field(b);
    for(auto preset:{"AxialCD2","Ji2022S21"}){
        auto schemes=Schemes::builtin(scheme::preset(preset).defaults);
        for(bool normal:{false,true}){
            auto expr=normal?sym::div(A*sym::normalized(sym::grad(B))):sym::div(A*sym::grad(B));
            auto d=expr.expand("",sym::ExpansionMode::Structured).discretize(schemes);
            auto reference=evaluate(d,out,false);
            double sum=0,scale=0;
            for(int j=0;j<n;++j)for(int i=0;i<n;++i){
                double v=reference[out.index(i,j)];sum+=v;scale+=std::abs(v);
            }
            require(std::abs(sum)<tol*std::max(1.,scale),std::string(preset)+(normal?" normal":" diffusion")+
                    " periodic link flux fails conservation: sum="+std::to_string(sum)+" scale="+std::to_string(scale));
            compare(d,out,[&](int i,int j){return reference[out.index(i,j)];});
        }
        // Grid-aligned stationary points exercise explicit normalization, with
        // bit-identical periodic ghosts rather than trigonometric roundoff.
        fill(b,[](double x,double y){return std::sin(x)+.3*std::cos(y)+.2*std::sin(x+y);});
        PeriodicBC px(mesh.facePatch(Axis::X,Side::LOW)),py(mesh.facePatch(Axis::Y,Side::LOW));
        applyBCsCPU2D(b,{&px,&py});b.uploadAllToDevice();
        auto regularized=sym::divNormal(A,B,1e-4).expand().discretize(schemes);
        auto stationary=evaluate(regularized,out,false);
        double residual=0;
        for(int j=0;j<n;++j)for(int i=0;i<n;++i)residual+=stationary[out.index(i,j)];
        close(residual,0,"regularized periodic stationary point conservation",50*tol);
        compare(regularized,out,[&](int i,int j){return stationary[out.index(i,j)];});
        fill(b,[](double,double){return 0.;});b.uploadAllToDevice();
        compare(sym::divNormal(A,B).expand().discretize(schemes),out,[](int,int){return 0.;});
        fill(b,[](double x,double y){return .2*x+.1*y;});b.uploadAllToDevice();
        compare(sym::divNormal(0,B).expand().discretize(schemes),out,[](int,int){return 0.;});
        // Explicit floor limits the normal; it is not an implicit model approximation.
        fill(a,[](double x,double){return 1+.1*x;});a.uploadAllToDevice();
        compare(sym::divNormal(A,B,1).expand().discretize(schemes),out,[](int,int){return .02;});
        fill(a,[](double x,double y){return 1+.2*std::sin(x)*std::cos(y);});a.uploadAllToDevice();
        fill(b,[](double x,double y){return std::sin(x+.17)+.3*std::cos(y+.29)+.2*std::sin(x+y+.43);});b.uploadAllToDevice();
    }
}

// 4. Interior convergence and rotation of a smooth one-dimensional profile.
void convergenceAndIsotropy(){
#ifndef PHIX_REAL_FLOAT
    for(bool normal:{false,true}){
        double lastIsoSpread=0;
        double previousError[2]={};
        for(double h:{.4,.2,.1}){
            auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,1,h,-.5*h,1,h,-.5*h);
            ScalarField a(mesh,"a",2),b(mesh,"b",2),out(mesh,"out",2);out.fill(0);
            double spreads[2]={};
            for(int kind=0;kind<2;++kind){
                auto schemes=Schemes::builtin(scheme::preset(kind?"Ji2022S21":"AxialCD2").defaults);
                double low=1e9,high=-1e9,error=0;
                for(int angle=0;angle<=6;++angle){
                    double c=std::cos(angle*M_PI/24),s=std::sin(angle*M_PI/24),phase=.37;
                    fill(a,[&](double x,double y){return 1+.3*std::cos(c*x+s*y+phase);});
                    fill(b,[&](double x,double y){double z=c*x+s*y+phase;return normal?z+.2*std::sin(z):std::sin(z);});
                    auto e=normal?sym::divNormal(sym::field(a),sym::field(b)):
                                  sym::divGrad(sym::field(a),sym::field(b));
                    auto result=evaluate(e.expand().discretize(schemes),out,false)[out.index(0,0)];
                    double exact=normal?-.3*std::sin(phase):-std::sin(phase)-.6*std::sin(phase)*std::cos(phase);
                    require(std::abs(result-exact)<h*h,"smooth composite consistency");
                    error=std::max(error,std::abs(result-exact));
                    low=std::min(low,double(result));high=std::max(high,double(result));
                }
                spreads[kind]=high-low;
                if(previousError[kind])require(previousError[kind]/error>3.6,
                                               "overall composite convergence must be second order");
                previousError[kind]=error;
            }
            require(spreads[1]<.15*spreads[0],"Ji recipe did not suppress leading orientation error");
            if(lastIsoSpread)require(lastIsoSpread/spreads[1]>12,"isotropic angular error must converge as h^4");
            lastIsoSpread=spreads[1];
            std::cout<<(normal?"divNormal":"divGrad")<<" h="<<h<<" axial angular spread="<<spreads[0]
                     <<" Ji="<<spreads[1]<<'\n';
        }
    }
#endif
}

// 5. Read-only preparation, strict configuration round trip, and error reports.
void preparation(){
    auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,8,.5,0,7,.5,0);
    ScalarField a(mesh,"a",2),b(mesh,"b",2),out(mesh,"out",2);
    a.fill(.2);b.fill(.4);out.fill(0);
    PeriodicBC bx(mesh.facePatch(Axis::X,Side::LOW)),by(mesh.facePatch(Axis::Y,Side::LOW));
    auto A=sym::field(a),B=sym::field(b);
    sym::PreparationOptions options;options.preset=scheme::preset("Ji2022S21");
    num::System system(Schemes{},sym::ExpansionMode::Structured);
    system.bc(a,{&bx,&by});system.bc(b,{&bx,&by});
    system.add(sym::ddt(a)==sym::lap(A)+sym::normSquared(sym::grad(A))+
               sym::div((1-A)*sym::grad(B))+sym::div(A*sym::normalized(sym::grad(B))));
    auto initial=a.curr;auto prepared=system.prepare(options);
    require(prepared.report["ready"] && prepared.report["completeSymbolicInventory"],"valid preparation not ready");
    require(prepared.report["operators"].size()==5,"composite preparation operator inventory");
    require(!a.deviceAllocated() && a.curr==initial && system.step()==0,"preparation changed runtime state");
    require(prepared.schemes["divNormal"]["default"]=="none","draft must use explicit strict selections");
    auto config=Schemes::fromJson(prepared.schemes);
    num::System run(config,sym::ExpansionMode::Structured);
    run.bc(a,{&bx,&by});run.bc(b,{&bx,&by});
    run.add(sym::ddt(a)==sym::lap(A)+sym::normSquared(sym::grad(A))+
            sym::div((1-A)*sym::grad(B))+sym::div(A*sym::normalized(sym::grad(B))));
    run.compile({num::Backend::CPU});run.advance(.001);
    close(a.curr[a.index(3,3)],.2,"prepared configuration round trip");
    options.overrides={{"divNormal",{{"default","CD2"}}}};
    require(system.prepare(options).report["presetModified"],"customized preset was not reported");
    options.overrides={{"gradient",{{"grad(a)","CD4"}}}};
    auto gradient=sym::prepare(sym::dx(A).expand(),a,options);
    require(gradient.schemes["gradient"]["grad(a,x)"]=="CD4","generic gradient override ignored");
    ScalarField narrow(mesh,"narrow",1);
    auto missing=sym::prepare(sym::divNormal(1,sym::field(narrow)).expand(),narrow,{scheme::preset("Ji2022S21"),{}});
    require(!missing.report["ready"],"preparation must report insufficient halo");
    num::System noBC(config,sym::ExpansionMode::Structured);noBC.add(sym::ddt(a)==sym::divNormal(1,A));
    require(!noBC.prepare({scheme::preset("Ji2022S21"),{}}).report["ready"],"missing boundary contract accepted");
    num::System cyclic(Schemes{});cyclic.define(a,B);cyclic.define(b,A);
    require(!cyclic.prepare({}).report["ready"],"cyclic definitions accepted by preparation");
    num::System opaque(Schemes{});opaque.define(out,num::Spatial(Schemes{}).value(a));
    require(!opaque.prepare({}).report["completeSymbolicInventory"],"opaque expression claimed a complete inventory");
    auto rect=Mesh::makeUniform2D(CoordSys::CARTESIAN,4,.4,0,4,.5,0);
    ScalarField r(rect,"r",2);
    require(!sym::prepare(sym::divNormal(1,sym::field(r)).expand(),r,
                         {scheme::preset("Ji2022S21"),{}}).report["ready"],"rectangular Ji grid accepted");
    const auto root=std::filesystem::temp_directory_path()/
        ("phix-preparation-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    prepared.write(root.string());
    std::ifstream file(root/"schemes.jsonc");nlohmann::json disk;file>>disk;
    require(disk==prepared.schemes,"written scheme draft mismatch");
    rejects([&]{prepared.write(root.string());},"preparation overwrote an existing draft");
    std::filesystem::remove_all(root);
}

// 6. Wide reflection, periodic/reflection corners, and actual CPU/CUDA evolution.
void reflectedBoundaries(){
    constexpr int nx=8,ny=7,g=2;
    auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,nx,.4,0,ny,.4,0);
    NoFluxBC xl(mesh.facePatch(Axis::X,Side::LOW),NoFluxBC::Closure::Reflect),
             xh(mesh.facePatch(Axis::X,Side::HIGH),NoFluxBC::Closure::Reflect),
             yl(mesh.facePatch(Axis::Y,Side::LOW),NoFluxBC::Closure::Reflect),
             yh(mesh.facePatch(Axis::Y,Side::HIGH),NoFluxBC::Closure::Reflect);
    PeriodicBC periodicY(mesh.facePatch(Axis::Y,Side::LOW));
    ScalarField f(mesh,"f",g);f.fill(-1234);
    for(int j=0;j<ny;++j)for(int i=0;i<nx;++i)f.curr[f.index(i,j)]=Real(.5+.01*i*i+.03*j);
    const auto initial=f.curr;
    auto mirror=[](int i,int n){return i<0?-i-1:i>=n?2*n-i-1:i;};
    for(bool periodic:{false,true}){
        std::vector<BoundaryCondition*> bcs=periodic?std::vector<BoundaryCondition*>{&xl,&xh,&periodicY}:
                                                            std::vector<BoundaryCondition*>{&xl,&xh,&yl,&yh};
        for(bool gpu:{false,true}){
            f.curr=initial;upload(f);
            if(gpu){BCBatch batch;batch.build(f,bcs);batch.applyOnGPU(f);f.downloadCurrFromDevice();}
            else applyBCsCPU2D(f,bcs);
            for(int j=-g;j<ny+g;++j)for(int i=-g;i<nx+g;++i){
                int jj=periodic?(j+ny)%ny:mirror(j,ny);
                close(f.curr[f.index(i,j)],initial[f.index(mirror(i,nx),jj)],"reflected halo or corner",0);
            }
        }
        f.curr=initial;upload(f);
        for(auto* bc:bcs)bc->applyOnGPU(f);
        f.downloadCurrFromDevice();
        for(int j=0;j<ny;++j)for(int i:{-2,-1,nx,nx+1})
            close(f.curr[f.index(i,j)],initial[f.index(mirror(i,nx),j)],"standalone reflected GPU edge",0);
        std::vector<Real> reference;
        for(auto backend:{num::Backend::CPU,num::Backend::CUDA}){
            f.curr=initial;upload(f);auto F=sym::field(f);
            auto schemes=Schemes::builtin(scheme::preset("Ji2022S21").defaults);
            num::System system(schemes,sym::ExpansionMode::Structured);
            system.bc(f,bcs);system.add(sym::ddt(f)==.001*sym::divNormal(F,F));
            require(system.prepare({scheme::preset("Ji2022S21"),{}}).report["ready"],"reflected closure not accepted");
            system.compile({backend});system.advance(.01);
            if(backend==num::Backend::CUDA)f.downloadCurrFromDevice();
            if(reference.empty())reference=f.curr;
            for(int j=0;j<ny;++j)for(int i=0;i<nx;++i)
                close(f.curr[f.index(i,j)],reference[f.index(i,j)],"wide-halo evolution CPU/CUDA");
        }
    }
    // The legacy constant extension remains explicit and unchanged.
    NoFluxBC old(mesh.facePatch(Axis::X,Side::LOW));f.curr=initial;old.applyOnCPU(f);
    close(f.curr[f.index(-2,2)],initial[f.index(0,2)],"legacy NoFlux changed",0);
    auto config=nlohmann::json{
        {"x_min",{{"type","NoFlux"},{"closure","reflect"}}},
        {"x_max",{{"type","NoFlux"},{"closure","reflect"}}},
        {"y_min","Periodic"},{"y_max","Periodic"}};
    auto factory=buildBCs(mesh,config);f.curr=initial;applyBCsCPU2D(f,factory.ptrs);
    close(f.curr[f.index(-2,-2)],initial[f.index(1,ny-2)],"JSON reflected boundary",0);
    config["x_min"]["closure"]="typo";
    rejects([&]{buildBCs(mesh,config);},"unknown NoFlux closure accepted");
}
}
int main(){
    try{
        reconstructionMoments();algebraAndRecipes();conservationAndNormals();
        convergenceAndIsotropy();preparation();reflectedBoundaries();
        std::cout<<"composite recipes, Ji moments/isotropy/conservation, preparation, CPU/CUDA and wide boundaries passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
