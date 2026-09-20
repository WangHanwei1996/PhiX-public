#include "numerics/Preparation.h"
#include "boundary/PeriodicBC.h"
#include "core/Version.h"
#include <cmath>
#include <iostream>

using namespace PhiX;
namespace num=PhiX::numerics;
namespace sym=num::symbolic;

int main(int argc,char** argv){
    try{
        std::cout<<versionString()<<'\n';
        // 1. Choose inspection or execution. Outputs need an explicit case path.
        std::string prepareDirectory,schemePath,presetName="Ji2022S21";
        num::Backend backend=num::Backend::CPU;
        for(int i=1;i<argc;++i){
            std::string arg=argv[i];
            auto value=[&](){
                if(++i>=argc)throw std::invalid_argument("missing value for "+arg);
                return std::string(argv[i]);
            };
            if(arg=="--prepare")prepareDirectory=value();
            else if(arg=="--schemes")schemePath=value();
            else if(arg=="--preset")presetName=value();
            else if(arg=="--backend"){
                auto name=value();
                if(name!="cpu" && name!="cuda")throw std::invalid_argument("backend must be cpu or cuda");
                backend=name=="cpu"?num::Backend::CPU:num::Backend::CUDA;
            }else throw std::invalid_argument("unknown option: "+arg);
        }
        if(!prepareDirectory.empty() && !schemePath.empty())
            throw std::invalid_argument("choose --prepare or --schemes");
        const auto& preset=scheme::preset(presetName);
        auto schemes=schemePath.empty()?Schemes::builtin(preset.defaults):Schemes::fromFile(schemePath);

        // 2. Host layout and boundary declarations; no device allocation yet.
        constexpr int n=32;const double h=2*M_PI/n;
        auto mesh=Mesh::makeUniform2D(CoordSys::CARTESIAN,n,h,0,n,h,0);
        ScalarField u(mesh,"u",2),v(mesh,"v",2);
        PeriodicBC bx(mesh.facePatch(Axis::X,Side::LOW)),by(mesh.facePatch(Axis::Y,Side::LOW));
        u.fill(0);v.fill(0);
        for(int j=0;j<n;++j)for(int i=0;i<n;++i){
            double x=mesh.coord(0,i),y=mesh.coord(1,j);
            u.curr[u.index(i,j)]=Real(.2+.02*std::sin(x)*std::cos(y));
            v.curr[v.index(i,j)]=Real(.3+.03*std::cos(x+y));
        }

        // 3. Original mathematical formulas; Structured retains selectable composites.
        num::System system(schemes,sym::ExpansionMode::Structured);
        auto U=sym::field(u),V=sym::field(v);
        system.add(sym::ddt(u)==.05*sym::lap(U)+.01*sym::normSquared(sym::grad(U)));
        system.add(sym::ddt(v)==.05*sym::div((1-U)*sym::grad(V))+
                    .00001*sym::div(U*sym::normalized(sym::grad(V),1e-6)));
        system.bc(u,{&bx,&by});system.bc(v,{&bx,&by});

        // 4. Optional preparation returns the editable scheme draft and diagnostics.
        if(!prepareDirectory.empty()){
            auto draft=system.prepare({preset,{}});
            draft.write(prepareDirectory);
            std::cout<<"Prepared "<<prepareDirectory<<" (ready="<<draft.report["ready"]<<")\n";
            std::cout<<draft.report.dump(2)<<'\n';
            return draft.report["ready"].get<bool>()?0:2;
        }

        // 5. Bind the selected recipes, fuse, then execute a small verification run.
        if(backend==num::Backend::CUDA)
            for(auto f:{&u,&v}){f->allocDevice();f->uploadAllToDevice();}
        system.compile({backend,num::Fusion::Auto,num::Coupling::SameLevel});
        system.reportEquations(std::cout);system.report(std::cout);
        for(int step=0;step<10;++step)system.advance(.001);
        if(backend==num::Backend::CUDA){u.downloadCurrFromDevice();v.downloadCurrFromDevice();}
        double sumU=0,sumV=0;
        for(int j=0;j<n;++j)for(int i=0;i<n;++i){
            double a=u.curr[u.index(i,j)],b=v.curr[v.index(i,j)];
            if(!std::isfinite(a) || !std::isfinite(b))throw std::runtime_error("non-finite demo result");
            sumU+=a;sumV+=b;
        }
        std::cout<<"step="<<system.step()<<" mean(u)="<<sumU/(n*n)<<" mean(v)="<<sumV/(n*n)<<'\n';
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
