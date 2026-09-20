#include "SymbolicInternal.h"
#include "numerics/Stencil.h"
#include <cmath>
#include <map>

namespace PhiX::scheme::detail {
namespace {
template<CompositeMethod M> CompositeMethod method(){return M;}
}
const std::vector<Entry<CompositeFactory>>& divGradFactories(){
    static const std::vector<Entry<CompositeFactory>> entries={
        {{"CD2",2,1,false,Grid2D::Rectangular,"CD2","axial conservative coefficient-gradient flux"},
          &method<CompositeMethod::Axial>},
        {{"Iso9",2,1,true,Grid2D::SquareRequired,"Iso9","Ji 2022 Eq. 27-29, eight links and coefficient reconstruction"},
          &method<CompositeMethod::Isotropic>}};
    return entries;
}
const std::vector<Entry<CompositeFactory>>& divNormalFactories(){
    static const std::vector<Entry<CompositeFactory>> entries={
        {{"CD2",2,1,true,Grid2D::Rectangular,"CD2","axial normalized gradient flux"},
          &method<CompositeMethod::Axial>},
        {{"Iso21",2,2,true,Grid2D::SquareRequired,"Iso21","Ji 2022 Appendix B, full normalized flux without equilibrium approximation"},
          &method<CompositeMethod::IsotropicNormal>},
        {{"Iso21Vector",2,2,true,Grid2D::SquareRequired,"Iso21Vector","isotropic conservative vector flux; shared fourth-order gradient in numerator and norm"},
          &method<CompositeMethod::IsotropicNormalVector>}};
    return entries;
}
} // namespace PhiX::scheme::detail

namespace PhiX::numerics::symbolic {
Expr weightedSum(Expr value,const LinearStencil& weights){
    Expr sum=0;
    for(auto w:weights){
        if(!std::isfinite(w.weight))throw std::invalid_argument("stencil weights must be finite");
        sum=sum+w.weight*sample(value,w.x,w.y);
    }
    return sum;
}
Expr weightedDifferences(Expr value,const LinearStencil& weights){
    if(weights.empty())throw std::invalid_argument("difference stencil needs weights");
    double total=0,scale=0;
    for(auto w:weights){
        if(!std::isfinite(w.weight))throw std::invalid_argument("stencil weights must be finite");
        total+=w.weight;scale+=std::abs(w.weight);
    }
    if(std::abs(total)>1e-13*scale)
        throw std::invalid_argument("difference stencil weights must sum to zero");
    const auto reference=sample(value,weights.front().x,weights.front().y);
    Expr result=0;
    for(auto w:weights)result=result+w.weight*(sample(value,w.x,w.y)-reference);
    return result;
}
namespace {
// Rotate or reflect a stencil about the receiving cell. The off-lattice
// sampling position moves with it. Gradient components transform as vectors.
FluxLink transform(const FluxLink& base,int xx,int xy,int yx,int yy){
    FluxLink out{xx*base.x+xy*base.y,yx*base.x+yy*base.y,base.weight,{},{},{}};
    auto append=[&](LinearStencil& to,const LinearStencil& from,double scale){
        if(scale==0)return;
        for(auto w:from)to.push_back({xx*w.x+xy*w.y,yx*w.x+yy*w.y,scale*w.weight});
    };
    append(out.coefficient,base.coefficient,1);
    append(out.gradientX,base.gradientX,xx);append(out.gradientX,base.gradientY,xy);
    append(out.gradientY,base.gradientX,yx);append(out.gradientY,base.gradientY,yy);
    return out;
}
FluxLink opposite(const FluxLink& base){
    auto out=base;out.x=-base.x;out.y=-base.y;
    for(auto* weights:{&out.coefficient,&out.gradientX,&out.gradientY})
        for(auto& w:*weights){w.x-=base.x;w.y-=base.y;}
    return out;
}
void pair(std::vector<FluxLink>& to,const FluxLink& link){
    // Translation preserves identical arithmetic at the shared half-point.
    // Merely reflecting/reordering equal weights needlessly amplifies roundoff
    // near a vanishing normalization denominator.
    to.push_back(link);to.push_back(opposite(link));
}
void fourAxes(std::vector<FluxLink>& to,const FluxLink& x){
    pair(to,x);pair(to,transform(x,0,1,1,0));
}
// Appendix B Eq. 80-81: gradient at (1/2,0), with zero O(h^2) error.
void highOrderAxial(FluxLink& e){
    e.gradientX={{1,1,35./48},{0,-1,-35./48},{0,1,-35./48},{1,-1,35./48},
                 {2,1,-1./48},{-1,-1,1./48},{-1,1,1./48},{2,-1,-1./48},
                 {1,2,-1./6},{0,-2,1./6},{0,2,1./6},{1,-2,-1./6}};
    e.gradientY={{1,1,35./96},{0,-1,-35./96},{0,1,35./96},{1,-1,-35./96},
                 {2,1,-1./32},{-1,-1,1./32},{-1,1,-1./32},{2,-1,1./32},
                 {1,2,-1./24},{0,-2,1./24},{0,2,-1./24},{1,-2,1./24}};
}
// Appendix B Eq. 83-84: gradient at (1/2,1/2).
void highOrderDiagonal(FluxLink& e){
    e.gradientX={{1,1,5./8},{0,0,-5./8},{0,1,-5./8},{1,0,5./8},
                 {2,1,-1./48},{-1,0,1./48},{-1,1,1./48},{2,0,-1./48},
                 {1,2,-1./16},{0,-1,1./16},{0,2,1./16},{1,-1,-1./16}};
    e.gradientY={{1,1,5./8},{0,0,-5./8},{0,1,5./8},{1,0,-5./8},
                 {2,1,-1./16},{-1,0,1./16},{-1,1,-1./16},{2,0,1./16},
                 {1,2,-1./48},{0,-1,1./48},{0,2,-1./48},{1,-1,1./48}};
}
}
FluxStencil FluxStencil::axial(){
    FluxStencil result;
    FluxLink east{1,0,1,{{0,0,.5},{1,0,.5}},
                  {{0,0,-1},{1,0,1}},
                  {{0,1,.25},{1,1,.25},{0,-1,-.25},{1,-1,-.25}}};
    fourAxes(result.links,east);
    return result;
}
FluxStencil FluxStencil::isotropic(bool highOrderNormal){
    FluxStencil result;result.squareRequired=true;
    FluxLink east{1,0,2./3,
        {{0,0,3./8},{1,0,3./8},{0,1,1./16},{1,1,1./16},{0,-1,1./16},{1,-1,1./16}},
        {{0,0,-1},{1,0,1}},{{0,1,.25},{1,1,.25},{0,-1,-.25},{1,-1,-.25}}};
    if(highOrderNormal)highOrderAxial(east);
    fourAxes(result.links,east);
    FluxLink diagonal{1,1,1./3,{{0,0,.25},{1,0,.25},{0,1,.25},{1,1,.25}},
        {{1,1,.5},{0,1,-.5},{1,0,.5},{0,0,-.5}},
        {{1,1,.5},{1,0,-.5},{0,1,.5},{0,0,-.5}}};
    if(highOrderNormal)highOrderDiagonal(diagonal);
    pair(result.links,diagonal);pair(result.links,transform(diagonal,-1,0,0,1));
    return result;
}
Expr FluxStencil::apply(Expr a,Expr b,double hx,double hy,bool normalize,double floor)const{
    if(!std::isfinite(hx) || !std::isfinite(hy) || hx<=0 || hy<=0)
        throw std::invalid_argument("flux stencil requires positive finite spacings");
    if(squareRequired && std::abs(hx-hy)>1e-12*std::max(hx,hy))
        throw std::invalid_argument("isotropic flux stencil requires square cells");
    if(!std::isfinite(floor) || floor<0)throw std::invalid_argument("invalid normal floor");
    if(links.empty())throw std::invalid_argument("flux stencil needs links");
    Expr result=0;
    for(const auto& link:links){
        if((link.x==0 && link.y==0) || !std::isfinite(link.weight))
            throw std::invalid_argument("invalid flux link");
        auto amplitude=weightedSum(a,link.coefficient);
        Expr gx,gy;
        if(normalize || projection==FluxProjection::ReconstructedGradient){
            if(link.gradientX.empty() || link.gradientY.empty())
                throw std::invalid_argument("vector flux requires gradient reconstruction");
            if(projection==FluxProjection::ReconstructedGradient){
                gx=weightedDifferences(b,link.gradientX)/hx;
                gy=weightedDifferences(b,link.gradientY)/hy;
            }else{
                gx=weightedSum(b,link.gradientX)/hx;
                gy=weightedSum(b,link.gradientY)/hy;
            }
        }
        const auto directional=projection==FluxProjection::LinkDifference ?
            sample(b,link.x,link.y)-b : (link.x*hx)*gx+(link.y*hy)*gy;
        auto numerator=amplitude*directional;
        Expr flux=numerator;
        if(normalize){
            auto norm=hypot(gx,gy);
            flux=where((abs(amplitude)>0) && (norm>0),numerator/max(norm,floor),0);
        }
        const double length2=link.x*link.x*hx*hx+link.y*link.y*hy*hy;
        result=result+(link.weight/length2)*flux;
    }
    return result;
}

namespace detail {
Expr lowerComposites(Expr expression,const Schemes& schemes,const ScalarField& layout,OperationInfo& info){
    std::map<const Node*,Expr> memo;
    std::function<Expr(Expr)> lower=[&](Expr e){
        if(auto it=memo.find(e.node().get());it!=memo.end())return it->second;
        const auto& n=*e.node();
        std::vector<Expr> args;for(auto a:n.args)args.push_back(lower(a));
        Expr result;
        if(composite(n)){
            auto choice=schemes.select(section(n),key(n));
            info.schemes.push_back(choice.describe());
            const double hx=layout.mesh.d[0],hy=layout.mesh.d[1];
            if(n.op==Op::GradientSquare){
                const auto& entry=scheme::detail::lookup(scheme::detail::gradSqFactories(),choice.name,"gradient square");
                const bool iso=std::string(entry.descriptor.name)=="Iso9";
                if(iso && std::abs(hx-hy)>1e-12*std::max(hx,hy))
                    throw std::invalid_argument("isotropic gradient square requires square cells");
                auto b=args[0],gx=(sample(b,1,0)-sample(b,-1,0))/(2*hx),
                     gy=(sample(b,0,1)-sample(b,0,-1))/(2*hy);
                result=gx*gx+gy*gy;
                if(iso){
                    auto d1=sample(b,1,1)-sample(b,-1,-1),d2=sample(b,-1,1)-sample(b,1,-1);
                    result=(2./3)*result+(1./3)*(d1*d1+d2*d2)/(8*hx*hx);
                }
            }else{
                bool normal=n.op==Op::DivNormal;
                const auto& entries=normal?scheme::detail::divNormalFactories():scheme::detail::divGradFactories();
                auto method=scheme::detail::lookup(entries,choice.name,"composite flux").factory();
                auto recipe=method==scheme::detail::CompositeMethod::Axial?FluxStencil::axial():
                    FluxStencil::isotropic(method==scheme::detail::CompositeMethod::IsotropicNormal ||
                                          method==scheme::detail::CompositeMethod::IsotropicNormalVector);
                if(method==scheme::detail::CompositeMethod::IsotropicNormalVector)
                    recipe.projection=FluxProjection::ReconstructedGradient;
                result=recipe.apply(args[0],args[1],hx,hy,normal,n.value);
            }
        }else result=make(n.op,args,n.value,n.field,n.x,n.y,n.name);
        memo.emplace(e.node().get(),result);return result;
    };
    return canonicalize(lower(expression));
}
} // namespace detail
} // namespace PhiX::numerics::symbolic
