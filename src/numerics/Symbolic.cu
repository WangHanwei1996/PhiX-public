#include "SymbolicInternal.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <tuple>

namespace PhiX::numerics::symbolic {
using detail::Op;
namespace {
bool constant(const Expr& e, double v) {
    return e.node()->op == Op::Constant && e.node()->value == v;
}
std::string number(double v) {
    std::ostringstream s; s << std::setprecision(17) << v; return s.str();
}
}
namespace detail {
Expr make(Op op, std::vector<Expr> args, double value, const ScalarField* field,
          int x, int y, std::string name) {
    if (op == Op::Add) {
        if (constant(args[0],0)) return args[1];
        if (constant(args[1],0)) return args[0];
        for (int i=0;i<2;++i)
            if (args[i].node()->op==Op::Negate &&
                args[i].node()->args[0].node()==args[1-i].node()) return Expr(0);
    }
    if (op == Op::Mul) {
        if (constant(args[0],0) || constant(args[1],0)) return Expr(0);
        if (constant(args[0],1)) return args[1];
        if (constant(args[1],1)) return args[0];
    }
    if (op == Op::Divide) {
        if (constant(args[1],0)) throw std::invalid_argument("symbolic: division by constant zero");
        if (constant(args[0],0)) return Expr(0);
        if (constant(args[1],1)) return args[0];
    }
    if (op == Op::Negate && args[0].node()->op == Op::Negate) return args[0].node()->args[0];
    if (op == Op::Pow) {
        if (value==0) return Expr(1);
        if (value==1) return args[0];
    }
    if (op == Op::Where && args[0].node()->op == Op::Constant)
        return args[args[0].node()->value != 0 ? 1 : 2];
    if (op == Op::Where && args[1].node()==args[2].node()) return args[1];
    if (op == Op::Negate && args[0].node()->op==Op::Constant) return Expr(-args[0].node()->value);
    if ((op==Op::Add || op==Op::Mul || op==Op::Divide) &&
        args[0].node()->op==Op::Constant && args[1].node()->op==Op::Constant) {
        double a=args[0].node()->value,b=args[1].node()->value;
        return Expr(op==Op::Add?a+b:op==Op::Mul?a*b:a/b);
    }
    return Expr(std::make_shared<Node>(Node{op,std::move(args),value,field,x,y,std::move(name)}));
}
std::vector<Expr> nodes(const Expr& expression) {
    std::vector<Expr> out; std::set<const Node*> seen;
    std::function<void(Expr)> visit = [&](Expr e) {
        if (!seen.insert(e.node().get()).second) return;
        for (const auto& a:e.node()->args) visit(a);
        out.push_back(e);
    };
    visit(expression); return out;
}
std::string key(const Node& n) {
    if (composite(n)) {
        if(!n.name.empty()) return n.name;
        const auto family=section(n);
        std::string k=family+"(";
        const auto count=n.op==Op::GradientSquare?1:n.args.size();
        for(std::size_t i=0;i<count;++i) k+=(i?",":"")+n.args[i].str();
        return k+")";
    }
    if (n.op!=Op::Jet) throw std::logic_error("not an expanded derivative");
    if (n.x<0) return "lap("+n.field->name+")";
    if (n.x+n.y==1) return "grad("+n.field->name+","+(n.x?"x":"y")+")";
    return "d"+std::string(n.x,'x')+std::string(n.y,'y')+"("+n.field->name+")";
}
std::string section(const Node& n) {
    if(n.op==Op::GradientSquare) return "gradSq";
    if(n.op==Op::DivGrad) return "divGrad";
    if(n.op==Op::DivNormal) return "divNormal";
    return n.x<0?"laplacian":n.x+n.y==1?"gradient":"secondDerivative";
}
bool composite(const Node& n) {
    return n.op==Op::GradientSquare || n.op==Op::DivGrad || n.op==Op::DivNormal;
}
Expr canonicalize(Expr expression) {
    std::map<const Node*,Expr> memo;
    std::map<const Node*,std::size_t> ids;
    std::map<std::string,Expr> intern;
    std::function<Expr(Expr)> visit=[&](Expr e) {
        if(auto it=memo.find(e.node().get());it!=memo.end())return it->second;
        auto n=*e.node();
        for(auto& a:n.args)a=visit(a);
        auto folded=make(n.op,n.args,n.value,n.field,n.x,n.y,n.name);
        const auto& f=*folded.node();
        std::ostringstream key;
        key<<int(f.op)<<':'<<number(f.value)<<':'<<f.field<<':'<<f.x<<':'<<f.y<<':'<<f.name;
        for(auto a:f.args)key<<':'<<ids.at(a.node().get());
        auto it=intern.find(key.str());
        Expr result=it==intern.end()?folded:it->second;
        if(it==intern.end()){ids[result.node().get()]=ids.size();intern.emplace(key.str(),result);}
        memo.emplace(e.node().get(),result);
        return result;
    };
    return visit(expression);
}
} // namespace detail
Expr::Expr(double v) {
    if (!std::isfinite(v)) throw std::invalid_argument("symbolic: constants must be finite");
    node_=std::make_shared<detail::Node>(detail::Node{Op::Constant,{},v});
}
Expr field(const ScalarField& f) { return detail::make(Op::Field,{},0,&f); }
Expr parameter(const ScalarField& f) { return detail::make(Op::Parameter,{},0,&f); }
Expr coordinate(int axis) {
    if(axis<0 || axis>1) throw std::invalid_argument("symbolic coordinate: axis must be x or y");
    return detail::make(Op::Coordinate,{},0,nullptr,axis);
}
Expr named(const std::string& name, Expr e) {
    if(name.empty()) throw std::invalid_argument("symbolic alias needs a name");
    return detail::make(Op::Alias,{e},0,nullptr,0,0,name);
}
Expr operator+(Expr a,Expr b){return detail::make(Op::Add,{a,b});}
Expr operator-(Expr a){return detail::make(Op::Negate,{a});}
Expr operator-(Expr a,Expr b){return a+(-b);}
Expr operator*(Expr a,Expr b){return detail::make(Op::Mul,{a,b});}
Expr operator/(Expr a,Expr b){return detail::make(Op::Divide,{a,b});}
#define PHIX_SYMBOLIC_UNARY(name,op) Expr name(Expr a){return detail::make(Op::op,{a});}
PHIX_SYMBOLIC_UNARY(sin,Sin) PHIX_SYMBOLIC_UNARY(cos,Cos)
PHIX_SYMBOLIC_UNARY(tanh,Tanh) PHIX_SYMBOLIC_UNARY(exp,Exp)
PHIX_SYMBOLIC_UNARY(log,Log) PHIX_SYMBOLIC_UNARY(sqrt,Sqrt) PHIX_SYMBOLIC_UNARY(abs,Abs)
#undef PHIX_SYMBOLIC_UNARY
Expr pow(Expr a,double p){
    if(!std::isfinite(p))throw std::invalid_argument("symbolic power must be finite");
    return detail::make(Op::Pow,{a},p);
}
Expr atan2(Expr y,Expr x){return detail::make(Op::Atan2,{y,x});}
#define PHIX_SYMBOLIC_BINARY(token,op) Expr operator token(Expr a,Expr b){return detail::make(Op::op,{a,b});}
PHIX_SYMBOLIC_BINARY(<,Less) PHIX_SYMBOLIC_BINARY(<=,LessEqual)
PHIX_SYMBOLIC_BINARY(>,Greater) PHIX_SYMBOLIC_BINARY(>=,GreaterEqual)
PHIX_SYMBOLIC_BINARY(&&,And) PHIX_SYMBOLIC_BINARY(||,Or)
#undef PHIX_SYMBOLIC_BINARY
Expr where(Expr c,Expr a,Expr b){return detail::make(Op::Where,{c,a,b});}
Expr min(Expr a,Expr b){return where(a<b,a,b);}
Expr max(Expr a,Expr b){return where(a>b,a,b);}
Expr derivative(Expr a,int axis){
    if(axis<0 || axis>1)throw std::invalid_argument("symbolic derivative: axis must be x or y");
    return detail::make(Op::Derivative,{a},0,nullptr,axis);
}
Expr dx(Expr a){return derivative(a,0);} Expr dy(Expr a){return derivative(a,1);}
Expr dxx(Expr a){return dx(dx(a));} Expr dyy(Expr a){return dy(dy(a));}
Expr dxy(Expr a){return dx(dy(a));}
Expr lap(Expr a){return detail::make(Op::Laplacian,{a});}
Expr gradSq(Expr a){return detail::make(Op::GradientSquare,{a});}
Expr divGrad(Expr a,Expr b){return detail::make(Op::DivGrad,{a,b});}
Expr divNormal(Expr a,Expr b,double floor){
    if(!std::isfinite(floor) || floor<0)throw std::invalid_argument("normal floor must be finite and nonnegative");
    return detail::make(Op::DivNormal,{a,b},floor);
}
Expr analytic(Expr a){return detail::make(Op::Analytic,{a});}
Expr sample(Expr a,int x,int y){
    if(x==0 && y==0)return a;
    return detail::make(Op::Shift,{a},0,nullptr,x,y);
}
Expr hypot(Expr a,Expr b){return detail::make(Op::Hypot,{a,b});}
Expr partial(Expr a,Expr b){return detail::make(Op::Partial,{a,b});}
Vector grad(Expr a){
    Vector v(dx(a),dy(a));v.divergence_=std::make_shared<Expr>(lap(a));
    v.potential_=std::make_shared<Expr>(a);v.coefficient_=std::make_shared<Expr>(1);return v;
}
Vector normalized(Vector v,double floor){
    if(!v.potential_ || !v.coefficient_ || !constant(*v.coefficient_,1) || v.normalized_)
        throw std::invalid_argument("normalized expects grad(potential); multiply the coefficient afterwards");
    if(!std::isfinite(floor) || floor<0)throw std::invalid_argument("normal floor must be finite and nonnegative");
    auto n=hypot(v.x,v.y),den=max(n,floor);
    Vector r(where(n>0,v.x/den,0),where(n>0,v.y/den,0));
    r.potential_=v.potential_;r.coefficient_=v.coefficient_;r.normalized_=true;r.normFloor_=floor;
    return r;
}
Expr div(Vector v){
    std::vector<Expr> args={v.x,v.y,v.divergence_?*v.divergence_:dx(v.x)+dy(v.y)};
    if(v.potential_){args.push_back(*v.coefficient_);args.push_back(*v.potential_);}
    return detail::make(Op::Divergence,args,v.normFloor_,nullptr,v.normalized_?1:0);
}
Vector operator+(Vector a,Vector b){
    Vector v(a.x+b.x,a.y+b.y);v.divergence_=std::make_shared<Expr>(div(a)+div(b));return v;
}
Vector operator-(Vector a,Vector b){return a+Expr(-1)*b;}
Vector operator*(Expr s,Vector a){
    Vector v(s*a.x,s*a.y);v.divergence_=std::make_shared<Expr>(dx(s)*a.x+dy(s)*a.y+s*div(a));
    if(a.potential_){v.potential_=a.potential_;v.coefficient_=std::make_shared<Expr>(s*(*a.coefficient_));
        v.normalized_=a.normalized_;v.normFloor_=a.normFloor_;}
    return v;
}
Vector operator*(Vector a,Expr s){return s*a;}
Vector operator/(Vector a,Expr s){return (Expr(1)/s)*a;}
Vector perpendicular(Vector a){return Vector(-a.y,a.x);}
Vector rotate(Vector a,Expr angle){
    auto c=cos(angle),s=sin(angle);return Vector(c*a.x+s*a.y,-s*a.x+c*a.y);
}
Expr dot(Vector a,Vector b){return a.x*b.x+a.y*b.y;}
Expr normSquared(Vector a){
    if(a.potential_ && !a.normalized_ && constant(*a.coefficient_,1))
        return detail::make(Op::GradientSquare,{*a.potential_,dot(a,a)},0,nullptr,1);
    return dot(a,a);
}
Vector partial(Expr a,Vector b){return Vector(partial(a,b.x),partial(a,b.y));}
Hessian hessian(Expr a){return {dxx(a),dxy(a),dyy(a)};}
Hessian rotate(Hessian h,Expr angle){
    auto c=cos(angle),s=sin(angle);
    return {c*c*h.xx+2*c*s*h.xy+s*s*h.yy,
            c*s*(h.yy-h.xx)+(c*c-s*s)*h.xy,
            s*s*h.xx-2*c*s*h.xy+c*c*h.yy};
}
namespace {
// Each expansion owns its intern table. No global registry or dangling node keys.
class Expander {
    ExpansionMode mode_;
    std::map<std::string,Expr> intern_;
    std::map<std::shared_ptr<const detail::Node>,Expr> memo_;
    std::map<std::tuple<const detail::Node*,int,const detail::Node*>,Expr> derivatives_;
    std::map<const detail::Node*,std::size_t> ids_;
    Expr canonical(Expr e) {
        auto n=*e.node();
        for(auto& a:n.args)a=canonical(a);
        auto folded=detail::make(n.op,n.args,n.value,n.field,n.x,n.y,n.name);
        const auto& f=*folded.node();
        std::ostringstream s;s<<int(f.op)<<':'<<number(f.value)<<':'<<f.field<<':'<<f.x<<':'<<f.y<<':'<<f.name;
        for(const auto& a:f.args)s<<':'<<ids_.at(a.node().get());
        auto it=intern_.find(s.str());if(it!=intern_.end())return it->second;
        ids_[folded.node().get()]=ids_.size();intern_.emplace(s.str(),folded);return folded;
    }
    Expr diff(Expr e,int axis,const detail::Node* variable=nullptr) {
        e=canonical(e);
        auto key=std::make_tuple(e.node().get(),axis,variable);
        if(auto i=derivatives_.find(key);i!=derivatives_.end())return i->second;
        const auto& n=*e.node();
        auto d=[&](Expr a){return diff(a,axis,variable);};
        Expr result;
        if(variable && e.node().get()==variable) result=1;
        else switch(n.op){
        case Op::Constant: case Op::Parameter: result=0;break;
        case Op::Coordinate:result=variable?0:(n.x==axis?1:0);break;
        case Op::Field:
            result=variable?Expr(0):detail::make(Op::Jet,{},0,n.field,axis==0,axis==1);break;
        case Op::Jet:
            if(variable){result=0;break;}
            if(n.x<0 || n.x+n.y>=2)
                throw std::invalid_argument("symbolic expansion: derivative order exceeds supported second order at "+detail::key(n));
            result=detail::make(Op::Jet,{},0,n.field,n.x+(axis==0),n.y+(axis==1));break;
        case Op::Add:result=d(n.args[0])+d(n.args[1]);break;
        case Op::Negate:result=-d(n.args[0]);break;
        case Op::Mul:result=d(n.args[0])*n.args[1]+n.args[0]*d(n.args[1]);break;
        case Op::Divide:result=(d(n.args[0])*n.args[1]-n.args[0]*d(n.args[1]))/pow(n.args[1],2);break;
        case Op::Sin:result=cos(n.args[0])*d(n.args[0]);break;
        case Op::Cos:result=-sin(n.args[0])*d(n.args[0]);break;
        case Op::Tanh:result=(1-pow(tanh(n.args[0]),2))*d(n.args[0]);break;
        case Op::Exp:result=exp(n.args[0])*d(n.args[0]);break;
        case Op::Log:result=d(n.args[0])/n.args[0];break;
        case Op::Sqrt:result=d(n.args[0])/(2*sqrt(n.args[0]));break;
        case Op::Pow:result=n.value*pow(n.args[0],n.value-1)*d(n.args[0]);break;
        case Op::Abs:result=where(n.args[0]>0,d(n.args[0]),where(n.args[0]<0,-d(n.args[0]),0));break;
        case Op::Atan2:result=(n.args[1]*d(n.args[0])-n.args[0]*d(n.args[1]))/(pow(n.args[0],2)+pow(n.args[1],2));break;
        case Op::Hypot:result=(n.args[0]*d(n.args[0])+n.args[1]*d(n.args[1]))/hypot(n.args[0],n.args[1]);break;
        case Op::GradientSquare: {
            auto x=diff(n.args[0],0),y=diff(n.args[0],1);
            result=d(x*x+y*y);break;
        }
        case Op::DivGrad: case Op::DivNormal: case Op::Shift:
            throw std::invalid_argument("cannot differentiate a retained discrete operator; use analytic() before differentiation or introduce an auxiliary");
        case Op::Where:result=where(n.args[0],d(n.args[1]),d(n.args[2]));
            rules.insert("piecewise rule: differentiate branches; keep the condition at the receiving cell");break;
        default:throw std::invalid_argument("symbolic: cannot differentiate a predicate; place it in where(condition, yes, no)");
        }
        result=canonical(result);derivatives_.emplace(key,result);return result;
    }
public:
    explicit Expander(ExpansionMode mode=ExpansionMode::Direct):mode_(mode){}
    std::set<std::string> rules;
    Expr run(Expr e){
        if(auto i=memo_.find(e.node());i!=memo_.end())return i->second;
        const auto& n=*e.node();Expr r;
        if(n.op==Op::Analytic){
            Expander direct(ExpansionMode::Direct);
            r=direct.run(n.args[0]);rules.insert("explicit analytic expansion boundary");
        }else if(n.op==Op::GradientSquare){
            if(n.x==1 && mode_==ExpansionMode::Direct)r=run(n.args[1]);
            else {r=gradSq(run(n.args[0]));rules.insert("retain gradient square for operator-specific discretization");}
        }else if(n.op==Op::Derivative){
            rules.insert("spatial product/chain rules -> direct derivatives of input fields");
            r=diff(run(n.args[0]),n.x);
        }else if(n.op==Op::Laplacian){
            auto a=run(n.args[0]);
            if(a.node()->op==Op::Field)r=detail::make(Op::Jet,{},0,a.node()->field,-1,0);
            else {rules.insert("Laplacian of a composite -> second-order chain rule");r=diff(diff(a,0),0)+diff(diff(a,1),1);}
        }else if(n.op==Op::Divergence){
            if(n.args.size()==5 && mode_==ExpansionMode::Structured){
                auto a=run(n.args[3]),b=run(n.args[4]);
                if(n.x==1)r=divNormal(a,b,n.value);
                else if(constant(a,1))r=run(lap(n.args[4]));
                else r=divGrad(a,b);
                rules.insert("retain conservative gradient flux for scheme binding");
            }else{
                rules.insert("divergence product rule; div(grad(field)) retains the explicit Laplacian trace");
                r=run(n.args[2]);
            }
        }else if(n.op==Op::Partial){
            rules.insert("symbolic partial derivative with respect to the declared argument");
            auto a=run(n.args[0]),v=run(n.args[1]);
            if(v.node()->op==Op::Constant)throw std::invalid_argument("symbolic partial: argument is constant");
            r=diff(a,-1,v.node().get());
        }else if(n.op==Op::Alias){
            rules.insert("inline definition: "+n.name);r=run(n.args[0]);
        }else{
            std::vector<Expr> args;for(auto a:n.args)args.push_back(run(a));
            r=detail::make(n.op,args,n.value,n.field,n.x,n.y,n.name);
        }
        r=canonical(r);memo_.emplace(e.node(),r);return r;
    }
};
std::string format(const detail::Node& n,const std::vector<std::string>& a){
    switch(n.op){
    case Op::Constant:return number(n.value);
    case Op::Field:return n.field->name;
    case Op::Parameter:return "parameter("+n.field->name+")";
    case Op::Coordinate:return n.x==0?"x":"y";
    case Op::Jet:return detail::key(n);
    case Op::GradientSquare:return "gradSq("+a[0]+")";
    case Op::DivGrad:return "divGrad("+a[0]+", "+a[1]+")";
    case Op::DivNormal:return "divNormal("+a[0]+", "+a[1]+", normFloor="+number(n.value)+")";
    case Op::Shift:return "sample("+a[0]+","+std::to_string(n.x)+","+std::to_string(n.y)+")";
    case Op::Sample:return "sample("+n.field->name+","+std::to_string(n.x)+","+std::to_string(n.y)+")";
    case Op::Analytic:return "analytic("+a[0]+")";
    case Op::Alias:return n.name+"{"+a[0]+"}";
    case Op::Add:return "("+a[0]+" + "+a[1]+")";
    case Op::Mul:return "("+a[0]+" * "+a[1]+")";
    case Op::Divide:return "("+a[0]+" / "+a[1]+")";
    case Op::Negate:return "(-"+a[0]+")";
    case Op::Pow:return "pow("+a[0]+", "+number(n.value)+")";
    case Op::Derivative:return std::string(n.x==0?"dx(":"dy(")+a[0]+")";
    case Op::Laplacian:return "lap("+a[0]+")";
    case Op::Divergence:return "div({"+a[0]+", "+a[1]+"})";
    case Op::Partial:return "partial("+a[0]+", "+a[1]+")";
    default:break;
    }
    const std::map<Op,std::string> names={{Op::Sin,"sin"},{Op::Cos,"cos"},{Op::Tanh,"tanh"},
        {Op::Exp,"exp"},{Op::Log,"log"},{Op::Sqrt,"sqrt"},{Op::Abs,"abs"},{Op::Atan2,"atan2"},{Op::Hypot,"hypot"},
        {Op::Less,"lt"},{Op::LessEqual,"le"},{Op::Greater,"gt"},{Op::GreaterEqual,"ge"},
        {Op::And,"and"},{Op::Or,"or"},{Op::Where,"where"}};
    std::string r=names.at(n.op)+"(";
    for(std::size_t i=0;i<a.size();++i)r+=(i?", ":"")+a[i];return r+")";
}
}
std::string Expr::str()const{
    std::function<std::string(Expr)> f=[&](Expr e){
        std::vector<std::string> args;
        const auto& n=*e.node();
        if(n.op==Op::Divergence && n.args.size()==5){
            auto gradient="grad("+f(n.args[4])+")";
            if(n.x)gradient="normalized("+gradient+", "+number(n.value)+")";
            return "div("+f(n.args[3])+" * "+gradient+")";
        }
        std::size_t count=n.op==Op::Divergence?2:n.op==Op::GradientSquare?1:n.args.size();
        for(std::size_t i=0;i<count;++i)args.push_back(f(n.args[i]));
        return format(n,args);
    };return f(*this);
}
Expanded Expr::expand(const std::string& name,ExpansionMode mode)const{
    Expander x(mode);auto result=x.run(*this);return Expanded(*this,result,name,{x.rules.begin(),x.rules.end()});
}
Expanded::Expanded(Expr original,Expr result,std::string name,std::vector<std::string> trace,ScalarField* state)
    :original_(std::move(original)),result_(std::move(result)),name_(std::move(name)),trace_(std::move(trace)),state_(state){}
Expanded Equation::expand(const std::string& name,ExpansionMode mode)const{
    auto e=rhs.expand(name.empty()?"ddt("+state->name+")":name,mode);e.state_=state;return e;
}
OperatorInventory Expanded::requiredOperators()const{
    OperatorInventory out;std::set<std::string> seen;
    for(auto e:detail::nodes(result_)){
        const auto& n=*e.node();if(n.op!=Op::Jet && !detail::composite(n))continue;
        auto key=detail::key(n),section=detail::section(n);
        if(!seen.insert(section+"/"+key).second)continue;
        RequiredOperator r;r.section=section;r.key=key;r.field=n.field?n.field->name:"expression";r.sources={name_};
        r.derivativeOrder=detail::composite(n)?2:(n.x<0?2:n.x+n.y);
        r.direction=detail::composite(n)?"composite":n.x<0?"trace":std::string(n.x,'x')+std::string(n.y,'y');
        if(n.op==Op::DivGrad || n.op==Op::DivNormal)r.location="oriented links";
        for(const auto& d:scheme::catalog(scheme::findFamily(section)->id))r.supportedSchemes.push_back(d.name);
        out.entries.push_back(std::move(r));
    }
    if(state_){RequiredOperator r;r.section="ddt";r.key="ddt("+state_->name+")";r.field=state_->name;
        r.direction="t";r.derivativeOrder=1;r.sources={name_};r.supportedSchemes={"EULER"};out.entries.push_back(std::move(r));}
    std::sort(out.entries.begin(),out.entries.end(),[](const auto&a,const auto&b){return a.section+"/"+a.key<b.section+"/"+b.key;});return out;
}
void OperatorInventory::print(std::ostream& out)const{
    for(const auto& r:entries){out<<r.section<<"/"<<r.key<<" ["<<r.location<<", "<<r.frame<<", "<<r.timeLevel<<"] formats:";
        for(const auto& s:r.supportedSchemes)out<<' '<<s;out<<"; source:";for(const auto& s:r.sources)out<<' '<<s;out<<'\n';}
}
OperatorInventory& OperatorInventory::merge(const OperatorInventory& other){
    if(this==&other)return *this;
    for(const auto& item:other.entries){
        auto it=std::find_if(entries.begin(),entries.end(),[&](const auto& e){return e.section==item.section && e.key==item.key;});
        if(it==entries.end())entries.push_back(item);
        else for(const auto& source:item.sources)
            if(std::find(it->sources.begin(),it->sources.end(),source)==it->sources.end())it->sources.push_back(source);
    }
    std::sort(entries.begin(),entries.end(),[](const auto&a,const auto&b){return a.section+"/"+a.key<b.section+"/"+b.key;});
    return *this;
}
nlohmann::json OperatorInventory::schemeTemplate()const{
    nlohmann::json j={{"schema",2},{"policy","strict"}};
    for(const auto& r:entries){j[r.section]["default"]="none";j[r.section][r.key]=r.supportedSchemes.front();}return j;
}
void OperatorInventory::writeSchemeTemplate(const std::string& path)const{
    std::ofstream f(path);if(!f)throw std::runtime_error("cannot write scheme template: "+path);
    f<<schemeTemplate().dump(2)<<'\n';if(!f)throw std::runtime_error("failed writing scheme template: "+path);
}
void Expanded::print(std::ostream& out)const{
    out<<"Equation: "<<name_<<"\nOriginal: "<<original_.str()<<"\nExpansion rules:\n";
    for(const auto& t:trace_)out<<"  "<<t<<'\n';
    out<<"Expanded scalar DAG (no derivatives of intermediate fields):\n";
    std::map<const detail::Node*,std::string> names;std::size_t id=0;
    for(auto e:detail::nodes(result_)){
        const auto& n=*e.node();std::vector<std::string>a;for(auto c:n.args)a.push_back(names.at(c.node().get()));
        auto text=format(n,a);
        if(n.args.empty())names[e.node().get()]=text;
        else{auto name="v"+std::to_string(id++);names[e.node().get()]=name;out<<"  "<<name<<" = "<<text<<'\n';}
    }
    out<<"  result = "<<names.at(result_.node().get())<<"\nRequired operators:\n";requiredOperators().print(out);
}
Discrete Expanded::discretize(const Schemes& schemes)const{
    const ScalarField* anchor=state_;
    if(!anchor)for(auto e:detail::nodes(result_))if(e.node()->field){anchor=e.node()->field;break;}
    if(!anchor)throw std::invalid_argument("constant/coordinate expression needs an explicit layout field");
    return discretize(schemes,*anchor);
}
Discrete Expanded::discretize(const Schemes& schemes,const ScalarField& layout)const{
    return Discrete(detail::bind(result_,name_,schemes,layout,state_));
}
OperationInfo Expanded::inspect(const Schemes& schemes,const ScalarField& layout)const{
    return detail::bind(result_,name_,schemes,layout,state_,false)->info;
}
Term Discrete::lower(Fusion mode)const{return detail::lower(plan_,mode);}
Discrete::operator Evolution()const{
    if(!plan_->state)throw std::invalid_argument("only a symbolic ddt equation can be added as an evolution");
    return {plan_->state,[self=*this](Fusion m){return self.lower(m);}};
}
std::string Discrete::cudaSource(Fusion mode)const{return detail::source(*plan_,mode);}
std::size_t Discrete::kernelLaunches(Fusion mode)const{return 1+(mode==Fusion::Off?plan_->primitives.size():0);}
void Discrete::report(std::ostream& out)const{
    out<<plan_->name<<": expanded direct-derivative plan; "<<plan_->primitives.size()<<" unique stencils, "
       <<detail::nodes(plan_->expression).size()<<" scalar DAG nodes; CUDA kernels Auto="<<kernelLaunches(Fusion::Auto)
       <<", Off="<<kernelLaunches(Fusion::Off)<<'\n';
    for(const auto& p:plan_->primitives)out<<"  "<<p.selection.describe()<<"; direct cell stencil, halo="<<p.halo<<", corners="<<p.corners<<'\n';
    for(auto f:plan_->fields)out<<"  reads "<<f->name<<" at current time level\n";
}
} // namespace PhiX::numerics::symbolic
