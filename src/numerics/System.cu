#include "numerics/System.h"
#include "numerics/Symbolic.h"
#include "numerics/Preparation.h"
#include "boundary/Apply2D.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include "boundary/FixedBC.h"
#include "field/FaceField.h"
#include <array>
#include <cmath>
#include <set>

namespace PhiX::numerics {
namespace {
std::function<Term(Fusion)> symbolicLower(std::shared_ptr<const symbolic::Expanded> expanded,
                                        const Schemes &schemes, const ScalarField &layout) {
    return [expanded = std::move(expanded), schemes, &layout](Fusion mode) {
        return expanded->discretize(schemes, layout).lower(mode);
    };
}
const void *identity(const ReadAccess &r) {
    return r.cell ? static_cast<const void *>(r.cell) : r.face;
}
__global__ void euler(Real *state, const Real *rhs, int nx, int ny, int sx, int sy, int g,
                      Real dt) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nx * ny)
        return;
    int c = (k % nx + g) + sx * ((k / nx + g) + sy * g);
    state[c] += dt * rhs[c];
}
void checkDt(double dt) {
    if (!std::isfinite(dt) || dt <= 0)
        throw std::invalid_argument("System: dt must be finite and positive");
}
} // namespace
System::~System() = default;
void System::editable() const {
    if (compiled_)
        throw std::logic_error("System: build a new plan to change definitions/BCs");
}
void System::checkCompiled() const {
    if (!compiled_)
        throw std::logic_error("System: call compile before execution");
    for (const auto &valid : layouts_)
        if (!valid())
            throw std::logic_error("System: field layout changed; rebuild the plan");
}
void System::bc(ScalarField &f, std::vector<BoundaryCondition *> bcs) {
    editable();
    auto &h = halos_[&f];
    h.bcs = std::move(bcs);
    h.width = f.ghost;
}
void System::halo(ScalarField &f, std::function<void()> cpu, std::function<void()> gpu, int width) {
    editable();
    auto &h = halos_[&f];
    h.cpu = std::move(cpu);
    h.gpu = std::move(gpu);
    h.width = width;
}
void System::addDefinition(ScalarField &f, std::function<Term(Fusion)> lower) {
    editable();
    Node n;
    n.cell = &f;
    n.outputs = {&f};
    n.lower = std::move(lower);
    nodes_.push_back(std::move(n));
}
void System::add(Evolution e) {
    addDefinition(*e.state, std::move(e.lower));
    nodes_.back().evolution = true;
}
void System::define(ScalarField &field, const symbolic::Expr &expr) {
    editable();
    auto expanded = std::make_shared<symbolic::Expanded>(expr.expand(field.name, expansionMode_));
    addDefinition(field, symbolicLower(expanded, schemes_, field));
    nodes_.back().expanded = std::move(expanded);
}
void System::add(const symbolic::Equation &equation) {
    editable();
    if (!equation.state)
        throw std::invalid_argument("System: symbolic equation needs a state field");
    auto expanded = std::make_shared<symbolic::Expanded>(equation.expand("", expansionMode_));
    addDefinition(*equation.state, symbolicLower(expanded, schemes_, *equation.state));
    nodes_.back().expanded = std::move(expanded);
    nodes_.back().evolution = true;
}
symbolic::OperatorInventory System::requiredOperators() const {
    symbolic::OperatorInventory inventory;
    for (const auto &n : nodes_)
        if (n.expanded)
            inventory.merge(n.expanded->requiredOperators());
    return inventory;
}
void System::reportEquations(std::ostream &out) const {
    out << "Symbolic equations (available before scheme binding):\n";
    for (const auto &n : nodes_)
        if (n.expanded)
            n.expanded->print(out);
        else
            out << (n.cell ? n.cell->name : n.external.name)
                << ": explicit discrete/external operation; no symbolic inventory\n";
}
symbolic::Preparation System::prepare()const{return prepare(symbolic::PreparationOptions{});}
symbolic::Preparation System::prepare(const symbolic::PreparationOptions& options)const{
    auto result=symbolic::prepareInventory(requiredOperators(),options);
    std::map<const void*,std::size_t> writers;
    std::vector<std::vector<std::size_t>> dependencies(nodes_.size());
    auto issue=[&](const std::string& text){
        result.report["ready"]=false;result.report["issues"].push_back(text);
    };
    for(std::size_t i=0;i<nodes_.size();++i)
        for(auto output:nodes_[i].outputs)
            if(!writers.emplace(output,i).second)issue("System: multiple producers for one field");
    for(const auto& n:nodes_){
        if(!n.expanded){
            result.report["completeSymbolicInventory"]=false;
            result.report["ready"]=false;
            result.report["issues"].push_back((n.cell?n.cell->name:n.external.name)+
                ": explicit discrete/external operation; inspect its own scheme contract");
            continue;
        }
        symbolic::inspectPreparation(result,*n.expanded,*n.cell);
        try{
            auto info=n.expanded->inspect(Schemes::fromJson(result.schemes),*n.cell);
            validate(info,n.cell,true);
            for(const auto& r:info.reads){
                auto it=writers.find(identity(r));
                if(it!=writers.end() && !nodes_[it->second].evolution)
                    dependencies[&n-nodes_.data()].push_back(it->second);
            }
        }catch(const std::exception& e){issue(e.what());}
    }
    std::vector<int> marks(nodes_.size());
    std::function<void(std::size_t)> visit=[&](std::size_t i){
        if(marks[i]==1)throw std::invalid_argument("System: cyclic auxiliary definitions");
        if(marks[i]==2)return;
        marks[i]=1;for(auto j:dependencies[i])visit(j);marks[i]=2;
    };
    try{for(std::size_t i=0;i<nodes_.size();++i)visit(i);}
    catch(const std::exception& e){issue(e.what());}
    if(!before_.empty() || !after_.empty()){
        result.report["completeSymbolicInventory"]=false;
        issue("External step barriers are not executed or validated by preparation.");
    }
    result.report["executionValidationPending"]=true;
    return result;
}
void System::replace(ScalarField &field, const symbolic::Expr &expr) {
    checkCompiled();
    auto it = producers_.find(&field);
    if (it == producers_.end() || !nodes_[it->second].equation)
        throw std::invalid_argument("replace expects a cell expression");
    auto expanded = std::make_shared<symbolic::Expanded>(
        nodes_[it->second].evolution ? (symbolic::ddt(field) == expr).expand("", expansionMode_)
                                    : expr.expand(field.name, expansionMode_));
    replaceExpression(field, symbolicLower(expanded, schemes_, field), expanded);
}
void System::producer(ExternalStep s) {
    editable();
    Node n;
    n.external = std::move(s);
    for (auto *f : n.external.writes)
        n.outputs.push_back(f);
    for (auto *f : n.external.faceWrites)
        n.outputs.push_back(f);
    if (n.outputs.empty())
        throw std::invalid_argument("producer must declare an output");
    nodes_.push_back(std::move(n));
}
void System::beforeStep(ExternalStep s) {
    editable();
    before_.push_back(std::move(s));
}
void System::afterStep(ExternalStep s) {
    editable();
    after_.push_back(std::move(s));
}

void System::validate(const OperationInfo &info, const ScalarField *output, bool hostInspection) const {
    if (!info.complete)
        throw std::invalid_argument("System: opaque operation has incomplete reads; supply an "
                                    "explicit ExternalStep contract");
    for (const auto &r : info.reads) {
        if (r.cell) {
            const auto &f = *r.cell;
            if (f.mesh.dim != 2)
                throw std::invalid_argument("System: only 2D fields are supported");
            check::checkCartesian(f.mesh, "System");
            if (!hostInspection && options_.backend == Backend::CPU && f.curr.size() != f.storedSize)
                throw std::invalid_argument("System: input has no complete host storage");
            if (output)
                check::checkSameMesh(f, *output, "System expression layout");
            check::checkGhost(f, r.halo, "System halo");
            if (!hostInspection && options_.backend == Backend::CUDA)
                check::checkOnDevice(f, "System input");
            if (!r.halo)
                continue;
            auto it = halos_.find(&f);
            if (it == halos_.end() || it->second.width < r.halo)
                throw std::invalid_argument("System: missing halo contract for '" + f.name + "'");
            auto &h = it->second;
            if (!h.bcs.empty()) {
                std::set<std::pair<int, int>> sides;
                for (auto *bc : h.bcs) {
                    if (!bc)
                        throw std::invalid_argument("System: null BC");
                    if (&f.mesh.patch(bc->patch.name) != &bc->patch)
                        throw std::invalid_argument("System: BC belongs to another mesh");
                    const int axis = int(bc->axis()), side = int(bc->side());
                    if (axis > 1)
                        throw std::invalid_argument("System: 2D BC axis out of range");
                    const int tangent = 1 - axis;
                    if (bc->patch.region.lo[tangent] != 0 ||
                        bc->patch.region.hi[tangent] != f.mesh.n[tangent])
                        throw std::invalid_argument(
                            "System: partial patches need an explicit halo callback");
                    if (!sides.insert({axis, side}).second)
                        throw std::invalid_argument("System: duplicate BC side");
                    const bool periodic = dynamic_cast<const PeriodicBC *>(bc);
                    if (periodic && !sides.insert({axis, 1 - side}).second)
                        throw std::invalid_argument("System: overlapping periodic BC");
                    const bool builtin = periodic || dynamic_cast<const NoFluxBC *>(bc) ||
                                         dynamic_cast<const FixedBC *>(bc);
                    if (!builtin)
                        throw std::invalid_argument(
                            "System: custom BC needs an explicit halo callback");
                    const auto* noflux = dynamic_cast<const NoFluxBC*>(bc);
                    const bool reflected = noflux && noflux->closure() == NoFluxBC::Closure::Reflect;
                    if (r.halo > 1 && !periodic && !reflected)
                        throw std::invalid_argument(
                            "System: wide nonperiodic closure is not certified");
                    if ((periodic || reflected) && f.ghost > f.mesh.n[axis])
                        throw std::invalid_argument("System: halo exceeds domain");
                }
                if (sides.size() != 4)
                    throw std::invalid_argument("System: provide all four 2D BC sides for '" +
                                                f.name + "'");
            } else if (!(hostInspection ? bool(h.cpu) || bool(h.gpu) :
                          options_.backend == Backend::CPU ? bool(h.cpu) : bool(h.gpu)))
                throw std::invalid_argument("System: halo callback missing for selected backend");
        }
        if (r.face) {
            const auto &f = *r.face;
            if (f.mesh.dim != 2 || f.normalAxis > 1)
                throw std::invalid_argument("System: unsupported face layout");
            if (output &&
                (!check::sameMeshGeometry(f.mesh, output->mesh) || f.ghost != output->ghost))
                throw std::invalid_argument("System: face/cell layouts differ");
            if (!hostInspection && options_.backend == Backend::CUDA && !f.d_data)
                throw std::invalid_argument("System: face input not on device");
        }
    }
}
System &System::compile(ExecutionOptions options) {
    editable();
    options_ = options;
    producers_.clear();
    order_.clear();
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        auto &n = nodes_[i];
        for (auto out : n.outputs)
            if (!out || !producers_.emplace(out, i).second)
                throw std::invalid_argument("System: duplicate/null writer");
        if (n.lower) {
            if (n.cell->mesh.dim != 2)
                throw std::invalid_argument("System: only 2D is supported");
            if (options.backend == Backend::CUDA)
                check::checkOnDevice(*n.cell, "System output");
            n.term = n.lower(options.fusion);
            n.info = operationInfo(n.term);
            validate(n.info, n.cell);
            if (!(options.backend == Backend::CPU ? bool(n.term.cpu_kernel)
                                                  : bool(n.term.gpu_launcher)))
                throw std::invalid_argument("System: no implementation for selected backend");
            n.equation = std::make_unique<Equation>(*n.cell);
            n.equation->setRHS(n.term);
            if (n.evolution) {
                const auto sel = schemes_.select("ddt", "ddt(" + n.cell->name + ")");
                if (sel.name != "EULER")
                    throw std::invalid_argument(
                        sel.describe() + "; System currently supports ForwardEuler (EULER) only");
                const auto description = sel.describe();
                if (std::find(n.info.schemes.begin(), n.info.schemes.end(), description) ==
                    n.info.schemes.end())
                    n.info.schemes.push_back(description);
                n.rhs = std::make_unique<ScalarField>(n.cell->mesh, n.cell->name + "@rhs",
                                                      n.cell->ghost);
                if (options.backend == Backend::CUDA)
                    n.rhs->allocDevice();
            }
        } else {
            if (n.external.configure)
                n.external.configure(options.backend, options.fusion);
            n.info = n.external.reads;
            validate(n.info);
            if (n.external.describe)
                n.info.execution = n.external.describe();
            for (auto *f : n.external.writes) {
                if (!f || f->mesh.dim != 2)
                    throw std::invalid_argument("System producer: invalid cell output");
                if (options.backend == Backend::CUDA)
                    check::checkOnDevice(*f, "producer output");
            }
            for (auto *f : n.external.faceWrites) {
                if (!f || f->mesh.dim != 2 || f->normalAxis > 1)
                    throw std::invalid_argument("System producer: invalid face output");
                if (options.backend == Backend::CUDA && !f->d_data)
                    throw std::invalid_argument("System producer: face output not allocated");
            }
            if (!(options.backend == Backend::CPU ? bool(n.external.cpu) : bool(n.external.gpu)))
                throw std::invalid_argument("System producer: missing implementation");
        }
    }
    auto validateExternal = [&](const ExternalStep &s) {
        validate(s.reads);
        if (!(options.backend == Backend::CPU ? bool(s.cpu) : bool(s.gpu)))
            throw std::invalid_argument("System external step: missing implementation");
    };
    for (const auto &s : before_)
        validateExternal(s);
    for (const auto &s : after_)
        validateExternal(s);
    std::vector<int> mark(nodes_.size());
    std::function<void(std::size_t)> visit = [&](std::size_t i) {
        if (nodes_[i].evolution)
            return; // state reads refer to old values, not producer evaluation
        if (mark[i] == 1)
            throw std::invalid_argument("System: cycle in auxiliary definitions");
        if (mark[i] == 2)
            return;
        mark[i] = 1;
        for (const auto &r : nodes_[i].info.reads) {
            auto it = producers_.find(identity(r));
            if (it != producers_.end())
                visit(it->second);
        }
        mark[i] = 2;
        order_.push_back(i);
    };
    for (std::size_t i = 0; i < nodes_.size(); ++i)
        visit(i);
    for (auto &h : halos_)
        if (!h.second.bcs.empty() && options.backend == Backend::CUDA) {
            h.second.batch = std::make_unique<BCBatch>();
            h.second.batch->build(*h.first, h.second.bcs);
        }
    std::map<const Real *, const void *> storage;
    std::set<const void *> described;
    layouts_.clear();
    auto record = [&](const ScalarField *f) {
        if (!f || f->mesh.dim != 2)
            throw std::invalid_argument("System: null or non-2D cell field");
        check::checkCartesian(f->mesh, "System field");
        if (options.backend == Backend::CPU && f->curr.size() != f->storedSize)
            throw std::invalid_argument("System: incomplete host output storage");
        if (!described.insert(f).second)
            return;
        const Real *data = options.backend == Backend::CUDA ? f->d_curr : f->curr.data();
        if (!data || !storage.emplace(data, f).second)
            throw std::invalid_argument("System: distinct field handles alias storage");
        const auto mesh = f->mesh;
        const int ghost = f->ghost;
        const auto size = f->storedSize;
        const std::array<int, 3> dims{f->storedDims[0], f->storedDims[1], f->storedDims[2]};
        layouts_.push_back([f, mesh, ghost, size, dims] {
            return check::sameMeshGeometry(f->mesh, mesh) && f->ghost == ghost &&
                   f->storedSize == size && std::equal(dims.begin(), dims.end(), f->storedDims);
        });
    };
    auto recordFace = [&](const FaceField *f) {
        if (!f || f->mesh.dim != 2 || f->normalAxis < 0 || f->normalAxis > 1)
            throw std::invalid_argument("System: null or unsupported face field");
        check::checkCartesian(f->mesh, "System face field");
        if (options.backend == Backend::CPU && f->data.size() != f->storedSize)
            throw std::invalid_argument("System: incomplete host face storage");
        if (!described.insert(f).second)
            return;
        const Real *data = options.backend == Backend::CUDA ? f->d_data : f->data.data();
        if (!data || !storage.emplace(data, f).second)
            throw std::invalid_argument("System: distinct face/cell handles alias storage");
        const auto mesh = f->mesh;
        const int ghost = f->ghost, axis = f->normalAxis;
        const auto size = f->storedSize;
        const std::array<int, 3> dims{f->storedDims[0], f->storedDims[1], f->storedDims[2]};
        layouts_.push_back([f, mesh, ghost, axis, size, dims] {
            return check::sameMeshGeometry(f->mesh, mesh) && f->ghost == ghost &&
                   f->normalAxis == axis && f->storedSize == size &&
                   std::equal(dims.begin(), dims.end(), f->storedDims);
        });
    };
    auto recordInfo = [&](const OperationInfo &i) {
        for (const auto &r : i.reads) {
            if (r.cell)
                record(r.cell);
            if (r.face)
                recordFace(r.face);
        }
    };
    for (const auto &n : nodes_) {
        recordInfo(n.info);
        if (n.cell)
            record(n.cell);
        for (auto *f : n.external.writes)
            record(f);
        for (auto *f : n.external.faceWrites)
            recordFace(f);
    }
    for (const auto *effects : {&before_, &after_})
        for (const auto &e : *effects) {
            recordInfo(e.reads);
            for (auto *f : e.writes)
                record(f);
            for (auto *f : e.faceWrites)
                recordFace(f);
        }
    compiled_ = true;
    invalidate();
    return *this;
}
void System::replaceExpression(ScalarField &field, std::function<Term(Fusion)> lower,
                               std::shared_ptr<const symbolic::Expanded> expanded) {
    checkCompiled();
    auto it = producers_.find(&field);
    if (it == producers_.end() || !nodes_[it->second].equation)
        throw std::invalid_argument("replace expects a cell expression");
    auto &n = nodes_[it->second];
    Term t = lower(options_.fusion);
    auto info = operationInfo(t);
    validate(info, &field);
    auto signature = [](const OperationInfo &i) {
        std::map<const void *, std::pair<int, bool>> s;
        for (const auto &r : i.reads)
            s[identity(r)] = {r.halo, r.corners};
        return s;
    };
    if (signature(info) != signature(n.info))
        throw std::invalid_argument("replace changes dependencies; build a new System");
    if (n.evolution) {
        const auto description = schemes_.select("ddt", "ddt(" + field.name + ")").describe();
        if (std::find(info.schemes.begin(), info.schemes.end(), description) == info.schemes.end())
            info.schemes.push_back(description);
    }
    n.equation->setRHS(t);
    n.term = std::move(t);
    n.info = std::move(info);
    n.lower = std::move(lower);
    n.expanded = std::move(expanded);
    n.valid = false;
    // Downstream caches must also be invalidated for changed captured parameters.
    invalidate();
}

void System::prepareReads(const OperationInfo &info) {
    for (const auto &r : info.reads) {
        auto it = producers_.find(identity(r));
        if (it != producers_.end() && !nodes_[it->second].evolution)
            ensure(it->second);
        if (!r.cell || !r.halo)
            continue;
        auto &h = halos_.at(r.cell);
        auto v = versions_[r.cell];
        if (h.valid && h.stamp == v)
            continue;
        auto &f = *const_cast<ScalarField *>(r.cell);
        if (options_.backend == Backend::CUDA) {
            if (h.batch)
                h.batch->applyOnGPU(f);
            else
                h.gpu();
        } else if (h.bcs.empty())
            h.cpu();
        else {
            applyBCsCPU2D(f, h.bcs);
        }
        h.stamp = v;
        h.valid = true;
    }
}
void System::ensure(std::size_t i) {
    auto &n = nodes_[i];
    if (n.evolution)
        return;
    prepareReads(n.info);
    std::vector<std::uint64_t> stamp;
    for (const auto &r : n.info.reads)
        stamp.push_back(versions_[identity(r)]);
    for (auto *resource : n.info.resources)
        stamp.push_back(resource->version);
    if (n.valid && stamp == n.stamp)
        return;
    if (n.equation) {
        if (options_.backend == Backend::CUDA) {
            n.equation->computeRHS(*n.cell);
            n.cell->hostCurrStale = true;
        } else
            n.equation->computeRHSCPU(*n.cell);
    } else
        runExternal(n.external, 0);
    for (auto out : n.outputs)
        versions_[out] = ++version_;
    n.stamp = std::move(stamp);
    n.valid = true;
}
void System::runExternal(const ExternalStep &s, double dt) {
    prepareReads(s.reads);
    if (options_.backend == Backend::CUDA)
        s.gpu(time_, dt);
    else
        s.cpu(time_, dt);
    for (auto *f : s.writes) {
        versions_[f] = ++version_;
        if (options_.backend == Backend::CUDA)
            f->hostCurrStale = true;
    }
    for (auto *f : s.faceWrites)
        versions_[f] = ++version_;
}
void System::evaluateRHS(std::size_t i) {
    auto &n = nodes_[i];
    prepareReads(n.info);
    if (options_.backend == Backend::CUDA)
        n.equation->computeRHS(*n.rhs);
    else
        n.equation->computeRHSCPU(*n.rhs);
}
void System::update(std::size_t i, double dt) {
    auto &n = nodes_[i];
    auto &f = *n.cell;
    if (options_.backend == Backend::CUDA) {
        euler<<<(f.mesh.n[0] * f.mesh.n[1] + 255) / 256, 256>>>(
            f.d_curr, n.rhs->d_curr, f.mesh.n[0], f.mesh.n[1], f.storedDims[0], f.storedDims[1],
            f.ghost, Real(dt));
        PHIX_KERNEL_CHECK("System ForwardEuler");
        f.advanceTimeLevelGPU();
        f.hostCurrStale = true;
    } else {
        for (int j = 0; j < f.mesh.n[1]; ++j)
            for (int k = 0; k < f.mesh.n[0]; ++k) {
                int c = f.index(k, j);
                f.curr[c] += Real(dt) * n.rhs->curr[c];
            }
        f.advanceTimeLevelCPU();
    }
    versions_[&f] = ++version_;
}
void System::start(double dt) {
    checkCompiled();
    for (const auto &s : before_)
        runExternal(s, dt);
}
void System::finish(double dt) {
    time_ += dt;
    ++step_;
    for (const auto &s : after_)
        runExternal(s, dt);
    if (health.due(static_cast<int>(step_)))
        for (const auto &n : nodes_)
            if (n.evolution) {
                if (options_.backend == Backend::CUDA)
                    check::checkFieldHealth(*n.cell, static_cast<int>(step_), time_,
                                            health.maxAbsLimit);
                else
                    check::checkFieldHealthCPU(*n.cell, static_cast<int>(step_), time_,
                                               health.maxAbsLimit);
            }
}
void System::advance(double dt) {
    checkDt(dt);
    start(dt);
    if (options_.coupling == Coupling::SameLevel) {
        for (std::size_t i = 0; i < nodes_.size(); ++i)
            if (nodes_[i].evolution)
                evaluateRHS(i);
        for (std::size_t i = 0; i < nodes_.size(); ++i)
            if (nodes_[i].evolution)
                update(i, dt);
    } else
        for (std::size_t i = 0; i < nodes_.size(); ++i)
            if (nodes_[i].evolution) {
                evaluateRHS(i);
                update(i, dt);
            }
    finish(dt);
}
void System::advanceAdaptive(
    const std::function<double(const std::vector<const ScalarField *> &)> &choose) {
    checkCompiled();
    if (options_.coupling != Coupling::SameLevel || !before_.empty())
        throw std::invalid_argument(
            "adaptive dt requires same-level coupling and no dt-dependent beforeStep");
    std::vector<const ScalarField *> rhs;
    for (std::size_t i = 0; i < nodes_.size(); ++i)
        if (nodes_[i].evolution) {
            evaluateRHS(i);
            rhs.push_back(nodes_[i].rhs.get());
        }
    if (options_.backend == Backend::CUDA)
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    double dt = choose(rhs);
    checkDt(dt);
    for (std::size_t i = 0; i < nodes_.size(); ++i)
        if (nodes_[i].evolution)
            update(i, dt);
    finish(dt);
}
void System::refresh(ScalarField &f) {
    checkCompiled();
    auto it = producers_.find(&f);
    if (it == producers_.end() || nodes_[it->second].evolution)
        throw std::invalid_argument("refresh expects a defined auxiliary");
    ensure(it->second);
    if (options_.backend == Backend::CUDA)
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
}
void System::refresh(FaceField &f) {
    checkCompiled();
    auto it = producers_.find(&f);
    if (it == producers_.end())
        throw std::invalid_argument("refresh expects a defined face field");
    ensure(it->second);
    if (options_.backend == Backend::CUDA)
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
}
void System::invalidate() {
    for (auto &n : nodes_)
        n.valid = false;
    for (auto &h : halos_)
        h.second.valid = false;
    for (auto &v : versions_)
        v.second = ++version_;
}
void System::touch(ScalarField &f) {
    checkCompiled();
    versions_[&f] = ++version_;
    halos_[&f].valid = false;
}
void System::restoreClock(double t, std::uint64_t s) {
    if (!std::isfinite(t) || t < 0)
        throw std::invalid_argument("System: invalid restart time");
    time_ = t;
    step_ = s;
    invalidate();
}
void System::report(std::ostream &out) const {
    checkCompiled();
    out << "2D " << (options_.backend == Backend::CUDA ? "CUDA" : "CPU")
        << " coupling=" << (options_.coupling == Coupling::SameLevel ? "same-level" : "sequential")
        << "; one macrostep clock\n";
    for (const auto &n : nodes_) {
        out << (n.cell ? n.cell->name : n.external.name) << ": "
            << (n.evolution ? "ForwardEuler old -> next; " : "auxiliary, stored; ")
            << n.info.execution << "\n";
        for (const auto &r : n.info.reads)
            out << "  read " << (r.cell ? r.cell->name : r.face->name) << " halo=" << r.halo
                << (r.corners ? " corners" : "") << "\n";
        for (auto *r : n.info.resources)
            out << "  resource " << r->name << " version=" << r->version << "\n";
        for (const auto &s : n.info.schemes)
            out << "  " << s << "\n";
        if (!n.evolution)
            out << "  refreshed on dependency version change; cross-definition fusion "
                   "unavailable\n";
    }
    auto reportEffects = [&](const char *phase, const std::vector<ExternalStep> &effects) {
        for (const auto &e : effects) {
            out << phase << " " << e.name << ": external barrier\n";
            for (const auto &r : e.reads.reads)
                out << "  read " << (r.cell ? r.cell->name : r.face->name) << " halo=" << r.halo
                    << (r.corners ? " corners" : "") << "\n";
            for (auto *r : e.reads.resources)
                out << "  resource " << r->name << " version=" << r->version << "\n";
            for (auto *f : e.writes)
                out << "  write " << f->name << "\n";
            for (auto *f : e.faceWrites)
                out << "  write face " << f->name << "\n";
        }
    };
    reportEffects("beforeStep", before_);
    reportEffects("afterStep", after_);
    out << "persistent RHS storage=" << persistentBytes()
        << " bytes; scratch owned/reused by each expression\n";
}
std::size_t System::persistentBytes() const {
    std::size_t bytes = 0;
    for (const auto &n : nodes_)
        if (n.rhs)
            bytes += n.rhs->storedBytes();
    return bytes;
}
} // namespace PhiX::numerics
