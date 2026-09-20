#include "numerics/Preparation.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>

namespace PhiX::scheme {
const std::vector<Preset>& presets(){
    static const std::vector<Preset> entries={
        {"AxialCD2","Second-order axial baseline; no model approximation",
         {{"ddt","EULER"},{"laplacian","CD2"},{"gradient","CD2"},{"secondDerivative","CD2"},
          {"gradSq","CD2"},{"divGrad","CD2"},{"divNormal","CD2"}}},
        {"Ji2022S21","2D square-grid leading operators with full normalized flux; physical anisotropy corrections remain analytic/CD2",
         {{"ddt","EULER"},{"laplacian","Iso9"},{"gradient","CD2"},{"secondDerivative","CD2"},
          {"gradSq","Iso9"},{"divGrad","Iso9"},{"divNormal","Iso21"}}}};
    return entries;
}
const Preset& preset(const std::string& name){
    for(const auto& p:presets())if(p.name==name)return p;
    throw std::invalid_argument("unknown scheme preset: "+name);
}
}
namespace PhiX::numerics::symbolic {
Preparation prepareInventory(const OperatorInventory& inventory,const PreparationOptions& options){
    Preparation out;
    auto config=options.overrides;
    if(config.is_null())config=nlohmann::json::object();
    if(!config.is_object())throw std::invalid_argument("preparation overrides must be a scheme object");
    config["schema"]=2;config["policy"]="compatible";
    auto choices=Schemes::fromJson(config,options.preset.defaults);
    out.schemes={{"schema",2},{"policy","strict"}};
    out.report={{"schema",1},{"preset",options.preset.name},{"description",options.preset.description},
                {"presetModified",false},{"completeSymbolicInventory",true},{"ready",true},
                {"operators",nlohmann::json::array()},{"expressions",nlohmann::json::array()},
                {"issues",nlohmann::json::array()},
                {"scope","Host inspection only. No evolution or CUDA compilation. Interior stencil checks do not prove PDE stability."}};
    std::set<std::string> used;
    for(const auto& r:inventory.entries){
        auto selected=r.section=="gradient" ?
            choices.gradient(r.field,r.direction=="x"?0:1) : choices.select(r.section,r.key);
        if(!selected.matchedKey.empty())used.insert(r.section+"/"+selected.matchedKey);
        if(std::find(r.supportedSchemes.begin(),r.supportedSchemes.end(),selected.name)==r.supportedSchemes.end())
            throw std::invalid_argument(r.key+": scheme unavailable for this expression");
        out.schemes[r.section]["default"]="none";
        out.schemes[r.section][r.key]=selected.name;
        auto base=options.preset.defaults.find(r.section);
        if(base==options.preset.defaults.end() || base->second!=selected.name)out.report["presetModified"]=true;
        const auto* descriptor=scheme::find(scheme::findFamily(r.section)->id,selected.name);
        out.report["operators"].push_back({{"section",r.section},{"key",r.key},{"selected",selected.name},
            {"supported",r.supportedSchemes},{"sources",r.sources},{"location",r.location},
            {"interiorOrder",descriptor->interiorOrder},{"primaryHalo",descriptor->primaryHalo},
            {"squareRequired",descriptor->grid2D==scheme::Grid2D::SquareRequired}});
    }
    out.report["unusedOverrides"]=nlohmann::json::array();
    for(auto section=config.begin();section!=config.end();++section)if(section.value().is_object())
        for(auto entry=section.value().begin();entry!=section.value().end();++entry)
            if(entry.key()!="default" && !used.count(section.key()+"/"+entry.key())){
                auto key=section.key()+"/"+entry.key();
                out.report["unusedOverrides"].push_back(key);
                out.report["issues"].push_back("Unused preparation override: "+key);
                out.report["ready"]=false;
            }
    return out;
}
void inspectPreparation(Preparation& out,const Expanded& expression,const ScalarField& layout){
    std::ostringstream equations;expression.print(equations);out.equations+=equations.str()+"\n";
    nlohmann::json entry={{"output",layout.name},{"reads",nlohmann::json::array()}};
    try{
        auto info=expression.inspect(Schemes::fromJson(out.schemes),layout);
        entry["selections"]=info.schemes;
        for(const auto& r:info.reads)if(r.cell){
            entry["reads"].push_back({{"field",r.cell->name},{"requiredHalo",r.halo},
                                     {"allocatedHalo",r.cell->ghost},{"corners",r.corners}});
            if(r.halo>r.cell->ghost){
                out.report["ready"]=false;
                out.report["issues"].push_back(r.cell->name+": needs halo "+std::to_string(r.halo)+
                                               ", allocated "+std::to_string(r.cell->ghost));
            }
        }
    }catch(const std::exception& e){out.report["ready"]=false;out.report["issues"].push_back(e.what());}
    out.report["expressions"].push_back(entry);
}
Preparation prepare(const Expanded& expression,const ScalarField& layout,const PreparationOptions& options){
    auto result=prepareInventory(expression.requiredOperators(),options);
    inspectPreparation(result,expression,layout);return result;
}
void Preparation::write(const std::string& directory)const{
    namespace fs=std::filesystem;
    fs::path root(directory);
    if(root.empty())throw std::invalid_argument("preparation output directory is empty");
    const std::vector<std::pair<std::string,std::string>> files={
        {"schemes.jsonc",schemes.dump(2)+"\n"},{"preparation.json",report.dump(2)+"\n"},{"equations.txt",equations}};
    for(const auto& f:files)if(fs::exists(root/f.first))
        throw std::runtime_error("preparation refuses to overwrite "+(root/f.first).string());
    fs::create_directories(root);
    for(const auto& f:files){
        std::ofstream stream(root/f.first);
        stream<<f.second;
        if(!stream)throw std::runtime_error("cannot write preparation file "+(root/f.first).string());
    }
}
} // namespace PhiX::numerics::symbolic
