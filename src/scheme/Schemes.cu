#include "scheme/Schemes.h"
#include "IO/ConfigFile.h"
#include "core/Error.h"
#include "solver/Solver.h"           // TimeScheme
#include "operators/Anisotropy.h"    // AnisoScheme, anisoSchemeFromString

#include <filesystem>
#include <iostream>
#include <vector>

namespace PhiX {

namespace {

// Throws E (ConfigError for file entries, ValidationError for solver
// defaults) when `name` is not a scheme the section's operator implements.
template<class E>
void checkName(const std::string& where, const std::string& section,
               const std::string& name) {
    const auto* sp = scheme::findFamily(section);
    if (!sp)
        throw E(where, "unknown scheme section \"" + section + "\"",
                "sections: " + scheme::sectionNames());
    if (scheme::find(sp->id, name)) return;
    throw E(where, "\"" + section + "\": unknown scheme \"" + name + "\"",
            "supported: " + scheme::names(sp->id));
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
Schemes Schemes::builtin(const Defaults& solverDefaults) {
    Schemes s;
    s.defaults_ = solverDefaults;
    s.validate();
    return s;
}

Schemes Schemes::fromJson(nlohmann::json data, const Defaults &defaults) {
    Schemes s;
    s.data_ = std::move(data);
    s.defaults_ = defaults;
    s.validate();
    return s;
}
bool Schemes::strict() const {
    return data_.value("policy", std::string("compatible")) == "strict";
}

Schemes Schemes::fromFile(const std::string& path, const Defaults& solverDefaults) {
    IO::ConfigFile file(path);              // JSONC parse; IOError if missing
    Schemes s;
    s.data_     = file.data();
    s.path_     = path;
    s.defaults_ = solverDefaults;
    s.validate();
    std::cout << "[PhiX] schemes: " << path << "  (" << s.summary() << ")" << std::endl;
    return s;
}

Schemes Schemes::load(const IO::ConfigFile& cfg, const Defaults& solverDefaults) {
    namespace fs = std::filesystem;
    const fs::path p = fs::path(cfg.path()).parent_path() / "schemes.jsonc";
    if (fs::exists(p))
        return fromFile(p.string(), solverDefaults);
    Schemes s = builtin(solverDefaults);
    std::cout << "[PhiX] schemes: " << p.string()
              << " not found — using solver defaults (" << s.summary() << ")"
              << std::endl;
    return s;
}

void Schemes::validate() const {
    for (const auto& kv : defaults_)
        checkName<ValidationError>("Schemes::load(solverDefaults)", kv.first, kv.second);

    const std::string where = path_.empty() ? "Schemes" : "Schemes(" + path_ + ")";
    if (!data_.is_object())
        throw ConfigError(where, "top level must be an object of sections",
                          "e.g. { \"laplacian\": { \"default\": \"CD2\" } }");
    if (data_.contains("schema") && (!data_["schema"].is_number_integer() || data_["schema"] != 2))
        throw ConfigError(where, "schema must be 2", "omit schema for legacy files");
    if (data_.contains("policy") &&
        (!data_.contains("schema") || !data_["policy"].is_string() ||
         (data_["policy"] != "strict" && data_["policy"] != "compatible")))
        throw ConfigError(where, "policy requires schema 2 and strict or compatible",
                          "check reserved keys");
    for (auto it = data_.begin(); it != data_.end(); ++it) {
        const std::string& section = it.key();
        if (section == "schema" || section == "policy")
            continue;
        if (!scheme::findFamily(section))
            throw ConfigError(where, "unknown scheme section \"" + section + "\"",
                              "sections: " + scheme::sectionNames());
        if (!it.value().is_object())
            throw ConfigError(where, "\"" + section + "\" must be an object",
                              "e.g. \"" + section + "\": { \"default\": \""
                              + scheme::catalog(scheme::findFamily(section)->id).front().name + "\" }");
        for (auto jt = it.value().begin(); jt != it.value().end(); ++jt) {
            if (!jt.value().is_string())
                throw ConfigError(where, "\"" + section + "\".\"" + jt.key()
                                  + "\" must be a scheme name (string)",
                                  "supported: " + scheme::names(scheme::findFamily(section)->id));
            if (jt.value() == "none" && data_.contains("schema") && jt.key() == "default")
                continue;
            checkName<ConfigError>(where, section, jt.value().get<std::string>());
        }
    }
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------
Schemes::Selection Schemes::select(const std::string& section, const std::string& key) const {
    const auto* sp = scheme::findFamily(section);
    if (!sp)
        throw ValidationError("Schemes::select",
                              "unknown scheme section \"" + section + "\"",
                              "sections: " + scheme::sectionNames());
    if (data_.contains(section)) {
        const auto& sec = data_[section];
        if (!key.empty() && sec.contains(key)) {
            used_->insert(section + "/" + key);
            return {section, key, sec[key].get<std::string>(), Source::Entry, key, path_};
        }
        if (sec.contains("default") && sec["default"] == "none")
            throw ValidationError("Schemes::select",
                                  section + "/" + key + ": default none requires an explicit entry",
                                  "declare this operator key");
        if (sec.contains("default"))
            return {section, key, sec["default"].get<std::string>(),
                    Source::SectionDefault, "default", path_};
    }
    if (strict())
        throw ValidationError("Schemes::select",
                              section + "/" + key + ": no selection under strict policy",
                              "declare a matching key or section default");
    auto d = defaults_.find(section);
    if (d != defaults_.end())
        return {section, key, d->second, Source::SolverDefault, "", ""};
    return {section, key, scheme::catalog(sp->id).front().name, Source::Builtin, "", ""};
}

Schemes::Selection Schemes::named(const std::string &section, const std::string &key) const {
    if (!data_.contains(section) || !data_[section].contains(key))
        throw ValidationError("Schemes::named",
                              "missing explicit operator name " + section + "/" + key,
                              "named operators require an exact entry");
    return select(section, key);
}
Schemes::Selection Schemes::gradient(const std::string &field, int axis) const {
    if (axis < 0 || axis > 2)
        throw std::invalid_argument("gradient selection: invalid axis");
    const std::string full = "grad(" + field + "," + std::string(1, "xyz"[axis]) + ")",
                      generic = "grad(" + field + ")";
    if (data_.contains("gradient") && !data_["gradient"].contains(full) &&
        data_["gradient"].contains(generic)) {
        auto s = select("gradient", generic);
        s.key = full;
        return s;
    }
    return select("gradient", full);
}
Schemes::Selection Schemes::advection(const std::string &velocity, const std::string &field) const {
    const std::string key = "adv(" + velocity + "," + field + ")", old = "adv(" + field + ")";
    if (!strict() && data_.contains("advection") && !data_["advection"].contains(key) &&
        data_["advection"].contains(old)) {
        auto s = select("advection", old);
        s.key = key;
        return s;
    }
    return select("advection", key);
}

std::string Schemes::resolve(const std::string& section, const std::string& key) const {
    return select(section, key).name;
}

std::string Schemes::Selection::describe() const {
    std::string out = section + "/" + (key.empty() ? "default" : key) + "=" + name;
    switch (source) {
    case Source::Entry:
    case Source::SectionDefault:
        return out + " <- " + path + ":" + section + "/" + matchedKey;
    case Source::SolverDefault: return out + " <- solver default";
    case Source::Builtin:       return out + " <- built-in default";
    }
    return out;
}

Schemes::Selection Schemes::requireFixed(const std::string& section, const std::string& key,
                                         const std::string& actual, const std::string& backend) const {
    checkName<ValidationError>("Schemes::requireFixed", section, actual);
    auto selected = select(section, key);
    if (selected.name != actual)
        throw ValidationError(backend, selected.describe()
                              + "; implementation is fixed to " + actual,
                              "select " + actual + " for this operator, or use an implementation "
                              "that supports " + selected.name);
    return selected;
}

static std::string termKey(const char* op, const std::string& field) {
    return field.empty() ? std::string() : std::string(op) + "(" + field + ")";
}

std::string Schemes::lap   (const std::string& f) const { return resolve("laplacian", termKey("lap",    f)); }
std::string Schemes::grad  (const std::string& f) const { return resolve("gradient",  termKey("grad",   f)); }
std::string Schemes::gradSq(const std::string& f) const { return resolve("gradSq",    termKey("gradSq", f)); }
std::string Schemes::adv   (const std::string& f) const { return resolve("advection", termKey("adv",    f)); }

AnisoScheme Schemes::anisoDiv(const std::string& f) const {
    return anisoSchemeFromString(resolve("anisoDiv", termKey("anisoDiv", f)));
}

TimeScheme Schemes::ddt() const {
    return scheme::detail::lookup(scheme::detail::timeFactories(), resolve("ddt", ""),
                                  "Schemes::ddt", false).factory();
}

std::string Schemes::summary() const {
    std::string out;
    for (const auto& sp : scheme::families()) {
        if (!out.empty()) out += " ";
        const std::string section = sp.section;
        if (data_.contains(section) && data_[section].contains("default"))
            out += section + "=" + data_[section]["default"].get<std::string>();
        else if (strict())
            out += section + "=<explicit entries only>";
        else
            out += section + "=" + resolve(section, "");
    }
    return out;
}

void Schemes::warnUnused() const {
    const std::string src = path_.empty() ? "schemes" : "schemes (" + path_ + ")";
    for (auto it = data_.begin(); it != data_.end(); ++it) {
        if (it.key() == "schema" || it.key() == "policy")
            continue;
        for (auto jt = it.value().begin(); jt != it.value().end(); ++jt) {
            if (jt.key() == "default") continue;
            if (!used_->count(it.key() + "/" + jt.key()))
                warn(src + ": entry \"" + it.key() + "\".\"" + jt.key()
                     + "\" was never looked up — misspelled field name? "
                       "(keys are <op>(<ScalarField::name>), e.g. \"lap(phi)\")");
        }
    }
}

} // namespace PhiX
