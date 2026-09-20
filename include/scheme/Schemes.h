#pragma once
// ---------------------------------------------------------------------------
// scheme/Schemes.h — case-level numerical-scheme selection
//
// The OpenFOAM fvSchemes analogue: which discretisation each operator uses
// is a property of the CASE, not of the solver.  A solver asks the Schemes
// object instead of naming a scheme literally:
//
//     Schemes sch = Schemes::load(cfg);                 // settings/schemes.jsonc
//     eq.setRHS( lap(psi, sch, W0_sq) + gradSq(psi, sch) );
//     EquationSystem sys(dt, sch.ddt());
//     sch.warnUnused();                                 // after setup
//
// settings/schemes.jsonc — sibling of the settings file, JSON + // comments:
//
//   {
//     "ddt"       : { "default": "EULER" },                      // EULER | RK4
//     "laplacian" : { "default": "CD2", "lap(psi)": "Iso9" },    // CD2 CD4 CD6 Iso9 Iso27
//     "gradient"  : { "default": "CD2" },                        // CD2 CD4 CD6 Iso9
//     "gradSq"    : { "default": "CD2", "gradSq(psi)": "Iso9" }, // CD2 Iso9
//     "advection" : { "default": "UW1" },                        // UW1 UW2 WENO5
//     "anisoDiv"  : { "default": "CD2" }                         // CD2 Iso9
//   }
//
// Resolution order for lap(psi):  "lap(psi)" entry → the section's
// "default" → the solver's compiled-in default (2nd argument of load) →
// built-in (CD2 / UW1 / EULER).  A missing file is NOT an error: the solver
// defaults apply and one notice line goes to stdout, so cases written before
// this file existed run unchanged.
//
// Validation happens at load: unknown sections, non-string entries and
// scheme names an operator does not implement are ConfigErrors.  Per-term
// keys that no lookup ever touched (typically a misspelled field name) are
// reported by warnUnused().
//
// Scope: Term-layer lap / grad / gradSq / adv, AnisoParams::scheme and the
// time integrator. Fused flap/fgrad_dot overloads taking Schemes check their
// fixed CD2 choice against the case. Legacy overloads and hand-written
// kernels remain outside case binding. Selection is not a full setup audit.
// ---------------------------------------------------------------------------

#include "scheme/SchemeCatalog.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <nlohmann/json.hpp>

namespace PhiX {

namespace IO { class ConfigFile; }
enum class TimeScheme;    // solver/Solver.h   (opaque here; complete where used)
enum class AnisoScheme;   // operators/Anisotropy.h

class Schemes {
public:
    /// section → scheme name, e.g. {{"laplacian","Iso9"},{"gradSq","Iso9"}}:
    /// what the solver used before schemes.jsonc existed, applied when the
    /// file (or the section's "default") is absent.
    using Defaults = std::map<std::string, std::string>;

    /// <directory of cfg.path()>/schemes.jsonc; missing → builtin(solverDefaults).
    static Schemes load(const IO::ConfigFile& cfg, const Defaults& solverDefaults = {});
    /// Explicit path; the file must exist (IOError otherwise).
    static Schemes fromFile(const std::string& path, const Defaults& solverDefaults = {});
    /// No file at all.
    static Schemes builtin(const Defaults& solverDefaults = {});
    static Schemes fromJson(nlohmann::json data, const Defaults &solverDefaults = {});
    bool strict() const;

    // Per-operator lookups.  `field` is the ScalarField::name the operator
    // acts on; "" asks for the section default only.
    std::string lap     (const std::string& field = "") const;  // "laplacian", "lap(<field>)"
    std::string grad    (const std::string& field = "") const;  // "gradient",  "grad(<field>)"
    std::string gradSq  (const std::string& field = "") const;  // "gradSq",    "gradSq(<field>)"
    std::string adv     (const std::string& field = "") const;  // "advection", "adv(<field>)"
    AnisoScheme anisoDiv(const std::string& field = "") const;  // "anisoDiv",  "anisoDiv(<field>)"
    TimeScheme  ddt() const;                                    // "ddt"

    enum class Source { Entry, SectionDefault, SolverDefault, Builtin };
    struct Selection {
        std::string section;
        std::string key;        // requested operator key (may be empty)
        std::string name;       // selected scheme
        Source source = Source::Builtin;
        std::string matchedKey; // actual JSON key, or empty for compiled defaults
        std::string path;       // originating config path, if applicable
        std::string describe() const;
    };

    /// Resolution plus provenance, with the same precedence/unused tracking as resolve.
    /// Owns its strings: safe to retain after the Schemes object is destroyed.
    Selection select(const std::string& section, const std::string& key) const;
    Selection named(const std::string &section, const std::string &key) const;
    Selection gradient(const std::string &field, int axis) const;
    Selection advection(const std::string &velocity, const std::string &field) const;

    /// Bind an explicitly fixed implementation to the case selection. A mismatch
    /// throws ValidationError; this does NOT validate that implementation's BCs.
    Selection requireFixed(const std::string& section, const std::string& key,
                           const std::string& actual, const std::string& backend) const;

    /// Generic form: resolve("laplacian", "lap(psi)"); key "" = section default.
    std::string resolve(const std::string& section, const std::string& key) const;

    /// [PhiX] WARNING for every per-term entry no lookup ever touched.
    void warnUnused() const;

    bool               hasFile() const { return !path_.empty(); }
    const std::string& path()    const { return path_; }
    /// One line of section defaults: "ddt=EULER laplacian=CD2 ... anisoDiv=CD2".
    std::string summary() const;

private:
    nlohmann::json data_ = nlohmann::json::object();   // {section: {key: name}}
    std::string    path_;                              // "" when no file
    Defaults       defaults_;                          // solver-supplied section defaults
    mutable std::shared_ptr<std::set<std::string>> used_ =
        std::make_shared<std::set<std::string>>(); // "section/key" entries looked up

    void validate() const;
};

} // namespace PhiX
