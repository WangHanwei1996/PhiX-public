#pragma once

#include "core/Error.h"

#include <nlohmann/json.hpp>

#include <string>
#include <type_traits>
#include <vector>

namespace PhiX {
namespace IO {

namespace detail {
// Types the implicit ConfigView conversion may produce: config scalars and
// flat arrays of them.  Anything else (json, OutputWriter, ...) must go
// through operator const json& / get<T>() — an unconstrained operator T()
// would make ConfigView convertible to ANYTHING and wreck overload
// resolution at ordinary call sites.
template<class T>
struct is_config_value
    : std::integral_constant<bool, std::is_arithmetic<T>::value
                                   || std::is_same<T, std::string>::value> {};
template<class U, class A>
struct is_config_value<std::vector<U, A>> : is_config_value<U> {};
} // namespace detail

// ---------------------------------------------------------------------------
// ConfigView — checked, path-tracking view into the parsed config tree.
//
// cfg["mesh"]["nx"] descends with a contains() check at EVERY level and
// throws ConfigError naming the full path plus the keys that DO exist in
// that section.  (nlohmann's const operator[] on a missing key is undefined
// behaviour — this proxy is what makes nested access safe.)
//
// A view converts implicitly to scalars / std::string (type errors become
// ConfigError) and to const nlohmann::json& (for APIs that take a whole
// section, e.g. OutputWriter(cfg["output"]), buildBCs(mesh, cfg["..."])).
// json-style helpers (contains/count/value/get/at/size/begin/end/dump)
// forward to the underlying node.
// ---------------------------------------------------------------------------

class ConfigView {
public:
    ConfigView(const nlohmann::json* node, std::string path)
        : node_(node), path_(std::move(path)) {}

    // Checked descend — the core of the proxy.
    ConfigView operator[](const std::string& key) const {
        if (!node_->is_object() || !node_->contains(key))
            throw ConfigError("cfg" + path_ + "[\"" + key + "\"]",
                node_->is_object() ? "key not found"
                                   : "parent is not a JSON object",
                availableKeysHint());
        return ConfigView(&node_->at(key), path_ + "[\"" + key + "\"]");
    }

    /// Literal-key overload: exact match for cfg["a"]["b"] — without it the
    /// templated operator T() (→ integral) makes the built-in subscript on
    /// the literal a competing candidate and the call ambiguous.
    ConfigView operator[](const char* key) const {
        return (*this)[std::string(key)];
    }

    /// at() mirrors operator[] (both are checked).
    ConfigView at(const std::string& key) const { return (*this)[key]; }

    // ------------------------------------------------------------------
    // json-like forwarding
    // ------------------------------------------------------------------
    bool contains(const std::string& key) const {
        return node_->is_object() && node_->contains(key);
    }
    std::size_t count(const std::string& key) const {
        return contains(key) ? 1 : 0;
    }
    std::size_t size()  const { return node_->size(); }
    bool        empty() const { return node_->empty(); }
    auto begin() const { return node_->begin(); }
    auto end()   const { return node_->end(); }
    std::string dump(int indent = -1) const { return node_->dump(indent); }

    /// Checked typed extraction; conversion failures become ConfigError.
    template<class T>
    T get() const {
        try {
            return node_->get<T>();
        } catch (const nlohmann::json::exception& e) {
            throw ConfigError("cfg" + path_,
                std::string("cannot convert value to the requested type: ")
                    + e.what(),
                "value is: " + node_->dump());
        }
    }

    /// value(key, default) — default if the key is absent, checked otherwise.
    template<class T>
    T value(const std::string& key, const T& def) const {
        return contains(key) ? (*this)[key].get<T>() : def;
    }
    std::string value(const std::string& key, const char* def) const {
        return value<std::string>(key, std::string(def));
    }

    // ------------------------------------------------------------------
    // Conversions
    // ------------------------------------------------------------------

    /// Pass a whole section to APIs taking const nlohmann::json&.
    operator const nlohmann::json&() const { return *node_; }

    /// Implicit typed read:  const double dx = cfg["mesh"]["dx"];
    /// Restricted to arithmetic / std::string / std::vector thereof.
    template<class T,
             class = std::enable_if_t<
                 detail::is_config_value<std::decay_t<T>>::value>>
    operator T() const { return get<T>(); }

    /// Explicit access to the underlying node.
    const nlohmann::json& json() const { return *node_; }

private:
    std::string availableKeysHint() const {
        if (!node_->is_object())
            return "value here is: " + node_->dump();
        std::string keys;
        for (auto it = node_->begin(); it != node_->end(); ++it)
            keys += (keys.empty() ? "" : ", ") + it.key();
        return keys.empty() ? "this section is empty"
                            : "available keys here: " + keys;
    }

    const nlohmann::json* node_;
    std::string           path_;
};

// ---------------------------------------------------------------------------
// ConfigFile
//
// Loads a JSONC file (JSON with // line comments) and exposes the parsed
// data through checked ConfigView access.
//
// Comment stripping correctly ignores '//' that appear inside string values.
//
// Usage:
//   ConfigFile cfg("settings/settings.jsonc");
//   int  nx = cfg["mesh"]["nx"];      // ConfigError on a missing key
//   bool ok = cfg.has("mesh");
//
// The underlying json is accessible via cfg.data() for advanced queries.
//
// PERFORMANCE NOTE: every cfg[...] access is a JSON tree lookup (string
// hash + conversion).  Never call it inside a time-stepping loop — hoist
// the values into local variables once at startup:
//   const int writeEvery = cfg["output"]["write_interval"];   // hoisted
// ---------------------------------------------------------------------------

class ConfigFile {
public:
    /// Load and parse a JSONC file.  Throws IOError/ConfigError on failure.
    explicit ConfigFile(const std::string& path);

    /// Construct from command-line arguments.
    /// argv[1] is used as the path if provided; otherwise defaultPath is used.
    /// Throws ConfigError on failure (usage line goes into the hint) —
    /// pair with PhiX::runGuarded in main() for a readable message.
    static ConfigFile fromArgs(int argc, char* argv[],
                               const std::string& defaultPath = "settings/settings.jsonc");

    /// Access a top-level key (read-only, checked at every nesting level).
    ConfigView operator[](const std::string& key) const;

    /// Check whether a top-level key exists.
    bool has(const std::string& key) const;

    /// Direct access to the underlying json object (for nested / advanced use).
    const nlohmann::json& data() const { return data_; }

    /// The file this config was loaded from (Schemes::load looks for
    /// schemes.jsonc next to it).
    const std::string& path() const { return path_; }

private:
    nlohmann::json data_;
    std::string    path_;

    /// Strip // line comments from a single line, respecting quoted strings.
    static std::string stripLineComment(const std::string& line);
};

} // namespace IO
} // namespace PhiX
