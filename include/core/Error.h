#pragma once

// ---------------------------------------------------------------------------
// Error.h — the PhiX exception hierarchy + lightweight warning feedback.
//
// Every error thrown by the framework carries four structured pieces:
//
//   category  "Config" | "Device" | "Validation" | "Numerics" | "IO"
//   where     "Module::function" (or file:line for CUDA errors)
//   message   what went wrong, with the offending values spelled out
//   hint      how to fix it (optional)
//
// what() is pre-formatted from them:
//
//   PhixError[Validation] Equation::setRHS: term requires ghost >= 2 but
//   field 'phi' has ghost = 1
//     hint: construct the field with ScalarField(mesh, name, /*ghost=*/2)
//
// PhixError derives from std::runtime_error, so existing
// catch (std::exception&) / catch (std::runtime_error&) sites keep working.
// Catch a subclass (ConfigError, DeviceError, ValidationError,
// NumericsError, IOError) to handle one failure class, or PhixError to
// handle any framework error; runGuarded() (core/RunGuard.h) is the
// ready-made catch-all for main().
//
// warn()/warnOnce() are the non-fatal channel: conditions that are probably
// a mistake but have legitimate uses print a [PhiX] WARNING to stderr
// instead of throwing.  warnOnce(key, msg) deduplicates per key per
// process — safe to call from per-step code paths.
//
// This header is host-only (<stdexcept>/<string>/<iostream>) and can be
// included from any TU, host or nvcc.  The CUDA-checking macros live in
// core/CudaCheck.h (nvcc-compiled TUs only).
// ---------------------------------------------------------------------------

#include <stdexcept>
#include <string>
#include <iostream>
#include <mutex>
#include <unordered_set>

namespace PhiX {

class PhixError : public std::runtime_error {
public:
    PhixError(std::string category, std::string where,
              std::string message, std::string hint = "")
        : std::runtime_error(format(category, where, message, hint)),
          category_(std::move(category)),
          where_(std::move(where)),
          message_(std::move(message)),
          hint_(std::move(hint)) {}

    const std::string& category() const noexcept { return category_; }
    const std::string& where()    const noexcept { return where_; }
    const std::string& message()  const noexcept { return message_; }
    const std::string& hint()     const noexcept { return hint_; }

private:
    static std::string format(const std::string& category,
                              const std::string& where,
                              const std::string& message,
                              const std::string& hint) {
        std::string s = "PhixError[" + category + "] " + where + ": " + message;
        if (!hint.empty())
            s += "\n  hint: " + hint;
        return s;
    }

    std::string category_, where_, message_, hint_;
};

// Bad or missing configuration input (settings.jsonc keys, config values).
struct ConfigError : PhixError {
    ConfigError(const std::string& where, const std::string& message,
                const std::string& hint = "")
        : PhixError("Config", where, message, hint) {}
};

// CUDA runtime / cuFFT / device-side failures.
struct DeviceError : PhixError {
    DeviceError(const std::string& where, const std::string& message,
                const std::string& hint = "")
        : PhixError("Device", where, message, hint) {}
};

// Invalid parameters or an invalid call sequence (API misuse).
struct ValidationError : PhixError {
    ValidationError(const std::string& where, const std::string& message,
                    const std::string& hint = "")
        : PhixError("Validation", where, message, hint) {}
};

// The solution went bad: NaN/Inf fields, divergence, blow-up.
struct NumericsError : PhixError {
    NumericsError(const std::string& where, const std::string& message,
                  const std::string& hint = "")
        : PhixError("Numerics", where, message, hint) {}
};

// File / stream input-output failures.
struct IOError : PhixError {
    IOError(const std::string& where, const std::string& message,
            const std::string& hint = "")
        : PhixError("IO", where, message, hint) {}
};

// ---------------------------------------------------------------------------
// Non-fatal feedback
// ---------------------------------------------------------------------------

inline void warn(const std::string& msg) {
    std::cerr << "[PhiX] WARNING: " << msg << std::endl;
}

// Emit `msg` at most once per `key` per process.  Thread-safe.
inline void warnOnce(const std::string& key, const std::string& msg) {
    static std::mutex mtx;
    static std::unordered_set<std::string> seen;
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!seen.insert(key).second)
            return;
    }
    warn(msg);
}

} // namespace PhiX
