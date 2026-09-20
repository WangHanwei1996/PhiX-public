#pragma once

// ---------------------------------------------------------------------------
// RunGuard.h — exception guard for application entry points.
//
// Without a guard, any exception thrown by the library unwinds out of
// main() into std::terminate: the user sees a core dump instead of the
// structured message the throw site prepared.  Wrap the app body once:
//
//   static int phixMain(int argc, char* argv[]) {
//       ... existing main body, unchanged ...
//       return 0;
//   }
//
//   int main(int argc, char* argv[]) {
//       return PhiX::runGuarded(phixMain, argc, argv);
//   }
//
// On entry the build stamp (suite version + git SHA, core/Version.h) is
// printed to stdout, so a run log redirected with `> run.log` records the
// exact build that produced it — provenance for cases stored outside the
// repo.  On success the body's return value is passed through.  On any
// exception a readable "[PhiX] FATAL ..." block is printed to stderr and 1
// is returned, so shell scripts and ctest see a nonzero exit code.
// ---------------------------------------------------------------------------

#include "core/Error.h"
#include "core/Version.h"

#include <exception>
#include <iostream>
#include <utility>

namespace PhiX {

template <class Fn, class... Args>
int runGuarded(Fn&& fn, Args&&... args) {
    std::cout << versionString() << std::endl;
    try {
        return static_cast<int>(fn(std::forward<Args>(args)...));
    } catch (const PhixError& e) {
        std::cerr << "\n[PhiX] FATAL " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n[PhiX] FATAL unhandled std::exception: " << e.what()
                  << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n[PhiX] FATAL unknown exception" << std::endl;
        return 1;
    }
}

} // namespace PhiX
