#include "core/Version.h"

#include "phix_version.h" // generated into build/generated/ at build time

namespace PhiX {

const char* version() { return PHIX_VERSION_STRING; }

const char* gitSha() { return PHIX_GIT_SHA; }

const char* versionString() {
    return "PhiX " PHIX_VERSION_STRING " (git " PHIX_GIT_SHA ")";
}

} // namespace PhiX
