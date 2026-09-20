#pragma once

// ---------------------------------------------------------------------------
// Version.h — build-time version stamp of the PhiX suite.
//
// Values come from build/generated/phix_version.h, regenerated on every
// build by cmake/PhixVersion.cmake: the suite version is the newest
// changelog/v<X.Y.Z> directory; the git SHA carries a "-dirty" suffix when
// tracked files had uncommitted changes at build time.  runGuarded() prints
// versionString() once at startup, so every run log records exactly what
// built the binary — cases stored outside the repo keep their provenance.
// ---------------------------------------------------------------------------

namespace PhiX {

const char* version();        // "v3.8.6"
const char* gitSha();         // "63ea00b" | "63ea00b-dirty" | "unknown"
const char* versionString();  // "PhiX v3.8.6 (git 63ea00b)"

} // namespace PhiX
