#pragma once

// ---------------------------------------------------------------------------
// material/Material.h  —  umbrella include for the PhiX materials module
//
// Include this single header to pull in all material-related types:
//
//   #include "material/Material.h"
//
//   using namespace PhiX::Material;
//   auto alloy = BinaryAlloy::fromFile("data/FeB.fetab", "Fe-B");
//   double f   = alloy.freeEnergy(0.05, 1400.0);
// ---------------------------------------------------------------------------

#include "material/IMaterial.h"
#include "material/FreeEnergyTable.h"
#include "material/BinaryAlloy.h"
#include "material/KKS.h"
#include "material/KKSAntiTrapping.h"
