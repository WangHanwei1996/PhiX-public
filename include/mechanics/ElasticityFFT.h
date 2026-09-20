#pragma once

// ---------------------------------------------------------------------------
// ElasticityFFT.h — spectral mechanical equilibrium for a HOMOGENEOUS
// cubic solid with a dilatational eigenstrain field (Khachaturyan).
//
// Solves  ∇·σ = 0,  σ = C : (ε − ε*),  ε*_ij(x) = eStar(x)·δ_ij
// on a fully periodic 2D cell (plane strain), matrix-free in Fourier space:
// per mode ξ the acoustic tensor K_ik = C_ijkl ξ_j ξ_l is inverted (2×2)
// and the strain amplitudes follow in closed form — ONE forward FFT of
// eStar and three inverse FFTs (ε11, ε22, ε12).  Spectral accuracy for
// smooth fields; the PFHub BM4 elastic-precipitate configuration
// (misfitting inclusion, homogeneous modulus variant) maps directly onto
// this: eStar = ε_misfit · h(φ).
//
// Mean-mode convention: zeroMeanStress = true (default) applies the
// stress-free homogeneous strain ⟨ε⟩ = ⟨ε*⟩ (free-standing periodic cell);
// false clamps ⟨ε⟩ = 0 (rigidly constrained cell).
//
// Outputs land in the PHYSICAL cells of the given fields (any pointer may
// be null).  elasticEnergyDensity e_el = ½(ε−ε*):C:(ε−ε*) is also
// available; integrate with reduce::fieldSum(e_el)·dV.
//
// Limits (documented): homogeneous modulus (inhomogeneous needs
// Moulinec–Suquet iteration on top), 2D plane strain, dilatational ε*.
// ---------------------------------------------------------------------------

#include "core/Real.h"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"

#include <cufft.h>

namespace PhiX {

struct ElasticParams2D {
    double C11 = 250.0;   // cubic stiffness (plane strain), any unit system
    double C12 = 150.0;
    double C44 = 100.0;

    void validate() const;   // positive definiteness checks
};

class ElasticityFFT2D {
public:
    ElasticityFFT2D(const Mesh& mesh, const ElasticParams2D& C);
    ~ElasticityFFT2D();

    ElasticityFFT2D(const ElasticityFFT2D&)            = delete;
    ElasticityFFT2D& operator=(const ElasticityFFT2D&) = delete;

    bool zeroMeanStress = true;   // ⟨σ⟩ = 0 (else ⟨ε⟩ = 0)

    // eStar: dilatational eigenstrain field (device-allocated, ghosts
    // irrelevant).  Outputs: total strains and/or elastic energy density.
    void solve(const ScalarField& eStar,
               ScalarField* e11, ScalarField* e22, ScalarField* e12,
               ScalarField* elasticEnergy = nullptr);

private:
    const Mesh&     mesh_;
    ElasticParams2D C_;
    int nx_, ny_, nc_;            // nc_ = (nx/2+1)·ny complex modes

    cufftHandle planF_ = 0, planB_ = 0;
    Real*  d_real_ = nullptr;     // packed nx·ny work array
    void*  d_hat_  = nullptr;     // eStar spectrum
    void*  d_s11_  = nullptr;     // strain spectra
    void*  d_s22_  = nullptr;
    void*  d_s12_  = nullptr;
    Real*  d_out_[3] = {nullptr, nullptr, nullptr};   // strain real arrays
};

} // namespace PhiX
