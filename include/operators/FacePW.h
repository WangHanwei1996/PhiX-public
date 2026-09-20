#pragma once

// ---------------------------------------------------------------------------
// FacePW.h — Point-wise (element-wise) transforms on FaceField data.
//
// Symmetric companion to TermPW.inl (which operates on cell-centred fields).
// Enables non-linear flux assembly on faces without writing custom CUDA kernels.
//
// API
// ----
// All overloads write  out[idx] = fn(a[idx])             (1-field)
//                      out[idx] = fn(a[idx], b[idx])      (2-field)
//                      out[idx] = fn(a[idx], b[idx], c[idx])  (3-field)
//                      out[idx] = fn(a[idx], b[idx], c[idx], d[idx])  (4-field)
// for every element of the FaceField (both physical and ghost in tangential
// directions, because the face layout already omits ghost along normalAxis).
//
// CPU and GPU paths are both provided.  GPU paths require all FaceField
// arguments to have d_data allocated (allocDevice()) and up-to-date on GPU.
// The result is written to out.d_data (GPU) or out.data (CPU).
//
// Usage example (anisotropic x-flux):
//
//   FaceField phi_x_xf(mesh, 0, "phi_x_xf");   // faceGradGPU(phi, 0, ...)
//   FaceField phi_y_xf(mesh, 0, "phi_y_xf");   // interpGPU(phi_y_cc, 0, ...)
//   FaceField jx(mesh, 0, "jx");
//
//   facePWGPU(jx, phi_x_xf, phi_y_xf,
//       PHIX_FN (double px, double py) {
//           double theta = atan2(py, px);
//           double a = 1.0 + eps * cos(4.0 * theta);
//           return W0sq * a * (a * px + eps * 4.0 * sin(4.0*theta) * py);
//       });
//
// Convenience macro: PHIX_FN defined in equation/Term.h
// ---------------------------------------------------------------------------

#include "field/FaceField.h"
#include "equation/Term.h"   // for PHIX_FN macro

namespace PhiX {

// ===========================================================================
// CPU implementations (defined in FacePW.inl, instantiated by the caller)
// ===========================================================================

template<typename Fn>
void facePW(FaceField& out, const FaceField& a, Fn fn);

template<typename Fn>
void facePW(FaceField& out, const FaceField& a, const FaceField& b, Fn fn);

template<typename Fn>
void facePW(FaceField& out,
            const FaceField& a, const FaceField& b, const FaceField& c,
            Fn fn);

template<typename Fn>
void facePW(FaceField& out,
            const FaceField& a, const FaceField& b, const FaceField& c,
            const FaceField& d, Fn fn);

// ===========================================================================
// GPU implementations
// ===========================================================================

template<typename Fn>
void facePWGPU(FaceField& out, const FaceField& a, Fn fn);

template<typename Fn>
void facePWGPU(FaceField& out, const FaceField& a, const FaceField& b, Fn fn);

template<typename Fn>
void facePWGPU(FaceField& out,
               const FaceField& a, const FaceField& b, const FaceField& c,
               Fn fn);

template<typename Fn>
void facePWGPU(FaceField& out,
               const FaceField& a, const FaceField& b, const FaceField& c,
               const FaceField& d, Fn fn);

} // namespace PhiX

#include "operators/FacePW.inl"
