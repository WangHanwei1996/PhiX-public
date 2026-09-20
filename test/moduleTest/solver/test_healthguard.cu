// ---------------------------------------------------------------------------
// module_healthguard — standalone health sentinel (solver/HealthGuard.h).
//
// The Solver/EquationSystem sentinel, detached for hand-driven loops.
// Checks:
//   1. Healthy fields + healthy predicates: check() passes on and off
//      cadence.
//   2. Cadence: a tripped predicate is IGNORED off-cadence (due() gating)
//      and throws exactly on the due step; every = 0 disables even force()
//      candidates going through check().
//   3. NaN in a watched field (device copy) → NumericsError from
//      checkFieldHealth.
//   4. maxAbsLimit blow-up gate honoured.
//   5. User predicate: false → NumericsError whose message carries the
//      guard's name; guards evaluated in registration order.
//   6. force(): fires regardless of cadence.
// ---------------------------------------------------------------------------
#include "solver/HealthGuard.h"
#include "field/ScalarField.h"
#include "mesh/Mesh.h"
#include "core/Error.h"

#include <cmath>
#include <cstdio>
#include <string>

using namespace PhiX;

static int failures = 0;
static void require(bool cond, const std::string& msg) {
    if (!cond) { ++failures; std::printf("  FAIL: %s\n", msg.c_str()); }
    else       {             std::printf("  ok  : %s\n", msg.c_str()); }
}

int main() {
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    32, 0.5, 0.0, 24, 0.5, 0.0);
    ScalarField f(mesh, "f", 1);
    f.fill(1.0);
    f.allocDevice(); f.uploadAllToDevice();

    // === 1. healthy everything passes ======================================
    {
        HealthGuard g;
        g.watch(f);
        g.addGuard("always fine", [] { return true; });
        bool ok = true;
        try { g.check(100, 1.0); g.check(101, 1.01); }
        catch (...) { ok = false; }
        require(ok, "healthy field + predicate: no throw");
    }

    // === 2. cadence gating ==================================================
    {
        HealthGuard g;
        bool bad = false;
        g.addGuard("switchable", [&] { return !bad; });
        bad = true;
        bool threwOff = false, threwOn = false;
        try { g.check(101, 1.0); } catch (const NumericsError&) { threwOff = true; }
        try { g.check(200, 2.0); } catch (const NumericsError&) { threwOn  = true; }
        require(!threwOff && threwOn, "cadence: trips only on due steps");

        g.cfg.every = 0;
        bool threwDisabled = false;
        try { g.check(200, 2.0); } catch (const NumericsError&) { threwDisabled = true; }
        require(!threwDisabled, "every = 0 disables check()");
    }

    // === 3. NaN in a watched field ==========================================
    {
        ScalarField fn(mesh, "fn", 1);
        fn.fill(1.0);
        fn.curr[fn.index(3, 4, 0)] = std::nan("");
        fn.allocDevice(); fn.uploadAllToDevice();

        HealthGuard g;
        g.watch(fn);
        bool threw = false;
        try { g.check(100, 1.0); }
        catch (const NumericsError& e) {
            threw = std::string(e.what()).find("fn") != std::string::npos;
        }
        require(threw, "NaN in watched field -> NumericsError naming it");
    }

    // === 4. maxAbsLimit ======================================================
    {
        ScalarField fb(mesh, "fb", 1);
        fb.fill(5.0e7);
        fb.allocDevice(); fb.uploadAllToDevice();

        HealthGuard g;
        g.cfg.maxAbsLimit = 1e6;
        g.watch(fb);
        bool threw = false;
        try { g.check(100, 1.0); } catch (const NumericsError&) { threw = true; }
        require(threw, "maxAbsLimit blow-up gate honoured");
    }

    // === 5. predicate name in the message ==================================
    {
        HealthGuard g;
        g.addGuard("first ok",       [] { return true;  });
        g.addGuard("front runaway",  [] { return false; });
        bool named = false;
        try { g.check(100, 1.0); }
        catch (const NumericsError& e) {
            named = std::string(e.what()).find("front runaway")
                    != std::string::npos;
        }
        require(named, "tripped guard's name carried in the error");
    }

    // === 6. force() ignores cadence =========================================
    {
        HealthGuard g;
        g.addGuard("dead", [] { return false; });
        bool threw = false;
        try { g.force(7, 0.07); } catch (const NumericsError&) { threw = true; }
        require(threw, "force() fires off-cadence");
    }

    if (failures) { std::printf("module_healthguard: %d FAILURES\n", failures); return 1; }
    std::printf("module_healthguard: all passed\n");
    return 0;
}
