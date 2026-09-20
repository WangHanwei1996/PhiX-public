// ---------------------------------------------------------------------------
// module_diagnostics — PFHubWriter CSV roundtrip + interfacePosition
// ---------------------------------------------------------------------------
#include "IO/PFHubWriter.h"
#include "diagnostics/Interface.h"
#include "field/ScalarField.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    // ---- PFHubWriter: header + rows roundtrip -------------------------------
    {
        const std::string path = "pfhub_test.csv";
        {
            IO::PFHubWriter w(path, {"time", "free_energy"});
            w.addRow({0.0, 3.25});
            w.addRow({1.5, 2.125});
            require(w.rows() == 2, "row count wrong");
            bool threw = false;
            try { w.addRow({1.0}); } catch (const std::invalid_argument&) { threw = true; }
            require(threw, "column mismatch did not throw");
        }
        std::ifstream in(path);
        std::string l1, l2, l3;
        std::getline(in, l1); std::getline(in, l2); std::getline(in, l3);
        require(l1 == "time,free_energy", "CSV header wrong: " + l1);
        require(l2.rfind("0,3.25", 0) == 0, "row 1 wrong: " + l2);
        require(l3.rfind("1.5,2.125", 0) == 0, "row 2 wrong: " + l3);

        IO::PFHubWriter::writeMeta("pfhub_meta_test.yaml", "1a", "smoke");
        std::ifstream m("pfhub_meta_test.yaml");
        std::stringstream ss; ss << m.rdbuf();
        require(ss.str().find("id: 1a") != std::string::npos,
                "meta.yaml missing benchmark id");
        std::remove(path.c_str());
        std::remove("pfhub_meta_test.yaml");
    }

    // ---- interfacePosition: tanh front with off-grid centre -----------------
    {
        const int N = 128;
        const double dx = 1.0 / N;
        const double x0 = 0.61234;                 // deliberately off-grid
        Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                        N, dx, 0.0, 32, dx, 0.0);
        ScalarField phi(mesh, "phi", 1);
        phi.initialize([=](double x, double, double) {
            return 0.5 * (1.0 - std::tanh((x - x0) / 0.03));
        });
        phi.allocDevice();
        phi.uploadAllToDevice();

        const double xm = interfacePosition(phi, 0, 16, 0, 0.5, true);
        std::printf("  tanh front: found %.6f (true %.6f, dev %.2e)\n",
                    xm, x0, std::fabs(xm - x0));
        require(std::fabs(xm - x0) < 0.05 * dx,
                "front position off by " + std::to_string(xm - x0));

        // circle radius along +x through the centre row
        ScalarField c(mesh, "c", 1);
        const double R = 0.317, yc = 32 * dx / 2.0;
        c.initialize([=](double x, double y, double) {
            const double r = std::sqrt((x - 0.5) * (x - 0.5)
                                       + (y - yc) * (y - yc));
            return 0.5 * (1.0 - std::tanh((r - R) / 0.02));
        });
        c.allocDevice();
        c.uploadAllToDevice();
        const double xr = interfacePosition(c, 0, 16, 0, 0.5, true);
        require(std::fabs((xr - 0.5) - R) < 0.05 * dx,
                "circle radius off: " + std::to_string(xr - 0.5));

        // no crossing must throw
        ScalarField flat(mesh, "flat", 1);
        flat.fill(1.0);
        flat.allocDevice();
        flat.uploadAllToDevice();
        bool threw = false;
        try { interfacePosition(flat, 0, 5, 0, 0.5); }
        catch (const std::runtime_error&) { threw = true; }
        require(threw, "flat field did not throw");
    }

    return 0;
}
