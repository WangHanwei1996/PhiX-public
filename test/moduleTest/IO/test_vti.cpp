// ---------------------------------------------------------------------------
// module_io_vti — VTK XML ImageData writer (FieldFormat::VTI) + .pvd index.
//
// 1. Header parse: WholeExtent / Origin / Spacing carry the mesh geometry
//    (coordScale applied); CellData DataArray is Float32 appended raw.
// 2. Payload: UInt64 byte count == nx*ny*nz*4, then float32 values
//    bit-compared against float(cell value); ghosts poisoned so any halo
//    leak flips the payload.
// 3. VTI_F64: same header/geometry, type="Float64", payload bit-exact
//    against the solver's doubles, byte count nx*ny*8.
// 4. VectorField + VTI: Vectors CellData, NumberOfComponents="3" with N<3
//    zero-padded, interleaved payload bit-exact; VTK_BIN still throws.
// 5. OutputWriter "VTI" token: writes <name>_<step>.vti and maintains
//    <name>.pvd with PHYSICAL time as timestep; a second write appends.
// 6. Restart seeding: a fresh OutputWriter (new run) reloads the existing
//    .pvd and drops entries at/after the restart time.
// 7. Combined "+" format tokens, vti_precision, and the VTS warning.
// ---------------------------------------------------------------------------

#include "field/ScalarField.h"
#include "field/VectorField.h"
#include "IO/FieldIO.h"
#include "IO/OutputWriter.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PhiX;
namespace fs = std::filesystem;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static int countOccurrences(const std::string& hay, const std::string& needle) {
    int n = 0;
    for (std::size_t p = hay.find(needle); p != std::string::npos;
         p = hay.find(needle, p + needle.size()))
        ++n;
    return n;
}

int main() {
try {
    fs::create_directories("vti_test_out");

    const int NX = 7, NY = 5;
    const double DX = 0.25, DY = 0.5, OX = 1.0, OY = -2.0;
    Mesh mesh = Mesh::makeUniform2D(CoordSys::CARTESIAN,
                                    NX, DX, OX, NY, DY, OY);
    ScalarField f(mesh, "phi", 1);
    f.fillCurr(1e200);   // poison ghosts
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
            f.curr[f.index(i, j, 0)] = 0.1 * i - 0.7 * j + 0.013 * i * j;

    // === 1+2. header + payload ============================================
    const std::string path = "vti_test_out/phi_test.vti";
    const double scale = 2.0;
    IO::writeField(f, path, FieldFormat::VTI, scale);

    const std::string blob = slurp(path);
    require(blob.find("<VTKFile type=\"ImageData\"") != std::string::npos,
            "VTKFile type is ImageData");
    require(blob.find("WholeExtent=\"0 7 0 5 0 1\"") != std::string::npos,
            "WholeExtent covers the corner-node lattice");
    {
        // Origin/Spacing carry mesh geometry * coordScale
        std::ostringstream want;
        want << std::scientific << std::setprecision(12)
             << "Origin=\"" << OX * scale << " " << OY * scale << " "
             << 0.0 * scale << "\"";
        require(blob.find(want.str()) != std::string::npos,
                "Origin == mesh.origin * coordScale");
        std::ostringstream wsp;
        wsp << std::scientific << std::setprecision(12)
            << "Spacing=\"" << DX * scale << " " << DY * scale << " "
            << DX * scale << "\"";
        require(blob.find(wsp.str()) != std::string::npos,
                "Spacing == mesh.d * coordScale (2D slab dz = dx)");
    }
    require(blob.find("type=\"Float32\" Name=\"phi\""
                      " format=\"appended\" offset=\"0\"")
            != std::string::npos, "Float32 appended DataArray");

    {
        const std::size_t mark = blob.find("encoding=\"raw\">\n_");
        require(mark != std::string::npos, "raw appended section present");
        const char* p = blob.data() + mark + std::strlen("encoding=\"raw\">\n_");
        std::uint64_t nBytes = 0;
        std::memcpy(&nBytes, p, 8);
        require(nBytes == std::uint64_t(NX) * NY * 4,
                "payload byte count == nx*ny*4");
        const char* data = p + 8;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                float got;
                std::memcpy(&got, data + 4 * (j * NX + i), 4);
                const float want =
                    static_cast<float>(f.curr[f.index(i, j, 0)]);
                require(std::memcmp(&got, &want, 4) == 0,
                        "float32 payload bit-exact (no ghost leak)");
            }
    }

    // === 3. VTI_F64 — same layout, float64 payload =========================
    {
        const std::string p64 = "vti_test_out/phi_test_f64.vti";
        IO::writeField(f, p64, FieldFormat::VTI_F64, scale);
        const std::string b64 = slurp(p64);
        require(b64.find("type=\"Float64\" Name=\"phi\""
                         " format=\"appended\" offset=\"0\"")
                != std::string::npos, "VTI_F64 declares Float64");
        // geometry header is identical to the float32 file
        const std::size_t gEnd = blob.find("<DataArray");
        require(gEnd != std::string::npos && b64.compare(0, gEnd,
                                                         blob, 0, gEnd) == 0,
                "VTI_F64 geometry header identical to VTI");

        const std::size_t mark = b64.find("encoding=\"raw\">\n_");
        require(mark != std::string::npos, "raw appended section present");
        const char* p = b64.data() + mark + std::strlen("encoding=\"raw\">\n_");
        std::uint64_t nBytes = 0;
        std::memcpy(&nBytes, p, 8);
        require(nBytes == std::uint64_t(NX) * NY * 8,
                "payload byte count == nx*ny*8");
        const char* data = p + 8;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                double got;
                std::memcpy(&got, data + 8 * (j * NX + i), 8);
                const double want = f.curr[f.index(i, j, 0)];
                require(std::memcmp(&got, &want, 8) == 0,
                        "float64 payload bit-exact (no ghost leak)");
            }
    }

    // === 4. VectorField + VTI — 3 components, zero-padded, interleaved =====
    {
        VectorField vf(mesh, "v", 2, 1);       // N = 2 -> padded to 3
        for (int c = 0; c < 2; ++c) {
            vf[c].fillCurr(1e200);             // poison ghosts
            for (int j = 0; j < NY; ++j)
                for (int i = 0; i < NX; ++i)
                    vf[c].curr[vf[c].index(i, j, 0)] =
                        (c + 1) * (0.3 * i - 0.11 * j);
        }
        const std::string pv = "vti_test_out/v.vti";
        IO::writeField(vf, pv, FieldFormat::VTI, scale);
        const std::string bv = slurp(pv);
        require(bv.find("<CellData Vectors=\"v\">") != std::string::npos,
                "vector VTI uses Vectors CellData");
        require(bv.find("type=\"Float32\" Name=\"v\""
                        " NumberOfComponents=\"3\""
                        " format=\"appended\" offset=\"0\"")
                != std::string::npos,
                "vector VTI declares 3 components");

        const std::size_t mark = bv.find("encoding=\"raw\">\n_");
        require(mark != std::string::npos, "raw appended section present");
        const char* p = bv.data() + mark + std::strlen("encoding=\"raw\">\n_");
        std::uint64_t nBytes = 0;
        std::memcpy(&nBytes, p, 8);
        require(nBytes == std::uint64_t(NX) * NY * 3 * 4,
                "payload byte count == nx*ny*3*4");
        const char* data = p + 8;
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                for (int c = 0; c < 3; ++c) {
                    float got;
                    std::memcpy(&got, data + 4 * (3 * (j * NX + i) + c), 4);
                    const float want = (c < 2)
                        ? static_cast<float>(vf[c].curr[vf[c].index(i, j, 0)])
                        : 0.0f;
                    require(std::memcmp(&got, &want, 4) == 0,
                            "vector payload interleaved, bit-exact, "
                            "zero-padded");
                }

        // VTK_BIN remains scalar-only
        bool threw = false;
        try { IO::writeField(vf, "vti_test_out/v.vtk", FieldFormat::VTK_BIN); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "VectorField + VTK_BIN still throws");
    }

    // === 5. OutputWriter VTI token + .pvd ==================================
    f.allocDevice();
    f.uploadAllToDevice();   // writeFields downloads from device
    fs::remove_all("output");
    nlohmann::json cfg = {
        {"print_interval", 1}, {"write_interval", 1}, {"format", "VTI"}
    };
    {
        IO::OutputWriter w(cfg);
        w.writeFields(f, 100, 0.5);
        w.writeFields(f, 200, 1.0);
        require(fs::exists("output/phi_100.vti")
                && fs::exists("output/phi_200.vti"),
                "OutputWriter writes <name>_<step>.vti");
        const std::string pvd = slurp("output/phi.pvd");
        require(pvd.find("type=\"Collection\"") != std::string::npos,
                ".pvd is a VTK Collection");
        require(pvd.find("file=\"phi_100.vti\"") != std::string::npos
                && pvd.find("file=\"phi_200.vti\"") != std::string::npos,
                ".pvd indexes both writes");
        require(pvd.find("timestep=\"5.000000000e-01\"") != std::string::npos
                && pvd.find("timestep=\"1.000000000e+00\"")
                   != std::string::npos,
                ".pvd timesteps carry PHYSICAL time");
    }

    // === 6. warm restart seeding ===========================================
    {
        // new run restarting from t = 1.0: the t = 0.5 entry must survive,
        // the t = 1.0 entry is rewritten by the new run
        IO::OutputWriter w2(cfg);
        w2.writeFields(f, 200, 1.0);
        w2.writeFields(f, 300, 1.5);
        const std::string pvd = slurp("output/phi.pvd");
        require(pvd.find("file=\"phi_100.vti\"") != std::string::npos,
                "restart keeps pre-restart entries");
        require(countOccurrences(pvd, "file=\"phi_200.vti\"") == 1,
                "restart does not duplicate the rewritten step");
        require(pvd.find("file=\"phi_300.vti\"") != std::string::npos,
                "restart appends new entries");
    }

    // === 7. combined format tokens, vti_precision, VTS warning =============
    {
        fs::remove_all("output");
        nlohmann::json c2 = {{"print_interval", 1}, {"write_interval", 1},
                             {"format", "BINARY+VTI"}};
        IO::OutputWriter w(c2);
        w.writeFields(f, 10, 0.25);
        require(fs::exists("output/phi_10.field")
                && fs::exists("output/phi_10.vti")
                && fs::exists("output/phi.pvd"),
                "\"BINARY+VTI\" writes both channels plus the .pvd");
        require(!fs::exists("output/phi_10.vts"),
                "\"BINARY+VTI\" does not write VTS");
    }
    {
        fs::remove_all("output");
        nlohmann::json c3 = {{"print_interval", 1}, {"write_interval", 1},
                             {"format", "VTI"}, {"vti_precision", "float64"}};
        IO::OutputWriter w(c3);
        w.writeFields(f, 10, 0.25);
        const std::string b = slurp("output/phi_10.vti");
        require(b.find("type=\"Float64\"") != std::string::npos,
                "vti_precision float64 reaches the writer");
        const std::size_t mark = b.find("encoding=\"raw\">\n_");
        std::uint64_t nBytes = 0;
        std::memcpy(&nBytes,
                    b.data() + mark + std::strlen("encoding=\"raw\">\n_"), 8);
        require(nBytes == std::uint64_t(NX) * NY * 8, "float64 payload size");
    }
    {
        // unknown token in a '+' list is rejected, and names the bad token
        nlohmann::json bad = {{"print_interval", 1}, {"write_interval", 1},
                              {"format", "VTI+NOPE"}};
        bool threw = false;
        try { IO::OutputWriter w(bad); }
        catch (const std::invalid_argument& ex) {
            threw = std::string(ex.what()).find("NOPE") != std::string::npos;
        }
        require(threw, "unknown '+' token throws and names itself");

        nlohmann::json badPrec = {{"print_interval", 1}, {"write_interval", 1},
                                  {"format", "VTI"},
                                  {"vti_precision", "float16"}};
        bool threw2 = false;
        try { IO::OutputWriter w(badPrec); }
        catch (const std::exception&) { threw2 = true; }
        require(threw2, "unknown vti_precision throws");
    }
    {
        // selecting VTS emits one [PhiX] WARNING on stderr
        std::ostringstream cap;
        std::streambuf* old = std::cerr.rdbuf(cap.rdbuf());
        nlohmann::json cv = {{"print_interval", 1}, {"write_interval", 1},
                             {"format", "VTS"}};
        try { IO::OutputWriter w(cv); } catch (...) {}
        std::cerr.rdbuf(old);
        require(cap.str().find("[PhiX] WARNING") != std::string::npos
                && cap.str().find(".vti") != std::string::npos,
                "VTS selection warns and points at .vti");
    }

    fs::remove_all("vti_test_out");
    fs::remove_all("output");
    std::printf("module_io_vti: all passed\n");
    return 0;
} catch (const std::exception& ex) {
    std::printf("FAIL: %s\n", ex.what());
    return 1;
}
}
