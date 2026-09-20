#include "IO/OutputWriter.h"
#include "core/Error.h"
#include "IO/FieldIO.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace {
// Format simulation time in scientific notation via a local stream, so the
// global std::cout format state is never touched (sticky flags like
// std::fixed would otherwise leak into every later double print).
std::string fmtTime(double t) {
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(3) << t;
    return ss.str();
}
} // anonymous namespace

namespace PhiX {
namespace IO {

OutputWriter::OutputWriter(const nlohmann::json& output_config)
    : printInterval(output_config.at("print_interval").get<int>())
    , writeInterval(output_config.at("write_interval").get<int>())
{
    // step % interval with interval 0 is SIGFPE; negative is meaningless.
    if (printInterval <= 0 || writeInterval <= 0)
        throw ConfigError("OutputWriter",
            "print_interval/write_interval must be >= 1 (got "
            + std::to_string(printInterval) + " / "
            + std::to_string(writeInterval) + ")",
            "to effectively disable an output channel, set its interval "
            "larger than n_steps");

    // "format" is one token, or several joined with '+' (e.g. "BINARY+VTI"):
    // the restart format alongside the visualization format.  Single tokens —
    // including "ALL" = BINARY+DAT+VTS — keep their historical meaning.
    const std::string fmt = output_config.at("format").get<std::string>();
    for (std::size_t pos = 0; pos <= fmt.size(); ) {
        const std::size_t plus = fmt.find('+', pos);
        const std::size_t end  = (plus == std::string::npos) ? fmt.size() : plus;
        std::string tok = fmt.substr(pos, end - pos);
        // tolerate spaces around the '+'
        const auto b = tok.find_first_not_of(" \t");
        const auto e = tok.find_last_not_of(" \t");
        tok = (b == std::string::npos) ? std::string() : tok.substr(b, e - b + 1);

        if      (tok == "BINARY")  writeBinary_ = true;
        else if (tok == "DAT")     writeDat_    = true;
        else if (tok == "VTK" || tok == "VTS") writeVtk_ = true;
        else if (tok == "VTK_BIN") writeVtkBin_ = true;
        else if (tok == "VTI")     writeVti_    = true;
        else if (tok == "ALL")   { writeBinary_ = writeDat_ = writeVtk_ = true; }
        else
            throw std::invalid_argument(
                "OutputWriter: unknown output format token \"" + tok
                + "\" in \"" + fmt
                + "\" (expected BINARY, DAT, VTK/VTS, VTK_BIN, VTI or ALL, "
                  "optionally joined with '+', e.g. \"BINARY+VTI\")");

        if (plus == std::string::npos) break;
        pos = plus + 1;
    }

    // VTS is ASCII StructuredGrid with an explicit node-coordinate array: on a
    // uniform grid it is ~41x larger than VTI and much slower for ParaView to
    // load.  Legal, but almost never what you want — say so once.
    if (writeVtk_)
        warnOnce("output-format-vts",
                 "OutputWriter: format \"" + fmt + "\" writes ASCII .vts "
                 "(VTK StructuredGrid with explicit node coordinates) — on a "
                 "uniform grid that is ~41x larger than .vti and far slower to "
                 "open in ParaView.  Prefer \"VTI\", or \"BINARY+VTI\" to keep "
                 "restart files");

    // Optional VTK coordinate scale (e.g. 1e9 → nm); default 1.0 = physical metres.
    coordScale_ = output_config.value("coord_scale", 1.0);

    // VTI payload precision: float32 (visualization default) or float64.
    const std::string prec =
        output_config.value("vti_precision", std::string("float32"));
    if (prec == "float32")      vtiFormat_ = FieldFormat::VTI;
    else if (prec == "float64") vtiFormat_ = FieldFormat::VTI_F64;
    else
        throw ConfigError("OutputWriter",
            "unknown vti_precision \"" + prec + "\"",
            "expected \"float32\" (default) or \"float64\"");

    std::filesystem::create_directories("output");
    resetTimer();
}

void OutputWriter::writeFields(ScalarField& f, int step, double simTime) {
    f.downloadCurrFromDevice();
    writeHostFields(f, step, simTime);
}

void OutputWriter::writeHostFields(const ScalarField& f, int step, double simTime) {
    std::string base = "output/" + f.name + "_" + std::to_string(step);
    if (writeBinary_) writeField(f, base + ".field", FieldFormat::BINARY);
    if (writeDat_)    writeField(f, base + ".dat",   FieldFormat::DAT);
    if (writeVtk_)    writeField(f, base + ".vts",   FieldFormat::VTS, coordScale_);
    if (writeVtkBin_) writeField(f, base + ".vtk",   FieldFormat::VTK_BIN, coordScale_);
    if (writeVti_) {
        writeField(f, base + ".vti", vtiFormat_, coordScale_);
        updatePvd(f.name, f.name + "_" + std::to_string(step) + ".vti",
                  simTime);
    }
    std::cout << "  step " << step
              << "  t=" << fmtTime(simTime)
              << "  written: " << base << "\n" << std::flush;
}

void OutputWriter::printProgress(int step, double simTime) {
    double elapsed = std::chrono::duration<double>(Clock::now() - t_start_).count();
    std::ostringstream el;
    el << std::fixed << std::setprecision(1) << elapsed;
    std::cout << "  [progress] step=" << step
              << "  t=" << fmtTime(simTime)
              << "  elapsed=" << el.str() << "s\n" << std::flush;
}

void OutputWriter::resetTimer() {
    t_start_ = Clock::now();
}

// ---------------------------------------------------------------------------
// .pvd collection (VTI): output/<field>.pvd lists every written .vti with
// its PHYSICAL time as the timestep.  The whole file is rewritten on every
// write (it is tiny).  On the first write of a run an existing .pvd is
// reloaded so warm restarts extend the series instead of truncating it;
// entries at/after the restart time are dropped (they are being rewritten).
// ---------------------------------------------------------------------------
void OutputWriter::updatePvd(const std::string& fieldName,
                             const std::string& fileName, double simTime) {
    PvdSeries& s = pvd_[fieldName];
    const std::string pvdPath = "output/" + fieldName + ".pvd";

    if (!s.seeded) {
        s.seeded = true;
        std::ifstream in(pvdPath);
        std::string line;
        while (in && std::getline(in, line)) {
            const auto tPos = line.find("timestep=\"");
            const auto fPos = line.find("file=\"");
            if (tPos == std::string::npos || fPos == std::string::npos)
                continue;
            const auto tEnd = line.find('"', tPos + 10);
            const auto fEnd = line.find('"', fPos + 6);
            if (tEnd == std::string::npos || fEnd == std::string::npos)
                continue;
            const double t = std::stod(line.substr(tPos + 10,
                                                   tEnd - tPos - 10));
            if (t < simTime)
                s.entries.emplace_back(
                    t, line.substr(fPos + 6, fEnd - fPos - 6));
        }
    }

    s.entries.emplace_back(simTime, fileName);

    std::ofstream ofs(pvdPath);
    if (!ofs)
        throw IOError("OutputWriter",
                      "cannot open collection file: " + pvdPath);
    ofs << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"Collection\" version=\"0.1\""
           " byte_order=\"LittleEndian\">\n"
        << "  <Collection>\n";
    std::ostringstream body;
    body << std::scientific << std::setprecision(9);
    for (const auto& e : s.entries)
        body << "    <DataSet timestep=\"" << e.first
             << "\" group=\"\" part=\"0\" file=\"" << e.second << "\"/>\n";
    ofs << body.str()
        << "  </Collection>\n"
        << "</VTKFile>\n";
}

} // namespace IO
} // namespace PhiX
