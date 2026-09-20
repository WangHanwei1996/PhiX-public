#pragma once

// ---------------------------------------------------------------------------
// PFHubWriter.h — CSV time-series output in the PFHub upload convention.
//
// PFHub benchmark submissions are CSV files with a header row (e.g.
// "time,free_energy") plus a metadata file.  This writer appends one row
// per call and flushes immediately, so a crashed/killed run keeps every
// row written so far.
//
//     IO::PFHubWriter fe("output/free_energy.csv", {"time", "free_energy"});
//     ... per output step:  fe.addRow({sys.time, F});
//
//     IO::PFHubWriter::writeMeta("output/meta.yaml", "1a",
//                                "PhiX explicit CH, 256^2, adaptive dt");
// ---------------------------------------------------------------------------

#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

namespace PhiX {
namespace IO {

class PFHubWriter {
public:
    PFHubWriter(const std::string& path,
                const std::vector<std::string>& columns);

    // One CSV row; the value count must match the header column count.
    // Flushes to disk before returning.
    void addRow(std::initializer_list<double> values);

    int rows() const { return rows_; }

    // Minimal PFHub meta.yaml (benchmark id like "1a", "3", ... plus a
    // one-line summary; fill in the remaining upload fields on the site).
    static void writeMeta(const std::string& path,
                          const std::string& benchmarkId,
                          const std::string& summary);

private:
    std::ofstream ofs_;
    std::size_t   nCols_;
    int           rows_ = 0;
};

} // namespace IO
} // namespace PhiX
