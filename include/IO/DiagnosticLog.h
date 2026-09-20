#pragma once

// ---------------------------------------------------------------------------
// DiagnosticLog.h — scalar time-series CSV logger.
//
// OutputWriter handles FIELD files; this handles the other output every
// solver ends up hand-rolling: a CSV of scalar diagnostics (front position,
// solid fraction, extrema, mass...) sampled on a step cadence.
//
//     IO::DiagnosticLog log("output/run_log.csv",
//                           {"step", "time_s", "front_W", "solid_frac"});
//     ...
//     if (log.due(step))
//         log.row({double(step), t, front, fs});
//
// Guarantees:
//   • the HEADER is flushed at construction — the file is never 0 bytes
//     while the run is alive;
//   • every row is flushed — tail -f works, and a crash loses nothing;
//   • row() validates the value count against the declared columns and
//     throws on mismatch.
//
// `every` gates due(step) (default 1 = every step; 0 disables).  Pass
// append = true to continue an existing file on a warm restart (the header
// is only written if the file is missing or empty).  Host-only, no CUDA.
// ---------------------------------------------------------------------------

#include "core/Error.h"

#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <string>
#include <vector>

namespace PhiX {
namespace IO {

class DiagnosticLog {
public:
    DiagnosticLog(const std::string& path,
                  std::vector<std::string> columns,
                  int every = 1,
                  bool append = false)
        : columns_(std::move(columns)), every_(every), path_(path)
    {
        if (columns_.empty())
            throw std::invalid_argument(
                "DiagnosticLog: at least one column required");

        bool writeHeader = true;
        if (append) {
            std::ifstream probe(path);
            writeHeader = !probe || probe.peek() == std::ifstream::traits_type::eof();
        }
        ofs_.open(path, append ? (std::ios::out | std::ios::app)
                               : std::ios::out);
        if (!ofs_)
            throw IOError("DiagnosticLog", "cannot open: " + path,
                          "does the parent directory exist?");
        if (writeHeader) {
            for (std::size_t i = 0; i < columns_.size(); ++i)
                ofs_ << (i ? "," : "") << columns_[i];
            ofs_ << "\n" << std::flush;   // flushed NOW — never a 0-byte file
        }
        ofs_ << std::setprecision(12);
    }

    bool due(int step) const { return every_ > 0 && step % every_ == 0; }

    void row(std::initializer_list<double> values) {
        if (values.size() != columns_.size())
            throw std::invalid_argument(
                "DiagnosticLog(" + path_ + "): row has "
                + std::to_string(values.size()) + " values, header declares "
                + std::to_string(columns_.size()) + " columns");
        bool first = true;
        for (double v : values) {
            ofs_ << (first ? "" : ",") << v;
            first = false;
        }
        ofs_ << "\n" << std::flush;
    }

    const std::vector<std::string>& columns() const { return columns_; }

private:
    std::vector<std::string> columns_;
    int every_;
    std::string path_;
    std::ofstream ofs_;
};

} // namespace IO
} // namespace PhiX
