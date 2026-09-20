#include "IO/PFHubWriter.h"

#include <iomanip>
#include <stdexcept>

namespace PhiX {
namespace IO {

PFHubWriter::PFHubWriter(const std::string& path,
                         const std::vector<std::string>& columns)
    : ofs_(path), nCols_(columns.size())
{
    if (!ofs_)
        throw std::runtime_error("PFHubWriter: cannot open " + path);
    if (columns.empty())
        throw std::invalid_argument("PFHubWriter: no columns given");
    for (std::size_t i = 0; i < columns.size(); ++i)
        ofs_ << columns[i] << (i + 1 < columns.size() ? "," : "\n");
    ofs_.flush();
}

void PFHubWriter::addRow(std::initializer_list<double> values) {
    if (values.size() != nCols_)
        throw std::invalid_argument(
            "PFHubWriter::addRow: expected " + std::to_string(nCols_)
            + " values, got " + std::to_string(values.size()));
    std::size_t i = 0;
    ofs_ << std::setprecision(12);
    for (double v : values)
        ofs_ << v << (++i < nCols_ ? "," : "\n");
    ofs_.flush();
    ++rows_;
}

void PFHubWriter::writeMeta(const std::string& path,
                            const std::string& benchmarkId,
                            const std::string& summary) {
    std::ofstream m(path);
    if (!m)
        throw std::runtime_error("PFHubWriter::writeMeta: cannot open " + path);
    m << "---\n"
      << "benchmark:\n"
      << "  id: " << benchmarkId << "\n"
      << "  version: '1'\n"
      << "metadata:\n"
      << "  summary: " << summary << "\n"
      << "  implementation:\n"
      << "    name: PhiX\n"
      << "    repo:\n"
      << "      url: https://github.com/WangHanwei1996/PhiX\n"
      << "  hardware:\n"
      << "    acc_architecture: gpu\n"
      << "data:\n"
      << "  - name: free_energy\n"
      << "    values: free_energy.csv\n";
    m.flush();
}

} // namespace IO
} // namespace PhiX
