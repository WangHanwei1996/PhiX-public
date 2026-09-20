#include "IO/ConfigFile.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace PhiX {
namespace IO {

// ---------------------------------------------------------------------------
// stripLineComment
//
// Walk the line character by character, tracking whether we are inside a
// double-quoted string (handles \" escape).  A '//' found outside a string
// marks the start of a comment — everything from that point is dropped.
// ---------------------------------------------------------------------------
std::string ConfigFile::stripLineComment(const std::string& line)
{
    bool in_string = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        char c = line[i];

        if (in_string) {
            if (c == '\\') {
                ++i;            // skip escaped character (e.g. \")
            } else if (c == '"') {
                in_string = false;
            }
        } else {
            if (c == '"') {
                in_string = true;
            } else if (c == '/' && i + 1 < line.size() && line[i + 1] == '/') {
                return line.substr(0, i);   // drop comment tail
            }
        }
    }
    return line;
}

// ---------------------------------------------------------------------------
// fromArgs
// ---------------------------------------------------------------------------
ConfigFile ConfigFile::fromArgs(int argc, char* argv[], const std::string& defaultPath)
{
    const std::string path = (argc >= 2) ? argv[1] : defaultPath;
    try {
        return ConfigFile(path);
    } catch (const PhixError&) {
        throw;   // already structured
    } catch (const std::exception& e) {
        throw ConfigError("ConfigFile::fromArgs", e.what(),
            std::string("usage: ") + (argc >= 1 ? argv[0] : "<app>")
            + " [path/to/settings.jsonc]  (default: " + defaultPath + ")");
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ConfigFile::ConfigFile(const std::string& path)
    : path_(path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw IOError("ConfigFile", "cannot open \"" + path + "\"",
            "run from the directory containing settings/ or pass the config "
            "path as argv[1]");
    }

    std::ostringstream stripped;
    std::string line;
    while (std::getline(file, line)) {
        stripped << stripLineComment(line) << '\n';
    }

    try {
        data_ = nlohmann::json::parse(stripped.str());
    } catch (const nlohmann::json::parse_error& e) {
        throw ConfigError("ConfigFile",
            "JSON parse error in \"" + path + "\": " + e.what(),
            "the reported byte offset counts the comment-stripped text; "
            "line numbers still match the original file");
    }
}

// ---------------------------------------------------------------------------
// operator[]
// ---------------------------------------------------------------------------
ConfigView ConfigFile::operator[](const std::string& key) const
{
    // Root view + checked descend — a missing key throws ConfigError with
    // the full path and the keys that do exist at that level.
    return ConfigView(&data_, "")[key];
}

// ---------------------------------------------------------------------------
// has
// ---------------------------------------------------------------------------
bool ConfigFile::has(const std::string& key) const
{
    return data_.contains(key);
}

} // namespace IO
} // namespace PhiX
