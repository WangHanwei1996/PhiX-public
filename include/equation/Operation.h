#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace PhiX {
class ScalarField;
class FaceField;

// Host description. Object identity survives storage rotation; raw pointers do not.
struct Resource {
    std::string name;
    std::uint64_t version = 0;
    void touch() { ++version; }
};
struct ReadAccess {
    const ScalarField *cell = nullptr;
    const FaceField *face = nullptr;
    int halo = 0;
    bool corners = false;
};
struct OperationInfo {
    // Unknown launchers must opt in with an explicit read contract.
    bool complete = false;
    std::vector<ReadAccess> reads;
    std::vector<const Resource *> resources;
    std::vector<std::string> schemes;
    std::string execution = "materialized";

    void read(const ScalarField *f, int halo = 0, bool corners = false) {
        if (!f)
            return;
        for (auto &r : reads)
            if (r.cell == f) {
                r.halo = std::max(r.halo, halo);
                r.corners |= corners;
                return;
            }
        reads.push_back({f, nullptr, halo, corners});
    }
    void read(const FaceField *f) {
        if (!f)
            return;
        for (auto &r : reads)
            if (r.face == f)
                return;
        reads.push_back({nullptr, f, 0, false});
    }
    void read(const Resource *r) {
        if (r && std::find(resources.begin(), resources.end(), r) == resources.end())
            resources.push_back(r);
    }
    void merge(const OperationInfo &other) {
        complete = complete && other.complete;
        for (const auto &r : other.reads) {
            if (r.cell)
                read(r.cell, r.halo, r.corners);
            if (r.face)
                read(r.face);
        }
        for (auto *r : other.resources)
            read(r);
        for (const auto &s : other.schemes)
            if (std::find(schemes.begin(), schemes.end(), s) == schemes.end())
                schemes.push_back(s);
    }
};
} // namespace PhiX
