#include "scheme/CentralDifference.h"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace PhiX;

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

int main() {
    double s[5] = {0.0, 1.0, 4.0, 9.0, 16.0};
    require(scheme::CD2::ghostRequired() == 1, "CD2 ghost mismatch");
    require(scheme::CD2::order() == 2, "CD2 order mismatch");
    require(std::abs(scheme::CD2::d1(s, 2, 1, 1.0) - 4.0) < 1e-12, "CD2 d1 mismatch");
    require(std::abs(scheme::CD2::d2(s, 2, 1, 1.0) - 2.0) < 1e-12, "CD2 d2 mismatch");
    return 0;
}
