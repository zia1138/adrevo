#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace selection {

// Decide whether to select the k-mer beginning at target within context.
// The harness supplies at most w-1 bases on either side of the target k-mer,
// clipped at sequence boundaries. Thus context.size() <= k + 2*(w-1).
// k and w are benchmark-owned: 1 <= k <= 31, w >= 1.
// Return exactly 0 or 1. A target containing anything except A/C/G/T (case
// insensitive) must return 0. Decisions must depend ONLY on these arguments:
// no mutable global/static state, I/O, clocks, randomness, or process inspection.
// target is a LOCAL offset, not an absolute sequence coordinate.
// No window-coverage guarantee or strand symmetry is imposed by this interface.
[[nodiscard]] std::uint32_t select_kmer(std::string_view context,
                                      std::size_t target,
                                      std::size_t k,
                                      std::size_t w);

}  // namespace selection
