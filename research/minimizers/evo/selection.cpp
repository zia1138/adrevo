#include "../selection_api.hpp"

#include <algorithm>
#include <optional>
#include <vector>

namespace {

int encode_base(char base) {
    switch (base) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

std::uint64_t hash64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::optional<std::uint64_t> rank_kmer(std::string_view context,
                                       std::size_t begin, std::size_t k) {
    std::uint64_t forward = 0, reverse = 0;
    for (std::size_t offset = 0; offset < k; ++offset) {
        const int base = encode_base(context[begin + offset]);
        if (base < 0) return std::nullopt;
        forward = (forward << 2U) | static_cast<std::uint64_t>(base);
        reverse |= static_cast<std::uint64_t>(3 - base) << (2U * offset);
    }
    return hash64(std::min(forward, reverse));
}

}  // namespace

// Baseline: the target is selected iff it is the leftmost minimum canonical
// hash in at least one complete window of w k-mers containing it. The local
// context includes every such window, reproducing the original simulator.
std::uint32_t selection::select_kmer(std::string_view context,
                                   std::size_t target, std::size_t k,
                                   std::size_t w) {
    if (context.size() < k) return 0;
    const std::size_t count = context.size() - k + 1;
    if (count < w) return 0;
    std::vector<std::optional<std::uint64_t>> ranks;
    ranks.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        ranks.push_back(rank_kmer(context, i, k));
    }
    if (!ranks[target]) return 0;
    const std::size_t first = target >= w - 1 ? target - (w - 1) : 0;
    const std::size_t last = std::min(target, count - w);
    for (std::size_t start = first; start <= last; ++start) {
        bool wins = true;
        for (std::size_t i = start; i < start + w; ++i) {
            if (!ranks[i] || *ranks[i] < *ranks[target] ||
                (i < target && *ranks[i] == *ranks[target])) {
                wins = false;
                break;
            }
        }
        if (wins) return 1;
    }
    return 0;
}
