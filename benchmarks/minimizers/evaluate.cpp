// Fixed k-mer selection harness. Build each translation unit separately:
// c++ -std=c++20 -O2 -c evaluate.cpp -o evaluate.o
// c++ -std=c++20 -O2 -c evo/selection.cpp -o selection.o
// c++ evaluate.o selection.o -o evaluate
// Existing simulation CLI options are retained; --json emits structured metrics.

#include "selection_api.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ksim {

struct Config {
    std::size_t sequence_length = 10'000;
    std::size_t sequences_per_gc = 20;
    std::vector<double> gc_fractions{0.30, 0.50, 0.70};
    double mutation_rate = 0.05;
    std::size_t k = 15;
    std::size_t minimizer_window = 19;  // Window size in k-mers.
    std::size_t metric_window = 250;    // Window size in bases.
    std::uint64_t seed = 42;
    bool report_gap_stats = true;
    bool json_output = false;
};

// Candidate code receives an owned, bounded context, never the full sequence.
// The harness alone maps local decisions to absolute positions.
[[nodiscard]] bool valid_kmer(std::string_view sequence, std::size_t begin,
                              std::size_t k) {
    return sequence.substr(begin, k).find_first_not_of("ACGTacgt") ==
           std::string_view::npos;
}

[[nodiscard]] std::uint32_t decide(std::string_view sequence, std::size_t begin,
                                   std::size_t k, std::size_t w) {
    const std::size_t left = std::min(begin, w - 1);
    const std::size_t right = std::min(sequence.size() - begin - k, w - 1);
    const std::string context(sequence.substr(begin - left, left + k + right));
    const auto decision = selection::select_kmer(context, left, k, w);
    if (decision > 1) throw std::runtime_error("selection must return 0 or 1");
    if (decision && !valid_kmer(sequence, begin, k)) {
        throw std::runtime_error("selection accepted an ambiguous target k-mer");
    }
    return decision;
}

[[nodiscard]] std::vector<std::size_t> select_positions(
    std::string_view sequence, std::size_t k, std::size_t w) {
    std::vector<std::size_t> result;
    if (sequence.size() < k) return result;
    for (std::size_t i = 0; i <= sequence.size() - k; ++i) {
        if (decide(sequence, i, k, w)) result.push_back(i);
    }
    return result;
}

// Replay in reverse order after selecting the partner sequence to catch common
// retained-state and call-order shortcuts. This is a check, not a purity proof.
void check_repeatability(std::string_view sequence, std::size_t k, std::size_t w,
                         const std::vector<std::size_t>& selected) {
    std::size_t remaining = selected.size();
    for (std::size_t i = sequence.size() - k + 1; i-- > 0;) {
        const bool expected = remaining > 0 && selected[remaining - 1] == i;
        if (decide(sequence, i, k, w) != static_cast<std::uint32_t>(expected)) {
            throw std::runtime_error("selection depends on call order or state");
        }
        if (expected) --remaining;
    }
}

void check_contract(std::size_t k, std::size_t w) {
    // Exercise ambiguous targets, repetitive contexts and sequence edges even
    // though the scored generator produces only unambiguous random DNA.
    const std::vector<std::string> cases{
        std::string(k + 2 * (w - 1), 'N'),
        std::string(k + 2 * (w - 1), 'A'),
        "ACGTacgtNNACGTacgtACGTacgtNNACGTacgt",
        std::string(k, 'C'),
    };
    std::vector<std::vector<std::size_t>> selections;
    for (const auto& sequence : cases) {
        selections.push_back(select_positions(sequence, k, w));
    }
    for (std::size_t i = cases.size(); i-- > 0;) {
        if (cases[i].size() >= k) {
            check_repeatability(cases[i], k, w, selections[i]);
        }
    }
}

struct PairMetrics {
    std::uint64_t reference_kmers = 0;
    std::uint64_t query_kmers = 0;
    std::uint64_t reference_selected = 0;
    std::uint64_t query_selected = 0;
    std::uint64_t conserved_anchors = 0;
    std::uint64_t windows = 0;
    std::uint64_t hit_windows = 0;
    std::vector<std::size_t> internal_anchor_gaps;
};

struct AggregateMetrics {
    std::uint64_t sequences = 0;
    std::uint64_t reference_kmers = 0;
    std::uint64_t query_kmers = 0;
    std::uint64_t reference_selected = 0;
    std::uint64_t query_selected = 0;
    std::uint64_t conserved_anchors = 0;
    std::uint64_t windows = 0;
    std::uint64_t hit_windows = 0;
    std::vector<std::size_t> internal_anchor_gaps;

    void add(const PairMetrics& pair, bool retain_gaps) {
        ++sequences;
        reference_kmers += pair.reference_kmers;
        query_kmers += pair.query_kmers;
        reference_selected += pair.reference_selected;
        query_selected += pair.query_selected;
        conserved_anchors += pair.conserved_anchors;
        windows += pair.windows;
        hit_windows += pair.hit_windows;

        if (retain_gaps) {
            internal_anchor_gaps.insert(internal_anchor_gaps.end(),
                                        pair.internal_anchor_gaps.begin(),
                                        pair.internal_anchor_gaps.end());
        }
    }
};

// The simulation knows the true coordinate homology. A conserved anchor must
// therefore be selected in both sequences at the same start coordinate and
// contain exactly the same k-mer. This is deliberately different from mapper
// anchor generation, which reports every matching occurrence, including repeats.
[[nodiscard]] std::vector<std::size_t> find_conserved_anchor_positions(
    std::string_view reference, std::string_view query, std::size_t k,
    const std::vector<std::size_t>& reference_seeds,
    const std::vector<std::size_t>& query_seeds) {
    std::vector<std::size_t> anchors;
    std::size_t reference_index = 0;
    std::size_t query_index = 0;

    while (reference_index < reference_seeds.size() &&
           query_index < query_seeds.size()) {
        const std::size_t reference_seed = reference_seeds[reference_index];
        const std::size_t query_seed = query_seeds[query_index];

        if (reference_seed < query_seed) {
            ++reference_index;
            continue;
        }
        if (query_seed < reference_seed) {
            ++query_index;
            continue;
        }

        if (reference.substr(reference_seed, k) ==
                query.substr(query_seed, k)) {
            anchors.push_back(reference_seed);
        }

        ++reference_index;
        ++query_index;
    }

    return anchors;
}

[[nodiscard]] PairMetrics evaluate_pair(
    std::string_view reference, std::string_view query,
    std::size_t k, std::size_t w, std::size_t metric_window,
    bool collect_gaps) {
    if (reference.size() != query.size()) {
        throw std::invalid_argument("reference and query must have equal lengths");
    }

    if (k == 0 || k > reference.size()) {
        throw std::invalid_argument("invalid k-mer length");
    }
    if (metric_window == 0 || metric_window > reference.size()) {
        throw std::invalid_argument("invalid metric window length");
    }

    const auto reference_seeds = select_positions(reference, k, w);
    const auto query_seeds = select_positions(query, k, w);
    check_repeatability(query, k, w, query_seeds);
    check_repeatability(reference, k, w, reference_seeds);

    const std::vector<std::size_t> anchors = find_conserved_anchor_positions(
        reference, query, k, reference_seeds, query_seeds);

    PairMetrics metrics;
    metrics.reference_kmers = reference.size() - k + 1;
    metrics.query_kmers = query.size() - k + 1;
    metrics.reference_selected = reference_seeds.size();
    metrics.query_selected = query_seeds.size();
    metrics.conserved_anchors = anchors.size();

    // Evaluate all overlapping, one-base-stride windows [start, end). An anchor
    // hits a window when the anchor's start coordinate lies inside the window.
    metrics.windows = reference.size() - metric_window + 1;
    std::size_t first_possible_anchor = 0;

    for (std::size_t start = 0; start + metric_window <= reference.size();
         ++start) {
        while (first_possible_anchor < anchors.size() &&
               anchors[first_possible_anchor] < start) {
            ++first_possible_anchor;
        }

        if (first_possible_anchor < anchors.size() &&
            anchors[first_possible_anchor] < start + metric_window) {
            ++metrics.hit_windows;
        }
    }

    if (collect_gaps && anchors.size() >= 2) {
        metrics.internal_anchor_gaps.reserve(anchors.size() - 1);
        for (std::size_t i = 1; i < anchors.size(); ++i) {
            metrics.internal_anchor_gaps.push_back(anchors[i] - anchors[i - 1]);
        }
    }

    return metrics;
}

[[nodiscard]] std::string generate_reference(std::size_t length, double gc,
                                             std::mt19937_64& rng) {
    std::bernoulli_distribution choose_gc(gc);
    std::bernoulli_distribution choose_first(0.5);
    std::string sequence;
    sequence.reserve(length);

    for (std::size_t i = 0; i < length; ++i) {
        if (choose_gc(rng)) {
            sequence.push_back(choose_first(rng) ? 'G' : 'C');
        } else {
            sequence.push_back(choose_first(rng) ? 'A' : 'T');
        }
    }
    return sequence;
}

// Substitutions preserve a direct coordinate map between reference and query.
[[nodiscard]] std::string mutate_by_substitution(std::string_view reference,
                                                 double mutation_rate,
                                                 std::mt19937_64& rng) {
    static constexpr char bases[] = {'A', 'C', 'G', 'T'};
    std::bernoulli_distribution mutate(mutation_rate);
    std::uniform_int_distribution<int> alternative(0, 2);
    std::string query(reference);

    for (char& base : query) {
        if (!mutate(rng)) continue;

        int choice = alternative(rng);
        for (const char candidate : bases) {
            if (candidate == base) continue;
            if (choice-- == 0) {
                base = candidate;
                break;
            }
        }
    }
    return query;
}

[[nodiscard]] double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0
               ? std::numeric_limits<double>::quiet_NaN()
               : static_cast<double>(numerator) /
                     static_cast<double>(denominator);
}

[[nodiscard]] std::size_t nearest_rank_percentile(
    std::vector<std::size_t> values, double percentile) {
    if (values.empty()) throw std::invalid_argument("percentile of empty sample");
    std::sort(values.begin(), values.end());
    const std::size_t rank = static_cast<std::size_t>(
        std::ceil(percentile * static_cast<double>(values.size())));
    return values[std::max<std::size_t>(1, rank) - 1];
}

void print_metrics(std::string_view label, const AggregateMetrics& metrics,
                   bool report_gap_stats) {
    std::cout << label << '\n'
              << "  sequences: " << metrics.sequences << '\n'
              << "  density_reference: "
              << ratio(metrics.reference_selected, metrics.reference_kmers) << '\n'
              << "  density_query: "
              << ratio(metrics.query_selected, metrics.query_kmers) << '\n'
              << "  density_combined: "
              << ratio(metrics.reference_selected + metrics.query_selected,
                       metrics.reference_kmers + metrics.query_kmers)
              << '\n'
              << "  conserved_anchors: " << metrics.conserved_anchors << '\n'
              << "  P_hit: " << ratio(metrics.hit_windows, metrics.windows) << '\n'
              << "  hit_windows: " << metrics.hit_windows << '/' << metrics.windows
              << '\n';

    if (!report_gap_stats) return;

    if (metrics.internal_anchor_gaps.empty()) {
        std::cout << "  anchor_gap_p99: n/a\n"
                  << "  anchor_gap_max: n/a\n";
    } else {
        std::cout << "  anchor_gap_p99: "
                  << nearest_rank_percentile(metrics.internal_anchor_gaps, 0.99)
                  << '\n'
                  << "  anchor_gap_max: "
                  << *std::max_element(metrics.internal_anchor_gaps.begin(),
                                       metrics.internal_anchor_gaps.end())
                  << '\n';
    }
}

void print_json_metrics(const AggregateMetrics& m, bool report_gaps) {
    std::cout << std::setprecision(17)
              << "{\"sequences\":" << m.sequences
              << ",\"reference_kmers\":" << m.reference_kmers
              << ",\"query_kmers\":" << m.query_kmers
              << ",\"reference_selected\":" << m.reference_selected
              << ",\"query_selected\":" << m.query_selected
              << ",\"conserved_anchors\":" << m.conserved_anchors
              << ",\"windows\":" << m.windows
              << ",\"hit_windows\":" << m.hit_windows;
    if (report_gaps && !m.internal_anchor_gaps.empty()) {
        std::cout << ",\"anchor_gap_p99\":"
                  << nearest_rank_percentile(m.internal_anchor_gaps, 0.99)
                  << ",\"anchor_gap_max\":"
                  << *std::max_element(m.internal_anchor_gaps.begin(),
                                       m.internal_anchor_gaps.end());
    }
    std::cout << '}';
}

[[nodiscard]] std::vector<double> parse_gc_list(std::string_view text) {
    std::vector<double> values;
    std::size_t begin = 0;

    while (begin <= text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end =
            comma == std::string_view::npos ? text.size() : comma;
        const std::string token(text.substr(begin, end - begin));
        if (token.empty()) throw std::invalid_argument("empty value in --gc list");

        std::size_t consumed = 0;
        const double value = std::stod(token, &consumed);
        if (consumed != token.size() || !std::isfinite(value) || value < 0.0 ||
            value > 1.0) {
            throw std::invalid_argument("each --gc value must be in [0, 1]");
        }
        values.push_back(value);

        if (comma == std::string_view::npos) break;
        begin = comma + 1;
    }
    return values;
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(std::string_view text,
                                    std::string_view option) {
    Integer value{};
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const auto [end, error] = std::from_chars(first, last, value);
    if (error != std::errc{} || end != last) {
        throw std::invalid_argument("invalid integer for " + std::string(option));
    }
    return value;
}

[[nodiscard]] double parse_probability(std::string_view text,
                                       std::string_view option) {
    const std::string owned(text);
    std::size_t consumed = 0;
    const double value = std::stod(owned, &consumed);
    if (consumed != owned.size() || !std::isfinite(value) || value < 0.0 ||
        value > 1.0) {
        throw std::invalid_argument(std::string(option) + " must be in [0, 1]");
    }
    return value;
}

void print_help(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --length N              Reference length (default: 10000)\n"
        << "  --sequences-per-gc N    Replicates per GC value (default: 20)\n"
        << "  --gc X,Y,...            GC fractions (default: 0.3,0.5,0.7)\n"
        << "  --mutation-rate X       Substitution probability (default: 0.05)\n"
        << "  --k N                   K-mer length, at most 31 (default: 15)\n"
        << "  --w N                   Minimizer window in k-mers (default: 19)\n"
        << "  --window N              Metric window in bases (default: 250)\n"
        << "  --seed N                Random seed (default: 42)\n"
        << "  --no-gap-stats          Omit p99 and maximum internal anchor gaps\n"
        << "  --json                  Emit JSON metrics\n"
        << "  --help                  Show this message\n";
}

[[nodiscard]] Config parse_arguments(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);

        if (option == "--help") {
            print_help(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (option == "--json") {
            config.json_output = true;
            continue;
        }
        if (option == "--no-gap-stats") {
            config.report_gap_stats = false;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + std::string(option));
        }

        const std::string_view value(argv[++i]);
        if (option == "--length") {
            config.sequence_length = parse_integer<std::size_t>(value, option);
        } else if (option == "--sequences-per-gc") {
            config.sequences_per_gc = parse_integer<std::size_t>(value, option);
        } else if (option == "--gc") {
            config.gc_fractions = parse_gc_list(value);
        } else if (option == "--mutation-rate") {
            config.mutation_rate = parse_probability(value, option);
        } else if (option == "--k") {
            config.k = parse_integer<std::size_t>(value, option);
        } else if (option == "--w") {
            config.minimizer_window = parse_integer<std::size_t>(value, option);
        } else if (option == "--window") {
            config.metric_window = parse_integer<std::size_t>(value, option);
        } else if (option == "--seed") {
            config.seed = parse_integer<std::uint64_t>(value, option);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
    }

    if (config.sequence_length == 0 || config.sequences_per_gc == 0 ||
        config.gc_fractions.empty()) {
        throw std::invalid_argument("length, replicates, and GC list must be nonempty");
    }
    if (config.k == 0 || config.k > 31 || config.k > config.sequence_length) {
        throw std::invalid_argument("--k must be in [1, min(31, --length)]");
    }
    const std::size_t kmer_count = config.sequence_length - config.k + 1;
    if (config.minimizer_window == 0 || config.minimizer_window > kmer_count) {
        throw std::invalid_argument("--w must be in [1, number of k-mers]");
    }
    if (config.metric_window == 0 ||
        config.metric_window > config.sequence_length) {
        throw std::invalid_argument("--window must be in [1, --length]");
    }

    return config;
}

}  // namespace ksim

int main(int argc, char** argv) {
    try {
        const ksim::Config config = ksim::parse_arguments(argc, argv);
        std::mt19937_64 rng(config.seed);
        ksim::check_contract(config.k, config.minimizer_window);

        if (!config.json_output) std::cout << std::fixed << std::setprecision(6)
                  << "selection_method: candidate-local-selection" << '\n'
                  << "seed: " << config.seed << '\n'
                  << "sequence_length: " << config.sequence_length << '\n'
                  << "sequences_per_gc: " << config.sequences_per_gc << '\n'
                  << "mutation_rate: " << config.mutation_rate << '\n'
                  << "k: " << config.k << '\n'
                  << "minimizer_window_kmers: " << config.minimizer_window << '\n'
                  << "metric_window_bases: " << config.metric_window << '\n';

        ksim::AggregateMetrics overall;
        if (config.json_output) std::cout << "{\"groups\":[";
        bool first_group = true;

        for (const double gc : config.gc_fractions) {
            ksim::AggregateMetrics group;

            for (std::size_t replicate = 0;
                 replicate < config.sequences_per_gc; ++replicate) {
                const std::string reference =
                    ksim::generate_reference(config.sequence_length, gc, rng);
                const std::string query = ksim::mutate_by_substitution(
                    reference, config.mutation_rate, rng);

                const ksim::PairMetrics pair = ksim::evaluate_pair(
                    reference, query, config.k, config.minimizer_window, config.metric_window,
                    config.report_gap_stats);

                group.add(pair, config.report_gap_stats);
                overall.add(pair, config.report_gap_stats);
            }

            if (config.json_output) {
                if (!first_group) std::cout << ',';
                first_group = false;
                std::cout << "{\"gc\":" << gc << ",\"metrics\":";
                ksim::print_json_metrics(group, config.report_gap_stats);
                std::cout << '}';
            } else {
                std::cout << '\n';
                ksim::print_metrics("GC=" + std::to_string(gc), group,
                                    config.report_gap_stats);
            }
        }

        if (config.json_output) {
            std::cout << "],\"overall\":";
            ksim::print_json_metrics(overall, config.report_gap_stats);
            std::cout << "}\n";
        } else {
            std::cout << '\n';
            ksim::print_metrics("overall", overall, config.report_gap_stats);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
