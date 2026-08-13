# ADRS benchmarks from skydiscover

`ADRS` benchmarks from the [AdaEvolve](https://arxiv.org/abs/2602.20133) paper
Authors report results in Table 2 and in Table 6.

Benchmarks were obtained from the [skydiscover](https://github.com/skydiscover-ai/skydiscover) repository
from [benchmarks/ADRS](https://github.com/skydiscover-ai/skydiscover/tree/main/benchmarks/ADRS)
and converted to adrevo format using claude. 



adrevo results so far on ADRS

| problem | best_score | cost | 
| ------- | -----------| ---- |
| cloud_cast | 618.069933 | $3.20 |
| eplb | 0.15160636497402213 | $1.99 |
| llm_sql | 0.9893395538070844 | $1.11 |
| prism |  26.255971749535238 | $0.22 |
| txn_scheduling |  4310.3448275862065 | $3.53 |

(*) prism, cloudcast, and txn_scheduling required hardening the evaluator from skydiscover.

* cloudcast - implement search algorithm to minimize overall data transfer cost across multiple clouds
* eplib - improve the Mixture-of-Expert models Expert Parallelism Load Balancer (MoE EPLB) expert rearrangement algorithm.
* llm_sql - maximize prefix hit count (PHC) for efficient LLM prompt caching.
* prism -  improve a model placement algorithm foravailable GPUs.
* txn_scheduling - improve a scheduling function to find better schedules for
    transactional workloads made up of read and write operations to data items.