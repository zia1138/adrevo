# math benchmarks from skydiscover

`math` bencmarks from the [AdaEvolve](https://arxiv.org/abs/2602.20133) paper
Authors report results in Table 1 and in Table 7.

Benchmarks were obtained from the [skydiscover](https://github.com/skydiscover-ai/skydiscover) repository
from [benchmarks/math](https://github.com/skydiscover-ai/skydiscover/tree/main/benchmarks/math)
and converted to adrevo format using claude. 


adrevo results so far:


| problem | best_score | cost | 
| ------- | -----------| ---- |
| circle_packing | 2.63598308499572 | $1.27 |
| circle_packing_rect | 2.3621225483750035| $1.81 |
| heilbronn_convex_13 | 0.03092332432109892 | $3.16 |
| heilbronn_triangle |  0.036478788244272434 | $4.61 |
| minimizing_max_min_dist_3 | 0.24005088264958455 | $1.62 | 
| signal_processing | 0.9556612329944687 | $2.61 | 

* circle_packing - pack 26 circles in a unit square
* circle_packing_rect - pack 21 in a rectangle of perimeter = 4 (width + height = 2)
* heilbronn_convex_13 - optimal placement of n points within a convex region of unit area to maximize the area of the smallest
    triangle formed by any three of these points.
* heilbronn_triangle - optimal placement of n points to maximize the minimum triangle area formed by any three points
* minimizing_max_min_dist_3 - generate an optimal arrangement of exactly 14 points
  in 3D space, maximizing the ratio of minimum distance to maximum distance between all point pairs
* signal_processing - improve a signal processing algorithm that filters volatile, non-stationary time series