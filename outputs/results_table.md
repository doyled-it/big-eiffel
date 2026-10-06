## P(clear) by method and fence definition

All probabilities are P(carry clears the fence). 'raw' uses the
ground distance; 'wall8' additionally requires the ball to be above
the 8 ft wall when it arrives. The France ball's measured spray
(-44 deg) is fair, so fairness does not reduce these further.

| method | central carry (ft) | generic 344 raw | generic 344 + 8ft wall | Milwaukee park raw | Milwaukee park + 8ft wall |
|---|---|---|---|---|---|
| Method 1 empirical (2015-2026 pool, n=280) | 319 | 11.8% [8.5, 16.1] | 8.6% [5.8, 12.4] | 8.6% [5.8, 12.4] | 6.4% [4.1, 9.9] |
| Method 1 kernel (sigma 1.5) | 320 | 14.1% |  |  |  |
| Method 2 physics (sensitivity MC) | 318 | 10.9% | 8.0% | 7.5% | 5.9% |
| Method 3 ML direct | 321 | 6.6% | 5.7% | 5.5% | 5.0% |
| Method 3 ML physics-residual | 321 | 5.0% | 5.0% | 5.0% | 5.0% |
| Method 3 ML + bat-tracking (2024+) | 320 | 8.5% | 7.6% | 7.4% | 6.5% |