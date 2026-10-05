## P(clear) by method and fence definition

All probabilities are P(carry clears the fence). 'raw' uses the
ground distance; 'wall8' additionally requires the ball to be above
the 8 ft wall when it arrives. The France ball's measured spray
(-44 deg) is fair, so fairness does not reduce these further.

| method | central carry (ft) | generic 344 raw | generic 344 + 8ft wall | Milwaukee park raw | Milwaukee park + 8ft wall |
|---|---|---|---|---|---|
| Method 1 empirical (EV/LA pool, n=250) | 320 | 12.4% [8.9, 17.1] | 9.6% [6.5, 13.9] | 9.6% [6.5, 13.9] | 6.8% [4.3, 10.6] |
| Method 1 kernel (sigma 1.5) | 320 | 14.6% |  |  |  |
| Method 2 physics (sensitivity MC) | 316 | 9.3% | 5.8% | 5.8% | 3.6% |
| Method 3 ML direct | 321 | 7.3% | 5.8% | 5.8% | 5.0% |
| Method 3 ML physics-residual | 321 | 11.5% | 5.0% | 5.0% | 5.0% |
| Method 3 ML + bat-tracking (2024+) | 314 | 6.8% | 5.7% | 5.7% | 5.0% |