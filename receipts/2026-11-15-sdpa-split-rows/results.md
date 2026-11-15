# Results (generated from summary.tsv)

Each cell: speedup over plain decode with the flag unset, then with the flag set. "ids differ" marks a cell whose token ids differ from plain decode in that arm.
Where a prompt was run twice (two samples per arm, then six), the cell with more samples is shown; `summary.tsv` has both.

## 14-inch M2 Max (G14C), prompt lookup k=3

| model | about prompt tokens | code | quote | chat | samples per arm |
| --- | --- | --- | --- | --- | --- |
| 2B | 400 | 0.86 to 0.86 (ids differ, unset) | 1.09 to 1.07 | 0.92 to 0.93 | 6 |
| 2B | 1000 | 0.79 to 0.83 | 1.02 to 1.08 |  | 6 |
| 2B | 2500 | 0.78 to 0.88 | 0.90 to 1.09 |  | 2 |
| 2B | 5000 | 0.71 to 0.89 | 0.82 to 1.12 |  | 2 |
| 2B | 10000 | 0.64 to 0.88 | 0.60 to 1.04 |  | 2 |
| 4B | 400 | 0.92 to 0.99 | 1.09 to 1.28 | 0.91 to 0.91 | 6 |
| 4B | 1000 | 0.80 to 1.01 (ids differ, unset) | 0.89 to 1.28 |  | 2 |
| 4B | 2500 | 0.69 to 1.01 (ids differ, unset) | 0.59 to 1.28 |  | 2 |
| 4B | 5000 | 0.50 to 0.94 (ids differ, unset) | 0.39 to 1.19 |  | 2 |
| 4B | 10000 | 0.47 to 0.89 | 0.39 to 1.09 |  | 2 |
| 9B | 400 | 0.93 to 0.93 | 1.55 to 1.72 | 0.92 to 0.92 | 6 |
| 9B | 1000 | 0.90 to 0.93 (ids differ, set) | 1.47 to 1.51 |  | 2 |
| 9B | 2500 | 0.89 to 0.93 | 1.17 to 1.25 |  | 2 |
| 9B | 5000 | 0.84 to 0.91 (ids differ, set) (ids differ, unset) | 1.02 to 1.19 (ids differ, set) (ids differ, unset) |  | 2 |
| 9B | 10000 | 0.77 to 0.90 | 0.77 to 1.13 |  | 2 |

## 14-inch M2 Max (G14C), draft route (9B target, 2B draft)

| model | about prompt tokens | code | quote | chat | samples per arm |
| --- | --- | --- | --- | --- | --- |
| draft: 9B target, 2B draft | 400 | 0.60 to 0.60 | 0.72 to 0.73 | 0.54 to 0.53 | 2 |
| draft: 9B target, 2B draft | 2500 | 0.55 to 0.63 | 0.56 to 0.61 |  | 2 |

## 16-inch M1 Max (G13C), prompt lookup k=3

| model | about prompt tokens | code | quote | chat | samples per arm |
| --- | --- | --- | --- | --- | --- |
| 2B | 400 | 0.95 to 0.93 (ids differ, unset) | 1.37 to 1.38 | 0.92 to 0.92 | 6 |
| 2B | 1000 | 0.88 to 0.91 | 1.30 to 1.38 |  | 6 |
| 2B | 2500 | 0.85 to 0.94 | 1.11 to 1.37 |  | 2 |
| 4B | 400 | 1.06 to 1.11 | 1.42 to 1.56 | 0.96 to 0.95 | 6 |
| 4B | 2500 | 0.77 to 1.10 (ids differ, unset) | 0.70 to 1.48 |  | 2 |
| 9B | 400 | 0.96 to 0.95 | 1.51 to 1.51 | 0.98 to 0.98 | 2 |
| 9B | 2500 | 0.91 to 0.95 | 1.14 to 1.25 |  | 2 |

## 16-inch M1 Max (G13C), draft route (9B target, 2B draft)

| model | about prompt tokens | code | quote | chat | samples per arm |
| --- | --- | --- | --- | --- | --- |
| 9B target, 2B draft | 400 | 0.59 to 0.60 | 0.71 to 0.71 | 0.52 to 0.52 (ids differ, set) (ids differ, unset) | 2 |

