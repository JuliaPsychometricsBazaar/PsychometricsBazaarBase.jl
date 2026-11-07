Run `julia --project=. benchmark/grid_moments.jl` with one thread. The script
warms compilation before timing and reports the minimum of five batches of
10,000 reductions. Input-array setup is measured separately; scalar reductions
need no workspace. Timings are descriptive, not performance assertions.

Example local run on Julia 1.12.7, one thread (2026-11-07):

| Points | Setup bytes | Setup ns | Bytes per reduction | ns per reduction |
| --- | --- | --- | --- | --- |
| 33 | 640 | 1696 | 0 | 226.7 |
| 65 | 1152 | 2648 | 0 | 423.7 |
| 129 | 2240 | 2066 | 0 | 851.3 |

This measures only the scalar numerical helper, not end-to-end CAT selection.
