# FANI temporal experiment — real-data results

Initialization: 2019-04-29 00 UTC; +72…+120 h in 6-hour steps. Five members, including control. Threshold 17.5 m/s, minimum area 5000 km². Forecast-only segment selection frozen before temporal execution.

| Method | Selected times /9 | Mean gale-centroid error km | Median km | Maximum km | Algorithmic segments |
|---|---:|---:|---:|---:|---:|
| member_0 | 9 | 460.18 | 481.42 | 731.96 | 2 |
| member_1 | 9 | 329.13 | 288.63 | 650.10 | 2 |
| member_2 | 4 | 464.46 | 464.43 | 519.52 | 5 |
| member_3 | 5 | 676.21 | 668.69 | 750.43 | 6 |
| member_4 | 4 | 536.37 | 542.76 | 584.15 | 4 |
| ensemble_mean | 1 | 579.95 | 579.95 | 579.95 | 1 |

| Lead h | Method | Forecast objects | ERA5 objects | Hits | Misses | False alarms | Footprint IoU | Forecast peak m/s | ERA5 peak m/s | IMD sustained m/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|72|ensemble_mean|0|1|0|1|0|0.0|15.239494323730469|28.5328426361084|54.01662|
|72|member_support_0p4|1|1|1|0|0|0.2153846153846154|34.23643112182617|28.5328426361084|54.01662|
|78|ensemble_mean|0|1|0|1|0|0.0|17.343006134033203|30.416839599609375|56.588840000000005|
|78|member_support_0p4|1|1|1|0|0|0.07035175879396985|33.030269622802734|30.416839599609375|56.588840000000005|
|84|ensemble_mean|0|1|0|1|0|0.0|16.050439834594727|26.13278579711914|59.16106|
|84|member_support_0p4|1|1|0|1|1|0.0|27.135984420776367|26.13278579711914|59.16106|
|90|ensemble_mean|0|1|0|1|0|0.0|15.003247261047363|26.54711151123047|59.16106|
|90|member_support_0p4|1|1|0|1|1|0.0|28.315196990966797|26.54711151123047|59.16106|
|96|ensemble_mean|0|1|0|1|0|0.0|18.009532928466797|26.854251861572266|54.01662|
|96|member_support_0p4|1|1|0|1|1|0.0|35.77524948120117|26.854251861572266|54.01662|
|102|ensemble_mean|1|1|0|1|1|0.0|19.540359497070312|27.9755916595459|43.727740000000004|
|102|member_support_0p4|2|1|0|1|2|0.0|35.396583557128906|27.9755916595459|43.727740000000004|
|108|ensemble_mean|0|1|0|1|0|0.0|17.968935012817383|22.44320297241211|36.01108|
|108|member_support_0p4|2|1|0|1|2|0.0|37.658451080322266|22.44320297241211|36.01108|
|114|ensemble_mean|0|1|0|1|0|0.0|17.221908569335938|23.24846839904785|28.294420000000002|
|114|member_support_0p4|2|1|0|1|2|0.0|40.421512603759766|23.24846839904785|28.294420000000002|
|120|ensemble_mean|0|1|0|1|0|0.0|16.622365951538086|19.853805541992188|20.57776|
|120|member_support_0p4|2|1|1|0|1|0.0|37.33979034423828|19.853805541992188|20.57776|

**Limits:** Wind-footprint centroids are not validated cyclone centers. The 300-km object matching gate permits hits with zero IoU. Split/merge labels and segment fragmentation are algorithmic, not independently verified storm identity errors. Temporal EFI is not included. No valid landfall timing was demonstrated; closest Puri approach is explicitly a proxy. No method/threshold was tuned on these results.
