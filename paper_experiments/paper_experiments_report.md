# Paper Experiments Report

## Held-out Question Prediction

| Dataset | Feature Set | Accuracy | Balanced Accuracy | AUC | Q-Local AUC | Length Q-Local AUC | Avg Selected Features |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt_oss_full | motif_1to3_full_trace | 0.5145 | 0.5110 | 0.5070 | 0.5422 | 0.6549 | 657.0 |
| gpt_oss_mixed_questions | motif_1to3_full_trace | 0.5454 | 0.5459 | 0.5403 | 0.5559 | 0.6541 | 607.4 |
| deepseek_full | motif_1to3_full_trace | 0.5473 | 0.5469 | 0.5500 | 0.5123 | 0.6137 | 454.6 |
| deepseek_mixed_questions | motif_1to3_full_trace | 0.5424 | 0.5444 | 0.5569 | 0.5328 | 0.6117 | 386.6 |

## Early-Prefix Prediction

| Dataset | Prefix Fraction | Motif AUC | Motif Q-Local AUC | Length Q-Local AUC | Motif Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpt_oss_prefix_0.25 | 0.25 | 0.5495 | 0.5476 | 0.6412 | 0.5391 |
| gpt_oss_prefix_0.50 | 0.50 | 0.5020 | 0.5384 | 0.6507 | 0.5249 |
| gpt_oss_prefix_0.75 | 0.75 | 0.4829 | 0.5201 | 0.6530 | 0.5154 |
| gpt_oss_prefix_1.00 | 1.00 | 0.5070 | 0.5422 | 0.6549 | 0.5110 |
| deepseek_prefix_0.25 | 0.25 | 0.5393 | 0.4906 | 0.5730 | 0.5433 |
| deepseek_prefix_0.50 | 0.50 | 0.5009 | 0.5027 | 0.6037 | 0.5132 |
| deepseek_prefix_0.75 | 0.75 | 0.5007 | 0.5077 | 0.6090 | 0.5091 |
| deepseek_prefix_1.00 | 1.00 | 0.5500 | 0.5123 | 0.6137 | 0.5469 |

## Cross-Model Transfer

| Transfer | Prefix Fraction | Motif AUC | Motif Q-Local AUC | Length Q-Local AUC | Motif Balanced Accuracy | Selected Features |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_gpt_oss_test_deepseek | 1.00 | 0.5976 | 0.5351 | 0.6111 | 0.5626 | 760 |
| train_deepseek_test_gpt_oss | 1.00 | 0.5622 | 0.4937 | 0.6541 | 0.5560 | 522 |

## Shared Motif Stability

- Shared features between gpt_oss and deepseek: 357
- Sign agreement rate on shared features: 0.5546
- Shared success motifs: 96
- Shared failure motifs: 102

| Motif | Direction | Left Weight | Right Weight | Weight Sum |
| --- | --- | ---: | ---: | ---: |
| solve conclude | success | 3.7155 | 3.2004 | 6.9159 |
| solve | success | 4.7721 | 1.0217 | 5.7937 |
| derive-intermediate solve | success | 3.1211 | 2.4216 | 5.5428 |
| simplify isolate | success | 1.9527 | 3.0704 | 5.0230 |
| isolate compute | success | 2.0429 | 2.7468 | 4.7897 |
| isolate | success | 1.1891 | 3.2408 | 4.4299 |
| instantiate substitute | success | 0.8606 | 3.2794 | 4.1401 |
| substitute compute | success | 0.3645 | 3.2794 | 3.6439 |
| apply-formula case-split | failure | -0.2722 | -3.1480 | 3.4202 |
| backtrack rewrite | failure | -0.9443 | -2.4066 | 3.3509 |
| compute backtrack | failure | -1.2936 | -2.0056 | 3.2992 |
| apply-formula compute analyze | failure | -1.8885 | -1.3001 | 3.1886 |
| derive-intermediate simplify conclude | success | 1.7116 | 1.4363 | 3.1479 |
| rewrite conclude | success | 1.3948 | 1.5135 | 2.9083 |
| backtrack | failure | -1.1610 | -1.7250 | 2.8860 |
| compare instantiate | failure | -0.9928 | -1.8390 | 2.8318 |
| instantiate rewrite analyze | success | 0.9558 | 1.8081 | 2.7639 |
| analyze compute check-constraint | failure | -1.2303 | -1.3700 | 2.6002 |
| instantiate check-constraint derive-intermediate | failure | -2.4113 | -0.0386 | 2.4499 |
| apply-formula derive-intermediate derive-intermediate | failure | -0.0296 | -2.2183 | 2.2479 |
| conclude compute conclude | success | 1.4680 | 0.7427 | 2.2108 |
| apply-formula apply-formula apply-formula | failure | -0.7637 | -1.4354 | 2.1990 |
| compute rewrite derive-intermediate | success | 0.5705 | 1.5878 | 2.1583 |
| compute analyze apply-formula | failure | -0.4786 | -1.6585 | 2.1371 |
| derive-intermediate rewrite simplify | failure | -2.0462 | -0.0387 | 2.0850 |

## Top Weighted Motifs: gpt_oss

### Success-leaning

| Motif | Weight | Support |
| --- | ---: | ---: |
| solve | 4.7721 | 92 |
| solve conclude | 3.7155 | 33 |
| solve compute | 3.6848 | 32 |
| simplify derive-intermediate compare | 3.6206 | 30 |
| compute solve | 3.5522 | 28 |
| simplify simplify isolate | 3.5163 | 27 |
| add-fractions | 3.4791 | 26 |
| compute-sum | 3.4791 | 26 |
| find-common-denominator | 3.4791 | 26 |
| find-common-denominator add-fractions | 3.4791 | 26 |
| identify-parameters | 3.4791 | 26 |
| identify-parameters compute-sum | 3.4791 | 26 |
| instantiate transform | 3.4406 | 25 |
| transform | 3.4406 | 25 |
| add-fractions identify-parameters | 3.4005 | 24 |

### Failure-leaning

| Motif | Weight | Support |
| --- | ---: | ---: |
| count-favorable | -3.2248 | 28 |
| identify-symmetry | -3.1890 | 27 |
| analyze count-total | -3.1519 | 26 |
| count-total | -3.1519 | 26 |
| analyze apply-identity | -3.1135 | 25 |
| rearrange compute | -3.1135 | 25 |
| case-split rearrange | -3.0736 | 24 |
| compute check-constraint rearrange | -3.0736 | 24 |
| case-split rearrange compute | -3.0320 | 23 |
| check-constraint rearrange compute | -3.0320 | 23 |
| count-favorable apply-formula | -3.0320 | 23 |
| rearrange compute check-constraint | -3.0320 | 23 |
| apply-identity case-split | -2.8964 | 20 |
| apply-identity case-split rearrange | -2.8964 | 20 |
| normalize | -2.8964 | 20 |

## Top Weighted Motifs: deepseek

### Success-leaning

| Motif | Weight | Support |
| --- | ---: | ---: |
| instantiate substitute | 3.2794 | 26 |
| instantiate substitute compute | 3.2794 | 26 |
| substitute compute | 3.2794 | 26 |
| isolate | 3.2408 | 25 |
| list | 3.2408 | 25 |
| analyze instantiate substitute | 3.2007 | 24 |
| equate | 3.2007 | 24 |
| equate solve | 3.2007 | 24 |
| equate solve conclude | 3.2007 | 24 |
| solve solve conclude | 3.2007 | 24 |
| solve conclude | 3.2004 | 49 |
| apply-formula equate | 3.1591 | 23 |
| apply-formula equate solve | 3.1591 | 23 |
| convert-units | 3.1591 | 23 |
| derive-intermediate solve solve | 3.1157 | 22 |

### Failure-leaning

| Motif | Weight | Support |
| --- | ---: | ---: |
| combine | -3.2785 | 24 |
| compute combine | -3.2785 | 24 |
| instantiate compute combine | -3.2367 | 23 |
| simplify solve-equation | -3.2367 | 23 |
| substitute-values | -3.2367 | 23 |
| substitute-values simplify | -3.2367 | 23 |
| apply-formula case-split | -3.1480 | 21 |
| analyze apply-formula case-split | -3.1005 | 20 |
| apply-formula case-split rewrite | -3.1005 | 20 |
| check-consistency | -2.8857 | 16 |
| solve-equation check-consistency | -2.8857 | 16 |
| compute compute backtrack | -2.8242 | 15 |
| simplify simplify solve-equation | -2.7588 | 14 |
| substitute-values simplify simplify | -2.7588 | 14 |
| simplify solve-equation check-consistency | -2.6889 | 13 |
