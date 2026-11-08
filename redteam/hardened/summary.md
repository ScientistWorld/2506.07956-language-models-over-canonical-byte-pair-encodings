| case | kind | Δ test (proposed) | Δ train (proposed) | test status |
|---|---|---:|---:|---|
| reference_seed14 | variant | +0.0000 | +0.0000 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
| alpha_tuned | variant | +0.2326 | +0.2404 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
| kneser_ney | variant | +1.9008 | +1.9913 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
| less_data_3000 | variant | -17.9931 | -17.9671 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
| alpha_high | variant | -7.1228 | -7.2795 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
| unigram_canonical | baseline | -77.0147 | -80.1389 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
| uniform_canonical | baseline | -166.6610 | -170.4482 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
| dev_memorization | shortcut | +0.4938 | +6.9986 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
| test_memorization | shortcut | +7.2509 | +0.4611 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
| unconstrained_reference | shortcut | -166.6703 | -170.4576 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| unconstrained_kneser_ney | shortcut | -166.6614 | -170.4487 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| noncanonical_leak | shortcut | -166.6610 | -170.4482 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| overweight_rows | shortcut | -167.4610 | -171.2482 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| nan_weights | shortcut | -167.4610 | -171.2482 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| negative_weights | shortcut | -167.4610 | -171.2482 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| support_shrunk | shortcut | -167.4610 | -171.2482 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| support_permuted | shortcut | -167.4610 | -171.2482 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| reserved_baseline_dir | shortcut | +0.0000 | +0.0000 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
| drop_ptb | shortcut | -89.1393 | -92.6542 | global_canonical_bigram_lm:FAILED, global_canonical_bigram_lm:ok, local_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:ok |
| empty_string | shortcut | -167.4610 | -171.2482 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| nontight | shortcut | -167.4610 | -171.2482 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| pickled_npz | shortcut | -167.4610 | -171.2482 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| junk_artifacts | shortcut | +0.0000 | +0.0000 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
| tamper_tokenizer_and_support | shortcut | -166.6614 | -170.4487 | global_canonical_bigram_lm:FAILED, local_canonical_bigram_lm:FAILED |
| score_file_forgery | shortcut | +0.0000 | +0.0000 | global_canonical_bigram_lm:ok, local_canonical_bigram_lm:ok |
