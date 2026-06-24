# Reproducibility Notes

1. Run scripts in numerical order from `task1` to `task4`.
2. Task 2 can take some time because it evaluates four corpus versions, six vocabulary sizes, and ten folds.
3. Task 3 depends on Task 2 prediction files.
4. Task 4 training-size experiment uses the `stop` configuration with `N = all`, because it had the highest TCR at lambda = 9 in the preprocessing comparison.
5. `task4_generate_final_outputs.py` creates the final report figures from CSV results.
