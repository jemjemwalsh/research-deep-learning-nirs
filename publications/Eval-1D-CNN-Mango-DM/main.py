import papermill as pm

from itertools import product


PREPROCESSING = ["raw", "anderson", "mishra", "mishra_adj"]
SET_SPLIT = ["anderson", "mishra", "mishra_with_outliers"]
TRAINING_ORDER = ["anderson", "mishra", "mishra_with_outliers", "random_1"]
INCLUDE_MISHRA_OUTLIERS = [True, False]
X_SCALED = [True, False]


def main():

    # generate all combinations of parameters
    combinations = list(
        product(PREPROCESSING, SET_SPLIT, TRAINING_ORDER, INCLUDE_MISHRA_OUTLIERS, X_SCALED)
    )

    # remove unwanted combinations
    combinations = [
        combo
        for combo in combinations
        if not (combo[1] == "mishra" and combo[3])
           and not (combo[1] == "mishra_with_outliers" and not combo[3])
    ]

    failed_combinations = []
    # loop through each combination and execute the notebook with the parameters
    for i, combo in enumerate(combinations):
        print(f"Training {i} of {len(combinations)} training combos")
        preprocessing, set_split, training_order, include_mishra_outliers, x_scaled = combo
        params = {
            "PREPROCESSING": preprocessing,
            "SET_SPLIT": set_split,
            "TRAINING_ORDER": training_order,
            "INCLUDE_MISHRA_OUTLIERS": include_mishra_outliers,
            "X_SCALED": x_scaled,
            "SAVE_RESULTS": False,
        }

        # execute the notebook with the specified parameters
        try:
            pm.execute_notebook(
                input_path="1-model-builds.ipynb",
                output_path=None,
                parameters=params
            )
        except Exception as e:
            print(e)
            failed_combinations.append(combo)

    if failed_combinations:
        print("Failed Combinations")
        print(failed_combinations)


if __name__ == "__main__":
    main()
