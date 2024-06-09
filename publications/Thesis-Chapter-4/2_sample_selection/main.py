import os
import papermill as pm


NOTEBOOKS = [
    "1_population_selection.ipynb",
    "2_train_test_partition.ipynb",
    "3_outlier_removal_v2.ipynb",
    "4_sample_order.ipynb",
    "5_subsequently_flagged.ipynb",
]


def main():
    
    # change current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    for notebook in NOTEBOOKS:
        print(notebook)
        pm.execute_notebook(
            input_path=notebook,
            output_path=None,
        )


if __name__ == "__main__":
    main()
