import os

import papermill as pm


NOTEBOOKS = [
    "1_combine_and_dedupe_raw_datasets.ipynb",
    "2_clean_deduped_dataset.ipynb",
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
    