import os
import papermill as pm


NOTEBOOKS = [
    "1_model_1.ipynb",
    "1_model_2.ipynb",
    "1_model_3.ipynb",
    "1_model_4.ipynb",
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
