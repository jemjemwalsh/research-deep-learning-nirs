# Deep Learning for NIR Spectroscopy :mango:


## Project Overview
A repository to accompany a Master of Research project conducted within the School of Engineering and Technology at CQUniversity. The research focuses on leveraging deep learning techniques for the estimation of fruit quality attributes using Near-Infrared (NIR) Spectroscopy. By exploring advanced models, this work aims to overcome the limitations of traditional chemometrics, facilitating the development of more robust, global models. These models are designed to perform consistently across diverse conditions, including different seasons, locations, fruit varieties, growing conditions and spectrometers.

The research is encapsulated in the thesis **Deep Learning In Estimation of Fruit Attributes Using Near Infrared Spectroscopy** by Jeremy Walsh (to be finalised). The latest publications associated with this work can be found on the authors [ORCID Profile](https://orcid.org/0000-0002-4360-3536). 


## Usage
This repository is home to a collection of Jupyter notebooks that document the modelling and analysis conducted for this research project. Once the environment is properly setup, users are encouraged to execute these notebooks. This allows for the replication of the study's results, as well as provides an opportunity for in-depth exploration of the underlying data and models. Whether you're aiming to validate the findings or delve into the nuances of the models and data used, these notebooks serve as a comprehensive guide to understanding and leveraging the research conducted.


### Installation and Setup
To get started with this project, follow these steps:
1. Clone this repository.
2. Ensure Python 3.10 is installed and consider creating a virtual environment for this project. For example navigate to the project directory and run:
    ```bash
    python -m venv venv
    ```
3. With the desired environment activated, install the required dependencies:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e ./research/src
    ```
4. Start a Jupyter server to explore the notebooks:
    ```bash
    jupyter notebook
    ```
    This will open a web browser to the URL of the Jupyter notebook web application, where you can access and run the project's notebooks.


## Supplementary Material for Publications

This repository contains specific supplementary material that accompanies the research presented in the following publications:
- [Evaluation of 1D Convolutional Neural Network in Estimation of Mango Dry Matter Content](publications/Eval-1D-CNN-Mango-DM/README.md)
- [Thesis (Chapter 4)](publications/Thesis-Chapter-4/README.md)


## Acknowledgments

Many thanks to the [Non-Invasive Sensor Group](https://www.cqu.edu.au/research/organisations/institute-for-future-farming-systems/non-invasive-sensor-technology) at CQUniversity's Institute for Future Farming Systems for the support and provision of their datasets. Collecting and compiling thousands of spectra and reference values is a huge undertaking over many years, and this research project would not have been possible without their combined effort.

Special thanks to Dário Passos for the inspiration from his open GitHub repository on deep learning for VIS-NIR spectra ([dario-passos/DeepLearning_for_VIS-NIR_Spectra](https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra)).


## Contact Information
For further information, collaboration, or inquiries about the project, please reach out via the following:

**Email**: jeremypaul.walsh@gmail.com

**LinkedIn:** [Jeremy Walsh](https://www.linkedin.com/in/jeremyp-walsh/)
