# Text Classification Project

This project is designed to classify text data using a machine learning pipeline. It involves text preprocessing, feature extraction, and classification.

## Features

- **Text Processing:** Cleans and preprocesses text data.
- **Feature Extraction:** Extracts features from text using TF-IDF for titles and bodies.
- **Classification:** Classifies text using RidgeClassifier.
- **Model Saving:** Saves the trained model using `joblib`.



- **`news_category_pipeline.ipynb`:**  
  A Jupyter notebook that demonstrates the entire pipeline for text classification. It includes:
  - Loading and configuring data from `config.yaml`.
  - Building and training a machine learning pipeline using `Pipeline` and `FeatureUnion` from scikit-learn.
  - Evaluating model performance on a test set.
  - Saving the trained pipeline to a file (`pipeline.joblib`) for later use.
- **`utils.py`:**  
  Contains utility functions and classes used throughout the project. Important components include:
  - **`TextPreprocessor`:** Handles text cleaning and preprocessing tasks.
  - **`ColumnSelector`:** Used to select specific columns from the DataFrame for processing.
  - **`TitleTfidfTransformer`:** Applies TF-IDF transformation to the title column.
  - **`BodyTfidfTransformer`:** Applies TF-IDF transformation to the body column with additional weighting options.

## Installation

To set up the project, follow these steps:

1. **Ensure Python and Pip are Installed:**  
   Make sure Python 3.6 or higher and pip are installed on your machine.

2. **Install Dependencies:**  
   Install the required packages using the provided `requirements.txt` file.

## Usage

1. **Configure the Project:**  
   Edit the `config.yaml` file with your project settings. This file should include paths for data, parameters for TF-IDF, and classifier settings.

2. **Train the Model:**  
   Run the script to process the data, train the model, and save it. Ensure the trained model is saved in the `models/` directory.

3. **Make Predictions:**  
   To use the trained model for making predictions, load the model, prepare new text data for prediction, and then make predictions using the model.

4. **Jupyter Notebooks:**  
   Several Jupyter notebooks are provided for exploring data and models:
   - `notebooks/exploring_data_and_models.ipynb`: Explore and visualize the data.
   - `notebooks/test_via_folktales_dataset.ipynb`: Test the algorithm with another datasets.


5. **Environment Setup:**  
   For setting up a Python environment with the required dependencies, use the provided `environment.yml` file with conda.

6. **Configuration:**  
   Adjust settings in the `config.yaml` file according to your needs.
   
## Data

The data used in this project can be accessed from the following sources:

- **Training Data: News Article Category Dataset**  
  The main dataset used for training the model can be found at [Data Source](https://www.kaggle.com/datasets/timilsinabimal/newsarticlecategories).

- **Folk Stories around the World:**  
  'This dataset is a collection of folktales and fairy tales from diverse cultural backgrounds. Whether you’re interested in ancient legends, whimsical tales, or magical narratives, this dataset offers a treasure trove of storytelling.'
  Used in `notebooks/test_via_folktales_dataset.ipynb`.

  Available at [Folktales Data](https://www.kaggle.com/datasets/chayanonc/1000-folk-stories-around-the-world).


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

Thanks to the open-source community for providing the tools and libraries used in this project.
