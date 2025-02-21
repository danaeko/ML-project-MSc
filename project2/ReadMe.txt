Run the script script_code.sh

The script performs the following tasks:

Reads the dataset and provides a description of the corresponding data.

  1. Creates PCA, FA, and KPCA plots.

  2. Tests six different classification models using K-fold cross-validation to select the best hyperparameters for     each model.

  3. Uses the best models from step 3 to run two different algorithms for feature selection.

  4. Evaluates the performance of each model using the optimal hyperparameters and selected features from steps 3 and   4, respectively.
