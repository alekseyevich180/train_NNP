# ML Demo Scripts

This folder contains two small `numpy`-based machine learning demo scripts:

- [ml_basics_demo.py](/abs/path/c:/Users/yingkaiwu/Desktop/Active-learning/NNP/scripts/others/ml_basics_demo.py)
- [ml_algorithms_demo.py](/abs/path/c:/Users/yingkaiwu/Desktop/Active-learning/NNP/scripts/others/ml_algorithms_demo.py)

They are written as educational examples, not production ML training code.

## 1. `ml_basics_demo.py`

This script introduces core supervised-learning ideas through a from-scratch logistic regression example.

What it does:

- Generates a small toy binary-classification dataset with two features:
  - `reaction_activity`
  - `surface_stability`
- Splits the data into train/test sets
- Standardizes the features
- Trains logistic regression using gradient descent
- Reports train/test loss and accuracy
- Prints a simple interpretation of feature weights
- Prints a short summary of basic ML concepts

Main concepts covered:

- dataset and labels
- train/test split
- feature standardization
- sigmoid function
- binary cross-entropy loss
- gradient descent
- probability prediction and classification
- model interpretation

Run it:

```powershell
python others/ml_basics_demo.py
```

Expected output:

- training progress every 50 epochs
- final train/test loss
- final train/test accuracy
- interpretation of feature weights and bias

## 2. `ml_algorithms_demo.py`

This script demonstrates several standard ML algorithms, each implemented with `numpy`.

Included demos:

- `K-nearest neighbors`
- `K-means clustering`
- `Linear regression`
- `Perceptron`

What each demo shows:

- `knn`: distance-based classification on a synthetic two-class dataset
- `kmeans`: unsupervised clustering and cluster center updates
- `regression`: linear regression trained by gradient descent with MSE reporting
- `perceptron`: iterative linear classification with mistake-driven updates

Run all demos:

```powershell
python others/ml_algorithms_demo.py
```

Run one demo only:

```powershell
python others/ml_algorithms_demo.py --method knn
python others/ml_algorithms_demo.py --method kmeans
python others/ml_algorithms_demo.py --method regression
python others/ml_algorithms_demo.py --method perceptron
```

Expected output:

- sample counts and simple metrics
- training progress for regression/perceptron
- cluster sizes and centers for k-means
- example predictions for KNN

## 3. Differences Between The Two Scripts

`ml_basics_demo.py` is a focused introduction to one supervised-learning workflow from start to finish.

`ml_algorithms_demo.py` is a broader survey script that shows multiple classic algorithms in separate demos.

In short:

- use `ml_basics_demo.py` to explain fundamental ML workflow
- use `ml_algorithms_demo.py` to compare several common algorithms

## 4. Notes

- Both scripts use only `numpy` plus Python standard library modules.
- The datasets are synthetic and generated inside the scripts.
- These demos are intended for learning and quick experimentation.
- No model files are saved.
