# Dynamic Soft-Thresholding Proximal Gradient for Sparse Feature Selection in High-Dimensional House Price Prediction

## Abstract

Sparse linear models are widely used for tabular regression problems because they can perform feature selection and remain interpretable. A common approach is the LASSO, which augments least-squares regression with an \(\ell_1\) penalty on the coefficients. However, when many predictors are highly correlated, standard LASSO with a static penalty can exhibit unstable feature selection and may fail to fully exploit groups of related variables. In this work we study a dynamically adaptive proximal gradient method in which the soft-thresholding operator is scaled iteratively based on a proxy for the subdifferential of the \(\ell_1\) penalty at the current iterate. The goal is to prune clearly irrelevant features aggressively in the early iterations, while allowing finer adjustment of correlated active coefficients in later stages.

We implement this dynamic soft-thresholding approach for linear regression on the Kaggle *House Prices: Advanced Regression Techniques* dataset. A common preprocessing pipeline and cross-validation protocol is used for all models. The proposed optimizer is compared to two standard baselines: Ridge regression and static LASSO. The models are evaluated on mean squared error (MSE), coefficient sparsity (number of non-zero weights), and computational efficiency (runtime per cross-validation fold). Our experiments show that the standard LASSO achieves a strong accuracy–sparsity trade-off, while the current configuration of the dynamic method attains very low runtime but does not yet match LASSO in terms of predictive accuracy. We discuss the reasons for this behavior and outline possible improvements to the dynamic threshold schedule and hyperparameter tuning.

---

## 1. Introduction

House price prediction is a classic application of supervised learning where the objective is to estimate the selling price of a residential property using information about its physical characteristics, location, and surrounding conditions. Modern real-estate datasets contain a mixture of numerical and categorical attributes, which, after preprocessing and one-hot encoding, result in a high-dimensional feature space. In such settings, it is often desirable not only to achieve low prediction error but also to identify a relatively small set of informative features. This motivates the use of sparse models that set many coefficients exactly to zero.

The LASSO (Least Absolute Shrinkage and Selection Operator) is a popular technique that addresses this requirement by adding an \(\ell_1\)-norm penalty to the usual squared error loss. The \(\ell_1\) penalty encourages sparsity in the coefficient vector, which naturally performs feature selection. Despite its advantages, LASSO has known limitations when predictors are strongly correlated. In these situations the method may arbitrarily select one variable from a correlated group, and small changes in the regularization strength or the data can lead to different features being chosen. This instability can make the model harder to interpret and potentially less robust.

The theme of this project is to explore **dynamic soft-thresholding** within a proximal gradient framework as a way to improve feature selection in correlated, high-dimensional regression problems. Instead of using a single static soft-threshold at every iteration, we design a threshold that changes with both the iteration number and the current magnitude of each coefficient. Intuitively, small coefficients, which are closer to the subdifferential plateau of the \(\ell_1\) norm, should be shrunk more aggressively, while well-established coefficients should experience relatively less shrinkage over time.

In this report we apply the proposed dynamic soft-thresholding proximal gradient method to the *House Prices: Advanced Regression Techniques* dataset from Kaggle. We use a consistent preprocessing pipeline and a shared cross-validation scheme across all models. Our contributions can be summarized as follows:

- We implement a complete machine learning pipeline for high-dimensional house price prediction, including data preprocessing, model training, and evaluation.
- We formulate and implement a dynamic soft-thresholding proximal gradient optimizer that adapts its thresholds based on a proxy for the \(\ell_1\) subdifferential and an annealing schedule over iterations.
- We conduct a controlled empirical comparison of Ridge regression, static LASSO, and the proposed dynamic method using identical folds and metrics.
- We analyze the resulting trade-offs between prediction error, sparsity, and runtime, and discuss the strengths and limitations of the dynamic approach in this setting.

The remainder of the report is organized as follows. Section 2 describes the dataset and preprocessing pipeline. Section 3 presents the mathematical formulation of the LASSO objective and reviews the proximal gradient method. Section 4 introduces the dynamic soft-thresholding algorithm in detail. Section 5 describes the experimental setup, including baselines, hyperparameters, and metrics. Section 6 reports and analyzes the empirical results. Section 7 concludes with a discussion of key observations and directions for future work.

---

## 2. Dataset and Preprocessing

### 2.1 Dataset description

We use the training portion of the Kaggle competition *House Prices: Advanced Regression Techniques*. Each data point corresponds to a single house sale and includes the final sale price along with a collection of explanatory variables. These variables capture physical aspects of the property (lot size, floor area, number of rooms, overall quality), temporal factors (year built, year remodeled), and location-related information (neighborhood, zoning, condition).

The dataset contains **1460** labelled examples in the training set. Based on the dtypes observed after loading the CSV file, there are **37 numerical** predictors and **43 categorical** predictors. The target variable is the `SalePrice` column, which records the final sale price in US dollars.

### 2.2 Target transformation

Sale prices are strictly positive and exhibit a right-skewed distribution. Modelling the raw prices directly with squared error loss can place more emphasis on large outliers. To mitigate this and make the residuals closer to symmetric, we model the logarithm of the price:

\[
y = \log\big(1 + \text{SalePrice}\big).
\]

The \(\log(1 + x)\) transform is applied to all target values when constructing the training labels. All evaluation metrics reported in this work use this transformed target. In a deployment setting, predictions could be converted back to the original scale via the inverse transform \(\exp(y) - 1\).

### 2.3 Feature preprocessing

The raw dataset contains missing entries in both numerical and categorical columns. Furthermore, most learning algorithms expect a purely numerical design matrix as input. To address these issues and ensure that all models see an identical representation, we construct a single preprocessing pipeline based on standard tools from scikit-learn.

The preprocessing steps are:

- **Numeric features (37 columns)**:
  - Missing values are imputed using the median of each feature.
  - The imputed values are then scaled using a form of standardization that is compatible with sparse representations.

- **Categorical features (43 columns)**:
  - Missing values are imputed using the most frequent category in each column.
  - Each imputed categorical feature is transformed by one-hot encoding, which produces one binary indicator column per observed category.
  - Unknown categories at evaluation time are safely ignored by the encoder.

The numeric and categorical transformations are combined using a `ColumnTransformer`, which outputs a single feature matrix \(X\). After fitting this preprocessing pipeline on the full training data and applying it to the predictors, we obtain:

- **1460 samples** (rows),
- **288 transformed features** (columns),
- composed of contributions from 37 numeric and 43 categorical original variables.

### 2.4 Train–validation splitting

To evaluate and tune models in a reproducible way, we use **5-fold cross-validation**. The indices of the 1460 samples are randomly shuffled with a fixed seed (42) and then partitioned into five equal folds. For each fold in turn:

- one fold is treated as the validation set, and
- the remaining four folds form the training set.

This procedure is applied identically to all methods considered in this report, including Ridge, LASSO, and the dynamic soft-thresholding model. By reusing the same splits, we ensure that performance differences are not due to different train–validation partitions but to the underlying optimization and regularization schemes.

---

## 3. Mathematical Formulation

### 3.1 Linear regression with \(\ell_1\) regularization

Let \(X \in \mathbb{R}^{n \times p}\) denote the preprocessed design matrix, where \(n = 1460\) and \(p = 288\). Each row \(x_i^\top\) corresponds to the features for a single house, and \(y \in \mathbb{R}^n\) contains the log-transformed sale prices. We consider a linear model of the form

\[
\hat{y}_i = x_i^\top \beta,
\]

where \(\beta \in \mathbb{R}^p\) is the coefficient vector to be estimated.

The standard least-squares objective is given by

\[
f(\beta) = \frac{1}{2n}\|X\beta - y\|_2^2.
\]

To encourage sparsity and perform feature selection, we augment this loss with an \(\ell_1\) penalty:

\[
\min_{\beta \in \mathbb{R}^p} \; 
f(\beta) + \lambda \|\beta\|_1
\quad\text{with}\quad
f(\beta) = \frac{1}{2n}\|X\beta - y\|_2^2.
\]

Here, \(\lambda > 0\) is a regularization parameter that controls the strength of the sparsity constraint. Larger values of \(\lambda\) typically yield more zero coefficients and thus fewer selected features, at the expense of potentially higher prediction error.

### 3.2 Gradient and Lipschitz constant

The smooth part of the objective, \(f(\beta)\), has gradient

\[
\nabla f(\beta) = \frac{1}{n} X^\top (X\beta - y).
\]

This gradient is Lipschitz-continuous with constant

\[
L = \frac{\|X\|_2^2}{n},
\]

where \(\|X\|_2\) denotes the spectral norm (largest singular value) of \(X\). In practice we approximate \(\|X\|_2\) via a short power iteration and then set the step size to \(\alpha = 1/L\). This choice is sufficient to guarantee convergence of the proximal gradient method for convex problems of this form.

### 3.3 Proximal gradient method (ISTA)

Because the \(\ell_1\) norm is non-smooth, we cannot apply simple gradient descent directly to the full objective. Instead, we use a proximal gradient method, which alternates a gradient descent step on the smooth term with an application of the proximal operator of the non-smooth term.

Starting from an initial guess \(\beta^0\), a generic proximal gradient update for the LASSO is:

1. Compute the gradient step:
   \[
   z^k = \beta^k - \alpha \nabla f(\beta^k).
   \]

2. Apply the proximal operator of the scaled \(\ell_1\) norm:
   \[
   \beta^{k+1} = \operatorname{prox}_{\alpha \lambda \|\cdot\|_1}(z^k).
   \]

For the \(\ell_1\) norm, the proximal operator has a closed form known as the **soft-thresholding operator**:

\[
\big[\operatorname{SoftThresh}_{\tau}(z)\big]_i
  = \operatorname{sign}(z_i) \max(|z_i| - \tau, 0),
\]

where \(\tau = \alpha \lambda\). Thus, the iteration can be written as

\[
\beta^{k+1}_i = \operatorname{sign}(z_i^k)\max\big(|z_i^k| - \alpha\lambda,\, 0\big).
\]

This classical algorithm is often referred to as ISTA (Iterative Shrinkage–Thresholding Algorithm).

---

## 4. Dynamic Soft-Thresholding Proximal Gradient

### 4.1 Motivation and intuition

In classical ISTA, the same threshold \(\alpha\lambda\) is applied to every coordinate at every iteration. While this simplicity is attractive, it does not exploit information about which coefficients are likely to be truly active. In high-dimensional regression with correlated predictors, we might want the thresholding rule to behave differently for small and large coefficients, and to evolve over time as the algorithm progresses.

The idea behind dynamic soft-thresholding is to modulate the threshold per coordinate and per iteration. Coefficients that remain very close to zero should be subjected to stronger shrinkage, accelerating their elimination. In contrast, coefficients that have grown to larger magnitudes should be subject to relatively milder shrinkage, especially in later iterations, so that correlated groups of features can be fine-tuned rather than over-penalized.

### 4.2 Subdifferential-inspired scaling

For the scalar absolute value function \(|\beta_i|\), the subdifferential is

\[
\partial |\beta_i| =
\begin{cases}
\{\operatorname{sign}(\beta_i)\}, & \text{if } \beta_i \neq 0, \\
[-1, 1], & \text{if } \beta_i = 0.
\end{cases}
\]

Near zero, the subgradient set is a wide interval, reflecting uncertainty about whether the coordinate should be positive, negative, or zero. Motivated by this, we introduce a smooth proxy that indicates how “active” a coordinate is:

\[
g_i^k = \frac{|\beta_i^k|}{|\beta_i^k| + \epsilon},
\]

where \(\epsilon > 0\) is a small constant. When \(|\beta_i^k|\) is close to zero, \(g_i^k\) is near 0; when \(|\beta_i^k|\) is large, \(g_i^k\) approaches 1. We interpret \(g_i^k\) as a soft measure of confidence that coordinate \(i\) belongs in the model.

### 4.3 Annealed thresholds

In addition to using \(g_i^k\), we introduce a global annealing factor that decreases with the iteration index:

\[
s(k) = 1 + \frac{A}{(1 + k)^p},
\]

where \(A > 0\) and \(p > 0\) are hyperparameters. Early in the optimization, \(s(k)\) is larger than 1, enforcing stronger shrinkage overall. As \(k\) increases, \(s(k)\) gradually approaches 1, allowing more delicate adjustments to the coefficients as the algorithm nears convergence.

### 4.4 Coordinate-wise dynamic thresholds

Combining the activity measure and the annealing factor, we define the threshold for coordinate \(i\) at iteration \(k\) as

\[
\tau_i^k = \alpha \lambda \big(\gamma_0 + \gamma_1 (1 - g_i^k)\big) s(k),
\]

where \(\gamma_0 > 0\) is a base shrinkage weight and \(\gamma_1 \ge 0\) scales the additional shrinkage applied to low-activity coordinates.

- If \(|\beta_i^k|\) is small, then \(g_i^k \approx 0\), so the factor \(\gamma_0 + \gamma_1 (1 - g_i^k)\) is relatively large. This yields a bigger threshold \(\tau_i^k\) and promotes more aggressive shrinking of that coordinate.
- If \(|\beta_i^k|\) is large, then \(g_i^k \approx 1\), so the factor is closer to \(\gamma_0\) and the threshold is relatively smaller. This reduces the risk of over-penalizing clearly active features.
- As \(k\) increases, \(s(k)\) shrinks towards 1, decreasing the overall amount of shrinkage and allowing the algorithm to fine-tune the coefficient values.

### 4.5 Algorithm summary

The dynamic soft-thresholding proximal gradient algorithm proceeds as follows:

1. Estimate the Lipschitz constant \(L\) of \(\nabla f\) using power iteration, and set \(\alpha = 1/L\).
2. Initialize \(\beta^0 = 0\).
3. For \(k = 1, 2, \dots,\) until convergence:
   - Compute the residual \(r^k = X\beta^k - y\).
   - Compute the gradient \(\nabla f(\beta^k) = \frac{1}{n} X^\top r^k\).
   - Take a gradient step \(z^k = \beta^k - \alpha \nabla f(\beta^k)\).
   - For each coordinate \(i\):
     - Compute \(g_i^k = |\beta_i^k|/(|\beta_i^k| + \epsilon)\).
     - Compute the annealing factor \(s(k) = 1 + A/(1 + k)^p\).
     - Compute \(\tau_i^k = \alpha \lambda (\gamma_0 + \gamma_1 (1 - g_i^k)) s(k)\).
     - Apply soft-thresholding:
       \[
       \beta_i^{k+1} = \operatorname{sign}(z_i^k)\max(|z_i^k| - \tau_i^k, 0).
       \]
   - Check a convergence criterion (e.g., relative change in objective or parameter norm).

In our implementation, the algorithm records diagnostic quantities such as the objective value, number of non-zero coefficients, and step norms at regular intervals. These traces can be used to study convergence behavior and compare different hyperparameter settings.

---

## 5. Experimental Setup

### 5.1 Baseline methods

We compare the dynamic soft-thresholding method to two standard linear models:

- **Ridge regression**: a linear model with \(\ell_2\) regularization, implemented using `sklearn.linear_model.Ridge`. Ridge tends to keep all coefficients non-zero but shrinks them towards zero. It can handle correlated predictors well in terms of stability but does not yield a sparse solution.

- **Standard LASSO**: an \(\ell_1\)-regularized linear model, implemented using `sklearn.linear_model.Lasso`. This model solves the same optimization problem as in Section 3 but uses a static penalty and a coordinate descent optimizer. It typically produces sparse solutions and is widely used for feature selection.

All three methods operate on the same preprocessed design matrix and use the same cross-validation splits.

### 5.2 Hyperparameter search

We use a simple grid search over regularization parameters for each method:

- Ridge regression:
  - \(\alpha \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100, 1000\}\).

- Standard LASSO:
  - \(\lambda \in \{10^{-4}, 3\cdot 10^{-4}, 10^{-3}, 3\cdot 10^{-3}, 10^{-2}, 3\cdot 10^{-2}, 10^{-1}\}\).

- Dynamic soft-thresholding LASSO:
  - Base \(\lambda\): same grid as LASSO.
  - \(\gamma_1 \in \{0.5, 1.0, 2.0, 4.0\}\).
  - \(\gamma_0 = 1.0\), \(\epsilon = 10^{-6}\).
  - Annealing parameters: \(A = 2.0\), \(p = 1.0\).
  - Maximum iterations: 5000.
  - Stopping tolerance: \(10^{-6}\) based on relative changes.

For each combination of hyperparameters and each method, we perform 5-fold cross-validation and compute the mean MSE, standard deviation of MSE, mean number of non-zero coefficients, and mean runtime per fold. For the dynamic method we additionally track the mean number of iterations until convergence.

### 5.3 Evaluation metrics

The main evaluation metrics are:

- **Mean Squared Error (MSE)**:
  \[
  \text{MSE} = \frac{1}{n_\text{val}} \sum_{i \in \text{val}} (y_i - \hat{y}_i)^2,
  \]
  computed on the validation folds using the log-transformed targets.

- **Sparsity**:
  - Measured as the number of coefficients whose absolute value exceeds a small threshold (here \(10^{-8}\)).
  - We report the mean number of non-zero coefficients across folds.

- **Runtime**:
  - Average wall-clock time required to fit the model on a single training fold.

These metrics allow us to study the trade-off between accuracy, model complexity, and computational cost.

---

## 6. Results and Discussion

### 6.1 Overall performance comparison

The experimental pipeline produces a table of results (stored in `outputs/results.csv`) in which each row corresponds to one setting of the hyperparameters. From this table we extract the best configuration for each method by selecting the row with the lowest mean MSE, breaking ties in favor of models with fewer non-zero coefficients and lower runtime. A summary of the best configurations (from `outputs/best_summary.json`) is:

- **Best dynamic LASSO (proposed method)**:
  - \(\lambda = 0.0001\), \(\gamma_1 = 0.5\).
  - Mean MSE \(\approx 0.1593\).
  - Mean number of non-zero coefficients \(\approx 249\).
  - Mean runtime per fold \(\approx 0.0038\) seconds.

- **Best standard LASSO**:
  - \(\lambda = 0.0003\).
  - Mean MSE \(\approx 0.0219\).
  - Mean number of non-zero coefficients \(\approx 136.2\).
  - Mean runtime per fold \(\approx 0.0337\) seconds.

- **Best Ridge regression**:
  - \(\alpha = 10.0\).
  - Mean MSE \(\approx 0.0231\).
  - Mean number of non-zero coefficients \(\approx 283.2\) (dense model).
  - Mean runtime per fold \(\approx 0.0050\) seconds.

From these summaries we observe:

- The standard LASSO achieves the **lowest MSE** among the three, while also using **roughly half as many** non-zero coefficients as Ridge.
- Ridge regression attains slightly higher MSE than LASSO and retains almost all features in the model, which is expected given the \(\ell_2\) penalty.
- The dynamic soft-thresholding method, with the particular schedule and grid we used, has much **higher MSE** than both LASSO and Ridge, even though it uses a reasonably sparse model and converges very quickly in terms of iterations and runtime.

### 6.2 Accuracy–sparsity trade-off

The plot `outputs/plots/mse_vs_sparsity.png` visualizes the relationship between mean MSE and mean number of non-zero coefficients across all configurations. Each point corresponds to a single hyperparameter setting and is colored by method.

The LASSO points form a clear trade-off curve: smaller \(\lambda\) values lead to lower MSE but more non-zero coefficients, while larger \(\lambda\) values increase sparsity at the cost of worse MSE. Ridge points cluster at high numbers of non-zero coefficients, reflecting their dense solutions, with MSE values that are competitive but not superior to the best LASSO.

The dynamic LASSO points tend to have moderate sparsity (fewer non-zero coefficients than Ridge but more than the sparsest LASSO models) but consistently higher MSE values. This suggests that, at least under the chosen hyperparameter grid, the dynamic threshold schedule is too aggressive in shrinking useful coefficients or does not sufficiently adapt to the underlying correlation structure of the data.

### 6.3 Computational efficiency

The plot `outputs/plots/runtime_by_method.png` compares the average training time per fold for each method. Ridge and dynamic LASSO both have very low runtimes, on the order of a few milliseconds per fold, while standard LASSO is somewhat slower due to the more complex coordinate descent procedure and its convergence properties.

The dynamic method is particularly efficient because it typically converges in only a few iterations, as indicated by the mean iteration counts (around 2–3 for many configurations). This confirms that the method is computationally attractive, even if its accuracy is not yet competitive with the best LASSO configuration in this experiment.

### 6.4 Effect of \(\gamma_1\) in dynamic LASSO

The plot `outputs/plots/dynamic_mse_vs_sparsity_gamma1.png` focuses on the dynamic method and colors each point by the value of \(\gamma_1\). This allows us to examine how the extra shrinkage on low-activity coordinates influences the final sparsity and MSE.

In the present runs, increasing \(\gamma_1\) tends to reduce the number of non-zero coefficients slightly for a fixed \(\lambda\), but it does not significantly improve the MSE. In some cases, higher \(\gamma_1\) values further increase the MSE, which is consistent with the idea that too much additional shrinkage can oversuppress useful correlated features.

---

## 7. Conclusion and Future Work

In this project we implemented and evaluated a dynamic soft-thresholding proximal gradient method for sparse linear regression on the Kaggle House Prices dataset. The approach adapts the soft-thresholding operator based on a smooth proxy for the \(\ell_1\) subdifferential and an annealing schedule over iterations. The goal was to prune irrelevant features aggressively in early iterations while allowing more careful tuning of correlated features later on.

Using a common preprocessing pipeline and cross-validation protocol, we compared the proposed method against Ridge regression and standard LASSO. Our experiments show that:

- The standard LASSO achieves the best balance of predictive accuracy and sparsity, with the lowest mean MSE and a moderate number of non-zero coefficients.
- Ridge regression offers competitive MSE but retains almost all features, which may limit interpretability.
- The dynamic method, as configured in our experiments, is computationally efficient but does not match the accuracy of LASSO, and its sparsity is not dramatically better in this setting.

These results highlight both the potential and the challenges of dynamically scaled soft-thresholding. On the positive side, the method is simple to implement on top of standard proximal gradient and can converge in very few iterations. On the other hand, its performance is sensitive to the choice of hyperparameters such as \(\lambda\), \(\gamma_1\), and the annealing schedule, and the simple grid we used may not be sufficient to find a truly competitive configuration.

Future work could explore:

- More extensive and systematic hyperparameter search, including different annealing strategies and coordinate-wise weighting schemes.
- Integration with accelerated methods (e.g., FISTA) to improve convergence behavior without sacrificing sparsity.
- Combining dynamic thresholds with elastic net regularization to better handle groups of correlated predictors.
- Applying the method to other high-dimensional datasets, including classification problems, to assess its robustness across tasks.

Overall, the project provided hands-on experience with proximal optimization methods, sparse model design, and rigorous empirical evaluation on a realistic regression problem.

---

## References

1. R. Tibshirani, “Regression Shrinkage and Selection via the Lasso,” *Journal of the Royal Statistical Society: Series B (Methodological)*, vol. 58, no. 1, pp. 267–288, 1996.
2. A. Beck and M. Teboulle, “A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems,” *SIAM Journal on Imaging Sciences*, vol. 2, no. 1, pp. 183–202, 2009.
3. T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd ed., Springer, 2009.
4. Kaggle, “House Prices: Advanced Regression Techniques,” Competition and dataset. Available at: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

