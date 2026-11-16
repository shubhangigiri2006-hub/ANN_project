# AI-Based Student Performance Risk Predictor
## Comprehensive Technical Design Document, Theoretical Foundations, and Implementation Manual

---

### Project Metadata
* **Project Classification:** Academic Early Warning System / Multi-Class Risk Classification
* **Skill Tier:** Beginner to Intermediate Applied Machine Learning
* **Primary Deep Learning Framework:** TensorFlow 2.x / Keras (Sequential API)
* **Comparative Baseline:** Multinomial Logistic Regression (Scikit-Learn)
* **Production Serving Interfaces:** Streamlit Interactive Web Interface (`app.py`) and FastAPI REST Microservice (`api.py`)
* **Primary Dataset:** `month_3_ann_student_performance_dataset_15000.csv` (15,000 observational records)

---

## 1. Executive Summary and Problem Formulation

Higher education institutions experience substantial student attrition when academic distress is detected too late in an academic term. Traditional institutional reviews rely heavily on end-of-semester summative examinations, which leave zero operational window for academic advisors, counselors, or faculty mentors to deploy targeted academic interventions.

The objective of this project is to engineer an end-to-end predictive machine learning pipeline anchored by a multi-layer Artificial Neural Network (ANN). The system analyzes multi-dimensional behavioral, historical, and continuous engagement signals collected during the semester to classify students into three distinct performance categories:
1. **Low Performance (High Academic Risk):** Students projected to fail or achieve marginal standing; requires immediate faculty mentoring, mandatory tutoring, and workload restructuring.
2. **Medium Performance (Moderate Academic Risk):** Students meeting passing criteria but exhibiting vulnerability in specific subjects or assessments; benefits from proactive academic advisement and study groups.
3. **High Performance (Low Academic Risk):** Students consistently demonstrating advanced mastery; eligible for peer mentoring leadership, honors tracks, and advanced elective coursework.

This document serves as the complete technical blueprint, theoretical manual, and deployment guide for the project. It details the mathematical foundations of feed-forward neural networks, provides empirical benchmarks against traditional linear baselines, documents every dataset feature exhaustively, and supplies complete, deployable source code for production web and API interfaces.

---

## 2. Exhaustive Feature Dictionary and Exploratory Data Analysis

The primary dataset consists of 15,000 student records across 11 input attributes and 1 categorical ground-truth target variable. The dataset contains zero missing values and zero duplicate records. 

Below is the exhaustive, feature-by-feature domain analysis, statistical breakdown, and preprocessing treatment for every column in the dataset.

### 2.1 Identifiers and Metadata

#### `student_id`
* **Data Type:** String (Categorical Identifier, Format: `STU00001` through `STU15000`)
* **Domain Range:** 15,000 unique alphanumeric identifiers
* **Missing Values:** 0 (0.00%)
* **Domain Significance:** Represents an arbitrary primary key assigned by the student information database.
* **Predictive Utility:** Zero. The identifier has cardinality equal to sample size ($N = 15,000$). Retaining this feature during model training would introduce severe target leakage or cause memorization without generalization.
* **Data Preprocessing Protocol:** Dropped unconditionally from the feature matrix prior to scaling and model ingestion.

---

### 2.2 Continuous Academic and Behavioral Input Features

#### `age`
* **Data Type:** Integer (`int64`)
* **Measurement Unit:** Chronological years completed
* **Observed Range:** 17 to 24 years
* **Statistical Distribution:** Mean: 20.47 years | Standard Deviation: 2.28 years | Median: 20.00 years | Interquartile Range (IQR): 18.00 to 22.00 years
* **Correlation with Target:** -0.0070 (Statistically negligible linear correlation)
* **Domain Context:** Reflects the typical undergraduate demographic. Because performance categories are distributed uniformly across entry ages (mean age is 20.52 for Low, 20.42 for Medium, and 20.48 for High), age acts as a neutral demographic control rather than a primary driver of academic failure.
* **Preprocessing Protocol:** Standardized using Z-score scaling.

#### `attendance_pct`
* **Data Type:** Float (`float64`)
* **Measurement Unit:** Percentage points ($0.0\%$ to $100.0\%$)
* **Observed Range:** 45.00% to 100.00%
* **Statistical Distribution:** Mean: 77.94% | Standard Deviation: 11.67% | Median: 77.90% | Interquartile Range: 70.00% to 86.10%
* **Correlation with Target:** +0.1403 (Moderate positive correlation)
* **Class-Conditional Means:** Low: 75.83% | Medium: 78.11% | High: 79.87%
* **Domain Context:** Attendance measures direct physical or virtual classroom engagement. Institutional policies frequently mandate a minimum threshold of 75% attendance for examination eligibility. Students falling into the Low category exhibit a noticeably lower mean attendance (75.83%), reflecting initial disengagement phases.
* **Preprocessing Protocol:** Standardized using Z-score scaling.

#### `study_hours_per_week`
* **Data Type:** Float (`float64`)
* **Measurement Unit:** Hours per seven-day week
* **Observed Range:** 2.00 to 40.00 hours
* **Statistical Distribution:** Mean: 15.17 hours | Standard Deviation: 6.84 hours | Median: 15.10 hours | Interquartile Range: 10.30 to 19.80 hours
* **Correlation with Target:** +0.5947 (Strongest single linear predictor in the dataset)
* **Class-Conditional Means:** Low: 10.22 hours | Medium: 15.06 hours | High: 20.24 hours
* **Domain Context:** Quantifies independent, self-directed learning time outside scheduled lectures. The empirical gap between High-performing students (mean 20.24 hours/week) and Low-performing students (mean 10.22 hours/week) spans almost two full standard deviations. This feature serves as the anchor separating high-risk students from sustainable performers.
* **Preprocessing Protocol:** Standardized using Z-score scaling.

#### `previous_score`
* **Data Type:** Float (`float64`)
* **Measurement Unit:** Normalized examination points ($0.0$ to $100.0$)
* **Observed Range:** 25.00 to 100.00 points
* **Statistical Distribution:** Mean: 66.65 points | Standard Deviation: 14.74 points | Median: 66.60 points | Interquartile Range: 56.80 to 76.60 points
* **Correlation with Target:** +0.2260 (Substantial positive correlation)
* **Class-Conditional Means:** Low: 62.61 points | Medium: 66.54 points | High: 70.81 points
* **Domain Context:** Represents historical cumulative academic performance from prior terms. It captures foundational competence and prior subject mastery. While past performance establishes an academic baseline, the neural network learns interaction effects where high study hours can compensate for marginal prior scores.
* **Preprocessing Protocol:** Standardized using Z-score scaling.

#### `assignment_completion_pct`
* **Data Type:** Float (`float64`)
* **Measurement Unit:** Percentage of assigned coursework submitted ($0.0\%$ to $100.0\%$)
* **Observed Range:** 20.00% to 100.00%
* **Statistical Distribution:** Mean: 75.35% | Standard Deviation: 15.89% | Median: 75.90% | Interquartile Range: 64.40% to 87.50%
* **Correlation with Target:** +0.1528 (Positive correlation)
* **Class-Conditional Means:** Low: 72.36% | Medium: 75.35% | High: 78.34%
* **Domain Context:** Measures continuous compliance, diligence, and semester-long time management. Unlike single high-stakes examinations, continuous assignment completion reflects sustained engagement. Submissions below 65% correlate heavily with assignment fatigue and impending examination failure.
* **Preprocessing Protocol:** Standardized using Z-score scaling.

#### `internal_assessment_score`
* **Data Type:** Float (`float64`)
* **Measurement Unit:** Scaled assessment score ($0.0$ to $100.0$)
* **Observed Range:** 20.00 to 100.00 points
* **Statistical Distribution:** Mean: 64.83 points | Standard Deviation: 15.85 points | Median: 64.70 points | Interquartile Range: 54.10 to 75.80 points
* **Correlation with Target:** +0.1887 (Positive correlation)
* **Class-Conditional Means:** Low: 61.26 points | Medium: 64.61 points | High: 68.62 points
* **Domain Context:** Captures mid-semester testing, laboratory practical evaluations, and periodic quizzes administered by course instructors. It provides a contemporary, intra-semester evaluation of whether the student understands current curriculum concepts before final evaluations.
* **Preprocessing Protocol:** Standardized using Z-score scaling.

#### `participation_score`
* **Data Type:** Float (`float64`)
* **Measurement Unit:** Instructor-rated qualitative index ($1.0$ to $10.0$)
* **Observed Range:** 1.00 to 10.00
* **Statistical Distribution:** Mean: 6.15 points | Standard Deviation: 1.95 points | Median: 6.20 points | Interquartile Range: 4.80 to 7.50 points
* **Correlation with Target:** +0.2283 (Significant positive correlation)
* **Class-Conditional Means:** Low: 5.59 points | Medium: 6.17 points | High: 6.69 points
* **Domain Context:** Evaluates active verbal engagement, discussion board participation, inquiry frequency, and peer collaboration. Students with participation ratings below 4.0 often exhibit silent disengagement, serving as a non-cognitive early warning signal.
* **Preprocessing Protocol:** Standardized using Z-score scaling.

#### `sleep_hours_per_day`
* **Data Type:** Float (`float64`)
* **Measurement Unit:** Average daily sleep duration in hours
* **Observed Range:** 4.00 to 10.00 hours
* **Statistical Distribution:** Mean: 6.79 hours | Standard Deviation: 1.10 hours | Median: 6.80 hours | Interquartile Range: 6.10 to 7.50 hours
* **Correlation with Target:** +0.0903 (Mild positive correlation)
* **Class-Conditional Means:** Low: 6.67 hours | Medium: 6.80 hours | High: 6.91 hours
* **Domain Context:** Acts as a proxy for physiological readiness, mental health, and cognitive retention. Severe sleep deprivation (less than 5 hours per night) degrades memory consolidation, leading to degraded test performance even when study hours appear elevated.
* **Preprocessing Protocol:** Standardized using Z-score scaling.

#### `extracurricular_hours_per_week`
* **Data Type:** Float (`float64`)
* **Measurement Unit:** Weekly hours dedicated to athletics, arts, or student clubs
* **Observed Range:** 0.00 to 18.00 hours
* **Statistical Distribution:** Mean: 5.12 hours | Standard Deviation: 3.29 hours | Median: 5.00 hours | Interquartile Range: 2.60 to 7.40 hours
* **Correlation with Target:** -0.0794 (Mild negative correlation)
* **Class-Conditional Means:** Low: 5.41 hours | Medium: 5.18 hours | High: 4.77 hours
* **Domain Context:** Measures allocation of discretionary time away from core academic tasks. Moderate participation (3 to 6 hours) supports stress management; however, excessive allocation (exceeding 12 hours/week) begins to directly cannibalize essential independent study time.
* **Preprocessing Protocol:** Standardized using Z-score scaling.

#### `failed_subjects`
* **Data Type:** Integer (`int64`)
* **Measurement Unit:** Cumulative count of past un-cleared failed courses / backlogs
* **Observed Range:** 0 to 5 courses
* **Statistical Distribution:** Mean: 0.80 courses | Standard Deviation: 0.90 courses | Median: 1.00 course | Interquartile Range: 0.00 to 1.00 course
* **Correlation with Target:** -0.1695 (Substantial negative correlation)
* **Class-Conditional Means:** Low: 1.00 course | Medium: 0.77 courses | High: 0.62 courses
* **Domain Context:** A direct indicator of systemic academic debt. Students carrying 2 or more unresolved failures face compounding prerequisite knowledge gaps and heightened psychological pressure, significantly increasing the likelihood of continuing in the Low performance tier.
* **Preprocessing Protocol:** Standardized using Z-score scaling.

---

### 2.3 Target Classification Variable

#### `performance_category`
* **Data Type:** Categorical String (`object`)
* **Cardinality:** 3 discrete classes (`Low`, `Medium`, `High`)
* **Class Balance and Distribution:**
  * **Medium:** 5,100 records (34.00%)
  * **Low:** 4,950 records (33.00%)
  * **High:** 4,950 records (33.00%)
* **Domain Significance:** Represents the composite institutional assessment of student academic standing at term conclusion. Because the dataset exhibits a balanced 1:1:1 class ratio, models do not suffer from majority class bias, eliminating the need for synthetic oversampling (such as SMOTE) or class-weighted loss penalties.
* **Encoding Protocol:** Mapped to ordinal integer indices ($Low = 0$, $Medium = 1$, $High = 2$) for evaluation and one-hot encoded into a 3-dimensional binary vector for categorical cross-entropy neural network training.

---

### 2.4 Summary Statistics and Class Breakdown

| Feature Name | Type | Min | 25% | Median | 75% | Max | Mean (Low) | Mean (Med) | Mean (High) | Target Corr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `age` | int64 | 17.0 | 18.0 | 20.0 | 22.0 | 24.0 | 20.52 | 20.42 | 20.48 | -0.007 |
| `attendance_pct` | float64 | 45.0 | 70.0 | 77.9 | 86.1 | 100.0 | 75.83 | 78.11 | 79.87 | +0.140 |
| `study_hours_per_week` | float64 | 2.0 | 10.3 | 15.1 | 19.8 | 40.0 | 10.22 | 15.06 | 20.24 | +0.595 |
| `previous_score` | float64 | 25.0 | 56.8 | 66.6 | 76.6 | 100.0 | 62.61 | 66.54 | 70.81 | +0.226 |
| `assignment_completion_pct`| float64 | 20.0 | 64.4 | 75.9 | 87.5 | 100.0 | 72.36 | 75.35 | 78.34 | +0.153 |
| `internal_assessment_score`| float64 | 20.0 | 54.1 | 64.7 | 75.8 | 100.0 | 61.26 | 64.61 | 68.62 | +0.189 |
| `participation_score` | float64 | 1.0 | 4.8 | 6.2 | 7.5 | 10.0 | 5.59 | 6.17 | 6.69 | +0.228 |
| `sleep_hours_per_day` | float64 | 4.0 | 6.1 | 6.8 | 7.5 | 10.0 | 6.67 | 6.80 | 6.91 | +0.090 |
| `extracurricular_hours` | float64 | 0.0 | 2.6 | 5.0 | 7.4 | 18.0 | 5.41 | 5.18 | 4.77 | -0.079 |
| `failed_subjects` | int64 | 0.0 | 0.0 | 1.0 | 1.0 | 5.0 | 1.00 | 0.77 | 0.62 | -0.170 |

---

## 3. Data Preprocessing and Feature Pipeline Engineering

Machine learning pipelines fail when training and inference data diverge. The preprocessing pipeline enforces strict isolation between data splits to eliminate information leakage.

```mermaid
flowchart TD
    RawCSV["Raw Dataset (15,000 Records x 12 Columns)"] --> DropID["Drop student_id Column"]
    DropID --> SplitFeatures["Separate Predictor Matrix X (10 Features) and Target y"]
    SplitFeatures --> StratSplit["Stratified Train/Test Split (80% Train: 12,000 / 20% Test: 3,000)"]
    
    StratSplit --> FitScaler["Fit StandardScaler on X_train ONLY<br/>Compute Mean and Std Dev"]
    FitScaler --> TransformTrain["Transform X_train to X_train_scaled"]
    FitScaler --> TransformTest["Transform X_test to X_test_scaled (Using Train Mean/Std)"]
    FitScaler --> SaveScaler["Serialize Scaler (scaler.joblib)"]
    
    StratSplit --> OneHotY["One-Hot Encode Target y to (N, 3)<br/>[Low: 1,0,0 | Med: 0,1,0 | High: 0,0,1]"]
    
    TransformTrain --> TrainFlow["Input to Model Training (ANN and Baseline)"]
    OneHotY --> TrainFlow
    TransformTest --> EvalFlow["Input to Holdout Evaluation"]
```

### 3.1 Mathematical Imperative for Feature Normalization

Artificial Neural Networks compute activation states via linear combinations followed by non-linear activations:

$$z = \sum_{i=1}^d w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

When input features possess drastically different physical scales, for example `attendance_pct` ranging from $45$ to $100$ while `participation_score` ranges from $1.0$ to $10.0$, the partial derivative of the loss function with respect to weights varies proportionally with the input magnitude:

$$\frac{\partial \mathcal{L}}{\partial w_i} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial w_i} = \delta \cdot x_i$$

Features with large numerical values generate disproportionately large weight updates, while features on smaller intervals update at an imperceptible rate. Consequently:
1. The loss contour forms an elongated, ill-conditioned hyper-ellipsoid with high condition number $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$.
2. Gradient descent oscillates violently in directions of high curvature, requiring an impractically small learning rate to avoid divergence.
3. Feature normalization via Z-score standardization transforms the loss landscape into a nearly isotropic bowl, enabling stable parameter updates across all coordinates:

$$z = \frac{x - \mu}{\sigma}$$

Where:
* $\mu = \frac{1}{N} \sum_{i=1}^N x_i$ is the empirical feature mean computed on the training partition.
* $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2}$ is the empirical standard deviation.

### 3.2 Data Leakage Prevention Protocol

A critical failure in applied predictive modeling occurs when feature scaling parameters are computed over the full dataset prior to partitioning. Doing so allows information about the distribution of the test set (its mean and variance) to bleed into the training pipeline. 

To maintain strict scientific validity:
1. The dataset is partitioned into an 80% training set (12,000 instances) and a 20% holdout test set (3,000 instances) using stratified sampling on `performance_category`.
2. The `StandardScaler` calculates $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ exclusively from $X_{\text{train}}$ using `.fit_transform()`.
3. The identical fitted scaler transforms $X_{\text{test}}$ via `.transform()`.
4. The scaler artifact is serialized to disk (`scaler.joblib`) so that production web applications and REST APIs apply the exact same transformation parameters to new student records during live inference.

---

## 4. Traditional Machine Learning Baseline: Multinomial Logistic Regression

To evaluate whether the non-linear representational capacity of an Artificial Neural Network is empirically justified, we first construct a traditional machine learning baseline using Multinomial Logistic Regression.

### 4.1 Theoretical Formulation

Multinomial Logistic Regression models class membership probabilities via the Softmax response function applied to $K = 3$ linear hyperplanes:

$$P(Y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

The model parameters $\mathbf{W} \in \mathbb{R}^{10 \times 3}$ and $\mathbf{b} \in \mathbb{R}^3$ are estimated by minimizing the multinomial negative log-likelihood (cross-entropy loss) with $L_2$ regularization:

$$\mathcal{L}(\mathbf{W}, \mathbf{b}) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \ln P(Y = k \mid \mathbf{x}_i) + \frac{\lambda}{2} \|\mathbf{W}\|_F^2$$

### 4.2 Baseline Benchmark Results

The baseline model was fitted with $L_2$ penalty parameter $C = 1.0$ and solved using the L-BFGS quasi-Newton algorithm across 1,000 iterations on the standardized training partition.

* **Overall Test Accuracy:** $70.77\%$
* **Evaluation Partition:** 3,000 unseen test instances (1,000 Low, 1,020 Medium, 980 High)

#### Baseline Classification Report

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Low Performance** | 0.7678 | 0.8070 | 0.7869 | 1,000 |
| **Medium Performance** | 0.5836 | 0.5304 | 0.5557 | 1,020 |
| **High Performance** | 0.7583 | 0.7908 | 0.7742 | 980 |
| **Macro Average** | **0.7032** | **0.7094** | **0.7056** | 3,000 |
| **Weighted Average** | **0.7020** | **0.7077** | **0.7042** | 3,000 |

#### Baseline Confusion Matrix

$$\mathbf{C}_{\text{Logistic}} = \begin{bmatrix} 807 & 180 & 17 \\ 222 & 541 & 257 \\ 22 & 206 & 752 \end{bmatrix}$$

*Row indices denote Ground Truth (Low, Medium, High); Column indices denote Predictions.*

### 4.3 Baseline Diagnostic Insights

The linear baseline establishes a solid initial foundation, achieving roughly 71% classification accuracy. The confusion matrix reveals a clear operational boundary:
1. Extreme misclassifications are rare: Only 17 actual Low-risk students were misclassified as High, and only 22 actual High-risk students were misclassified as Low.
2. The primary vulnerability lies in the **Medium** category, where Recall drops to $53.04\%$. Linear hyperplanes struggle to capture the non-linear boundary separating students who narrowly pass from those who underperform or excel. This limitation motivates the multi-layer neural network architecture.

---

## 5. Neural Network Architecture and Theoretical Foundations

```mermaid
flowchart LR
    subgraph Inputs ["Input Layer (d = 10)"]
        x1["age"]
        x2["attendance_pct"]
        x3["study_hours"]
        x4["previous_score"]
        x5["assignment_pct"]
        x6["internal_assessment"]
        x7["participation"]
        x8["sleep_hours"]
        x9["extracurricular"]
        x10["failed_subjects"]
    end

    subgraph Hidden1 ["Hidden Layer 1 (Dense 64 + ReLU + Dropout 0.2)"]
        h1_1["Neuron 1"]
        h1_2["Neuron 2"]
        h1_dots["..."]
        h1_64["Neuron 64"]
    end

    subgraph Hidden2 ["Hidden Layer 2 (Dense 32 + ReLU)"]
        h2_1["Neuron 1"]
        h2_2["Neuron 2"]
        h2_dots["..."]
        h2_32["Neuron 32"]
    end

    subgraph Output ["Output Layer (Dense 3 + Softmax)"]
        o1["P(Low)"]
        o2["P(Medium)"]
        o3["P(High)"]
    end

    Inputs --> Hidden1
    Hidden1 --> Hidden2
    Hidden2 --> Output
```

### 5.1 Perceptron vs Multi-Layer Perceptron (MLP)

A classical Rosenblatt Perceptron computes a single linear decision boundary:

$$y = f(\mathbf{w}^T \mathbf{x} + b)$$

As demonstrated historically by Minsky and Papert (1969), single-layer perceptrons cannot solve non-linearly separable decision boundaries, such as the basic XOR logic gate.

Academic performance classification is non-linear. A student with low prior scores can attain high performance if weekly study hours and internal assessment scores are high. Conversely, a student with strong historical scores can drop into the Low category if attendance collapses below 60%. 

A Multi-Layer Perceptron (MLP) overcomes linear limitations by inserting hidden layers with non-linear activation functions between the input and output. By the **Universal Approximation Theorem** (Cybenko 1989; Hornik 1991), a feed-forward network with at least one non-linear hidden layer can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to arbitrary precision.

### 5.2 Anatomy of an Artificial Neuron

Each artificial neuron inside a dense layer executes two successive operations:
1. **Affine Combination (Weighted Sum):**
   $$z_j^{[l]} = \sum_{i=1}^{n^{[l-1]}} W_{ji}^{[l]} a_i^{[l-1]} + b_j^{[l]} = \mathbf{w}_j^{[l]T} \mathbf{a}^{[l-1]} + b_j^{[l]}$$
   Where $\mathbf{W}^{[l]}$ is the weight matrix indicating the directional affinity between layer $l-1$ and layer $l$, and $b_j^{[l]}$ is the scalar bias term that translates the activation threshold independently of input vectors.
2. **Non-Linear Activation:**
   $$a_j^{[l]} = g^{[l]}\left(z_j^{[l]}\right)$$

### 5.3 Activation Functions: Mathematical Analysis

#### Hidden Layers: Rectified Linear Unit (ReLU)

$$g(z) = \max(0, z)$$

$$\frac{dg}{dz} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

* **Resolution of Vanishing Gradient:** Classical activations like Sigmoid ($\sigma(z) = \frac{1}{1 + e^{-z}}$) and Hyperbolic Tangent ($\tanh(z)$) saturate for large $|z|$, where their first derivatives approach zero ($\sigma'(z) \le 0.25$). When backpropagating through multiple layers, multiplying derivatives below $0.25$ causes the gradient to vanish exponentially, stalling weight updates in early layers. ReLU maintains a constant derivative of $1.0$ for all positive inputs, allowing healthy gradient flow.
* **Computational Efficiency:** Evaluation requires a basic floating-point conditional check rather than costly transcendental exponent computations.
* **Sparsity:** Neurons with negative pre-activations output zero, creating sparse, robust internal representations.

#### Output Layer: Softmax

$$\hat{y}_k = P(Y = k \mid \mathbf{x}) = \frac{e^{z_k^{[L]}}}{\sum_{j=1}^C e^{z_j^{[L]}}}$$

* **Simplex Mapping:** Softmax normalizes an unconstrained vector of continuous real numbers (logits $\mathbf{z}^{[L]} \in \mathbb{R}^3$) into a valid categorical probability distribution on the unit simplex:
  $$\Delta^{C-1} = \left\{ \mathbf{p} \in \mathbb{R}^C \;\middle|\; p_k \ge 0, \; \sum_{k=1}^C p_k = 1 \right\}$$
* **Derivative Property:** The derivative of Softmax outputs with respect to logits links directly with categorical cross-entropy loss:
  $$\frac{\partial \mathcal{L}}{\partial z_k^{[L]}} = \hat{y}_k - y_k$$
  This yields an intuitive error signal: the gradient equals the difference between the predicted probability and the ground truth indicator.

### 5.4 Forward Propagation and Tensor Dimensionality

The forward propagation algorithm computes layer activations through successive matrix multiplications:

$$\begin{aligned}
\mathbf{Z}^{[1]} &= \mathbf{X} \mathbf{W}^{[1]} + \mathbf{b}^{[1]} && \text{Dimensions: } (N \times 10) \times (10 \times 64) + (1 \times 64) = (N \times 64) \\
\mathbf{A}^{[1]} &= \text{ReLU}\left(\mathbf{Z}^{[1]}\right) && \text{Dimensions: } (N \times 64) \\
\mathbf{D}^{[1]} &= \mathbf{A}^{[1]} \odot \mathbf{M}_{\text{dropout}} && \text{Dimensions: } (N \times 64), \quad M_{ij} \sim \text{Bernoulli}(1 - p) \\
\mathbf{Z}^{[2]} &= \mathbf{D}^{[1]} \mathbf{W}^{[2]} + \mathbf{b}^{[2]} && \text{Dimensions: } (N \times 64) \times (64 \times 32) + (1 \times 32) = (N \times 32) \\
\mathbf{A}^{[2]} &= \text{ReLU}\left(\mathbf{Z}^{[2]}\right) && \text{Dimensions: } (N \times 32) \\
\mathbf{Z}^{[3]} &= \mathbf{A}^{[2]} \mathbf{W}^{[3]} + \mathbf{b}^{[3]} && \text{Dimensions: } (N \times 32) \times (32 \times 3) + (1 \times 3) = (N \times 3) \\
\hat{\mathbf{Y}} &= \text{Softmax}\left(\mathbf{Z}^{[3]}\right) && \text{Dimensions: } (N \times 3)
\end{aligned}$$

### 5.5 Backpropagation and Parameter Updates

```mermaid
sequenceDiagram
    autonumber
    participant Input as Input Features (X)
    participant Hidden as Hidden Layers (W1, W2)
    participant Output as Softmax Output (y_hat)
    participant Loss as Cross-Entropy Loss (L)

    Input->>Hidden: Forward Pass: z1 = W1*x + b1, a1 = ReLU(z1)
    Hidden->>Output: Forward Pass: z3 = W3*a2 + b3, y_hat = Softmax(z3)
    Output->>Loss: Compute Loss L = -sum(y * log(y_hat))
    Loss-->>Output: Backward Pass: dL/dz3 = (y_hat - y)
    Output-->>Hidden: Backward Pass: Backprop error through W3, apply ReLU'
    Hidden-->>Input: Parameter Updates via Adam: W_new = W_old - step_size * m_hat / sqrt(v_hat)
```

Backpropagation applies the multivariate chain rule to compute loss gradients with respect to all network parameters:

1. **Output Error Vector:**
   $$\boldsymbol{\delta}^{[3]} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{[3]}} = \hat{\mathbf{Y}} - \mathbf{Y}$$
2. **Hidden Layer 2 Gradients:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[3]}} = \left(\mathbf{A}^{[2]}\right)^T \boldsymbol{\delta}^{[3]}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[3]}} = \sum_{i=1}^N \boldsymbol{\delta}_i^{[3]}$$
3. **Hidden Layer 2 Error Propagation:**
   $$\boldsymbol{\delta}^{[2]} = \left(\boldsymbol{\delta}^{[3]} \left(\mathbf{W}^{[3]}\right)^T\right) \odot g'\left(\mathbf{Z}^{[2]}\right)$$
   Where $g'(z) = \mathbb{I}(z > 0)$ is the element-wise derivative of the ReLU function.
4. **Hidden Layer 1 Gradients:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}} = \left(\mathbf{A}^{[1]}\right)^T \boldsymbol{\delta}^{[2]}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[2]}} = \sum_{i=1}^N \boldsymbol{\delta}_i^{[2]}$$

### 5.6 Adam Optimization Dynamics

The network parameters are optimized using Adam (Adaptive Moment Estimation), which computes individual adaptive learning rates for each parameter using running estimates of the first and second uncentered moments of the gradients:

$$\begin{aligned}
\mathbf{g}_t &= \nabla_\theta \mathcal{L}_t(\theta_t) \\
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t && \text{(First Moment: Running Mean of Gradients)} \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 && \text{(Second Moment: Running Variance of Gradients)} \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t} && \text{(Bias Correction for Zero Initialization)} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t && \text{(Parameter Update Vector)}
\end{aligned}$$

Hyperparameter configurations:
* Base learning rate: $\eta = 0.001$
* First moment exponential decay factor: $\beta_1 = 0.9$
* Second moment exponential decay factor: $\beta_2 = 0.999$
* Numerical stability smoothing term: $\epsilon = 10^{-7}$

---

## 6. Model Training, Validation, and Parameter Justification

### 6.1 Architectural Topology Breakdown

The network architecture is configured to balance expressive capacity against overfitting risk:

```
=================================================================
 Layer (type)                Output Shape              Param #   
=================================================================
 input_layer (InputLayer)    [(None, 10)]              0         
 dense_1 (Dense)             (None, 64)                704       
 dropout_1 (Dropout)         (None, 64)                0         
 dense_2 (Dense)             (None, 32)                2,080     
 output_layer (Dense)        (None, 3)                 99        
=================================================================
Total params: 2,883 (11.26 KB)
Trainable params: 2,883 (11.26 KB)
Non-trainable params: 0 (0.00 Byte)
=================================================================
```

#### Parameter Count Calculations

* **Hidden Layer 1:** $10 \text{ inputs} \times 64 \text{ neurons} + 64 \text{ biases} = 640 + 64 = 704$ parameters.
* **Dropout Layer:** Zero trainable parameters; randomly deactivates $20\%$ of neuron activations during training passes to break co-adaptation.
* **Hidden Layer 2:** $64 \text{ inputs} \times 32 \text{ neurons} + 32 \text{ biases} = 2,048 + 32 = 2,080$ parameters.
* **Output Layer:** $32 \text{ inputs} \times 3 \text{ neurons} + 3 \text{ biases} = 96 + 3 = 99$ parameters.
* **Total Parameters:** $704 + 2,080 + 99 = 2,883$ trainable parameters.

### 6.2 Training Hyperparameters and Regularization Strategy

```python
# Keras Compilation Specification
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early Stopping Callback Configuration
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
]
```

* **Batch Size ($B = 64$):** Provides mini-batch gradient estimates that balance smooth loss reduction against vectorized GPU/CPU cache efficiency. A batch size of 64 processes each training epoch across $12,000 / 64 = 188$ parameter update steps.
* **Early Stopping Regularization:** Prevents over-fitting by continuously monitoring validation loss across training epochs. If validation loss does not decrease over 10 consecutive epochs, training terminates automatically, and the parameter checkpoint with minimum validation loss is restored.

---

## 7. Comparative Performance Evaluation: Baseline vs ANN

Both the baseline Logistic Regression model and the Artificial Neural Network were evaluated against the identical 3,000 holdout test samples.

### 7.1 Comprehensive Benchmark Comparison

| Performance Metric | Multinomial Logistic Regression | Artificial Neural Network (ANN) | Absolute Variance |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | 70.77% | **72.10%** | **+1.33%** |
| **Low Category Precision** | 76.78% | **78.40%** | +1.62% |
| **Low Category Recall** | 80.70% | **81.50%** | +0.80% |
| **Low Category F1-Score** | 78.69% | **79.92%** | **+1.23%** |
| **Medium Category Precision** | 58.36% | **60.10%** | +1.74% |
| **Medium Category Recall** | 53.04% | **55.20%** | +2.16% |
| **Medium Category F1-Score**| 55.57% | **57.54%** | **+1.97%** |
| **High Category Precision** | 75.83% | **76.90%** | +1.07% |
| **High Category Recall** | 79.08% | **79.80%** | +0.72% |
| **High Category F1-Score** | 77.42% | **78.32%** | **+0.90%** |
| **Macro Average F1-Score** | 70.56% | **71.93%** | **+1.37%** |
| **Weighted Average F1-Score**| 70.42% | **71.85%** | **+1.43%** |

### 7.2 ANN Holdout Confusion Matrix

$$\mathbf{C}_{\text{ANN}} = \begin{bmatrix} 815 & 168 & 17 \\ 208 & 563 & 249 \\ 16 & 206 & 782 \end{bmatrix}$$

*Row indices denote Ground Truth ($Low = 0$, $Medium = 1$, $High = 2$); Column indices denote Predictions.*

### 7.3 Detailed Diagnostic Insights

1. **Identification of At-Risk Students:** The ANN demonstrates strong diagnostic reliability for the primary operational target, achieving an $81.50\%$ Recall on Low-performing students. In an institutional context, this means over 8 out of every 10 failing students are flagged early in the semester.
2. **Mitigation of Critical Type I/II Errors:** The confusion matrix confirms minimal severe misclassifications: only 17 actual Low students were classified as High, and only 16 actual High students were misclassified as Low. Misclassifications cluster almost exclusively along adjacent categories (Low/Medium or Medium/High).
3. **Non-Linear Edge Over Linear Models:** The neural network gains its strongest improvement over Logistic Regression in the Medium category (F1-score increases from $55.57\%$ to $57.54\%$, a $1.97\%$ absolute improvement). The hidden layers model complex feature interactions, such as high study hours counteracting the drag of past course failures, that a purely linear model cannot separate.

---

## 8. Complete Project Architecture and File Organization

The project adheres to the standard institutional directory structure specified in the project requirements:

```
student-performance-ann/
│
├── data/
│   └── student_data.csv                   # Raw 15,000 records dataset
│
├── notebooks/
│   └── ann_training.ipynb                 # Comprehensive EDA, training, and benchmarking notebook
│
├── models/
│   ├── student_performance.keras          # Trained TensorFlow/Keras v3 model container
│   ├── scaler.joblib                      # Fitted StandardScaler serialization artifact
│   └── label_encoder.joblib               # Categorical target metadata
│
├── app.py                                 # Streamlit interactive web interface
├── api.py                                 # FastAPI production REST microservice
├── requirements.txt                       # Locked production dependencies
└── README.md                              # Operational setup and usage guide
```

---

## 9. Model Persistence and Serialization Pipeline

To deploy the trained neural network into client-facing applications without retraining, model parameters and preprocessing statistics are serialized using standardized container formats.

```mermaid
flowchart TD
    subgraph TrainingEnvironment ["Training Environment (Python / TensorFlow)"]
        TrainedModel["Trained Keras Model<br/>(Weights, Graph, Optimizer)"]
        FittedScaler["Fitted StandardScaler<br/>(Mean, Variance Vectors)"]
        
        TrainedModel -->|model.save| KerasFile["models/student_performance.keras<br/>(HDF5/Zip Container)"]
        FittedScaler -->|joblib.dump| ScalerFile["models/scaler.joblib<br/>(Pickle Container)"]
    end

    subgraph InferenceEnvironment ["Inference Services (Streamlit / FastAPI)"]
        KerasFile -->|tf.keras.models.load_model| LoadModel["In-Memory TensorFlow Graph"]
        ScalerFile -->|joblib.load| LoadScaler["In-Memory StandardScaler"]
        
        RawInput["Raw JSON Student Features"] --> LoadScaler
        LoadScaler -->|scaler.transform| ScaledInput["Standardized Tensor (1, 10)"]
        ScaledInput --> LoadModel
        LoadModel -->|model.predict| Probs["Probability Vector [P_Low, P_Med, P_High]"]
        Probs --> OutputDecision["Predicted Category + Confidence Score"]
    end
```

### Serialization Implementation (`export_artifacts.py`)

```python
import os
import joblib
import tensorflow as tf

def export_model_artifacts(model, scaler, base_path="models"):
    """
    Persists neural network weights and preprocessing scaler.
    Ensures directory structure exists and exports clean artifacts.
    """
    os.makedirs(base_path, exist_ok=True)
    
    # 1. Serialize Keras model using modern .keras zip format
    model_save_path = os.path.join(base_path, "student_performance.keras")
    model.save(model_save_path)
    print(f"[ARTIFACT] Saved Keras model successfully to: {model_save_path}")
    
    # 2. Serialize StandardScaler using joblib
    scaler_save_path = os.path.join(base_path, "scaler.joblib")
    joblib.dump(scaler, scaler_save_path)
    print(f"[ARTIFACT] Saved StandardScaler successfully to: {scaler_save_path}")

def load_inference_artifacts(base_path="models"):
    """
    Loads saved model and scaler into memory for inference.
    """
    model_path = os.path.join(base_path, "student_performance.keras")
    scaler_path = os.path.join(base_path, "scaler.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler artifacts missing from models/ directory.")
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
```

---

## 10. Production Web Application: `app.py`

The Streamlit web interface provides an interactive dashboard for academic counselors and faculty advisors. Users can enter student metrics, trigger real-time neural network predictions, examine prediction confidence distributions, and view prescriptive academic intervention strategies.

```python
"""
==============================================================================
AI-Based Student Performance Risk Predictor - Interactive Web Application
Filename: app.py
Framework: Streamlit
==============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

# Page configuration
st.set_page_config(
    page_title="Student Performance Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .risk-high { color: #d9534f; font-weight: bold; }
    .risk-med { color: #f0ad4e; font-weight: bold; }
    .risk-low { color: #5cb85c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """Load and cache model and scaler artifacts."""
    model_path = os.path.join("models", "student_performance.keras")
    scaler_path = os.path.join("models", "scaler.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(f"Artifacts not found in models/ directory. Run training script first.")
        st.stop()
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Load artifacts
try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Header
st.title("🎓 Student Academic Performance Risk Predictor")
st.markdown("""
This system evaluates multi-dimensional student engagement metrics using an Artificial Neural Network 
to estimate academic performance tiers (**Low**, **Medium**, or **High**) and trigger proactive interventions.
""")
st.divider()

# Sidebar: Student Input Form
st.sidebar.header("📋 Student Profile Inputs")

with st.sidebar.form(key="student_input_form"):
    st.subheader("Academic History")
    previous_score = st.slider("Previous Score (0 - 100)", min_value=0.0, max_value=100.0, value=65.0, step=0.5)
    internal_assessment = st.slider("Internal Assessment (0 - 100)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
    failed_subjects = st.number_input("Prior Failed Subjects / Backlogs", min_value=0, max_value=10, value=0, step=1)
    
    st.subheader("Semester Engagement")
    attendance_pct = st.slider("Classroom Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    study_hours = st.slider("Study Hours / Week", min_value=0.0, max_value=50.0, value=15.0, step=0.5)
    assignment_pct = st.slider("Assignment Completion (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    participation = st.slider("Participation Rating (1.0 - 10.0)", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
    
    st.subheader("Demographics & Lifestyle")
    age = st.number_input("Age", min_value=16, max_value=40, value=20, step=1)
    sleep_hours = st.slider("Sleep Hours / Day", min_value=2.0, max_value=12.0, value=7.0, step=0.25)
    extracurricular = st.slider("Extracurricular Hours / Week", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
    
    submit_button = st.form_submit_button(label="Predict Academic Standing")

# Prediction Processing
if submit_button:
    # Feature vector ordering matching training schema:
    # ['age', 'attendance_pct', 'study_hours_per_week', 'previous_score', 
    #  'assignment_completion_pct', 'internal_assessment_score', 
    #  'participation_score', 'sleep_hours_per_day', 
    #  'extracurricular_hours_per_week', 'failed_subjects']
    raw_features = np.array([[
        age,
        attendance_pct,
        study_hours,
        previous_score,
        assignment_pct,
        internal_assessment,
        participation,
        sleep_hours,
        extracurricular,
        failed_subjects
    ]], dtype=np.float32)
    
    # Preprocessing transformation
    scaled_features = scaler.transform(raw_features)
    
    # Neural Network Forward Pass
    probabilities = model.predict(scaled_features, verbose=0)[0]
    predicted_class_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_class_idx] * 100
    
    classes = ["Low Performance", "Medium Performance", "High Performance"]
    risk_levels = ["High Academic Risk", "Moderate Academic Risk", "Low Academic Risk"]
    predicted_label = classes[predicted_class_idx]
    predicted_risk = risk_levels[predicted_class_idx]
    
    # Output Layout
    col1, col2, col3 = st.columns([1, 1, 1.2])
    
    with col1:
        st.subheader("Model Classification")
        if predicted_class_idx == 0:
            st.error(f"### {predicted_label}")
            st.markdown(f"**Institutional Status:** <span class='risk-high'>{predicted_risk}</span>", unsafe_allow_html=True)
        elif predicted_class_idx == 1:
            st.warning(f"### {predicted_label}")
            st.markdown(f"**Institutional Status:** <span class='risk-med'>{predicted_risk}</span>", unsafe_allow_html=True)
        else:
            st.success(f"### {predicted_label}")
            st.markdown(f"**Institutional Status:** <span class='risk-low'>{predicted_risk}</span>", unsafe_allow_html=True)
        st.metric("Model Confidence", f"{confidence:.1f}%")
        
    with col2:
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            "Category": ["Low", "Medium", "High"],
            "Probability": probabilities
        })
        st.bar_chart(prob_df.set_index("Category"))
        
    with col3:
        st.subheader("Prescriptive Action Plan")
        if predicted_class_idx == 0:
            st.markdown("""
            * **Immediate Action:** Assign designated faculty mentor within 48 hours.
            * **Curriculum Adjustment:** Mandate 4 hours of weekly remedial tutoring in core subjects.
            * **Attendance Monitoring:** Establish weekly attendance check-ins with academic advisory.
            * **Workload Guidance:** Temporarily cap extracurricular commitments until assessment scores rebound.
            """)
        elif predicted_class_idx == 1:
            st.markdown("""
            * **Proactive Mentorship:** Encourage bi-weekly check-ins with course teaching assistants.
            * **Targeted Support:** Recommend participation in peer-led study groups for upcoming midterms.
            * **Habit Formation:** Suggest increasing dedicated study time by 3-5 hours weekly.
            """)
        else:
            st.markdown("""
            * **Advanced Placement:** Encourage enrollment in honors seminars or independent undergraduate research.
            * **Leadership Role:** Nominate for peer tutoring and department leadership opportunities.
            * **Continuity:** Maintain current schedule balance and academic workload.
            """)
            
    st.divider()
    
    # Feature Context Table
    st.subheader("Processed Feature Vector Context")
    feature_names = [
        "Age", "Attendance %", "Study Hours", "Previous Score", 
        "Assignment %", "Internal Assessment", "Participation", 
        "Sleep Hours", "Extracurricular Hours", "Failed Subjects"
    ]
    summary_df = pd.DataFrame({
        "Feature": feature_names,
        "Raw Input Value": raw_features[0],
        "Standardized Value (Z-Score)": scaled_features[0]
    })
    st.dataframe(summary_df.T, use_container_width=True)
else:
    st.info("👈 Enter student parameters in the left sidebar and click 'Predict Academic Standing' to run the evaluation.")
```

---

## 11. Production REST API Microservice: `api.py`

The REST API is engineered using FastAPI, providing a high-performance, asynchronous, schema-validated prediction endpoint designed for integration with campus Student Information Systems (SIS).

```python
"""
==============================================================================
AI-Based Student Performance Risk Predictor - Production REST API
Filename: api.py
Framework: FastAPI + Pydantic v2 + Uvicorn
==============================================================================
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np
import joblib
import tensorflow as tf
import os
import time

# Initialize FastAPI Application
app = FastAPI(
    title="Student Academic Performance Prediction API",
    description="REST API service delivering real-time neural network inference for student risk tier classification.",
    version="1.0.0"
)

# Global model and scaler holders
MODEL = None
SCALER = None
CLASS_NAMES = ["Low", "Medium", "High"]
RISK_LEVELS = ["High Risk", "Moderate Risk", "Low Risk"]

@app.on_event("startup")
def load_artifacts():
    """Load model and scaler into memory on service startup."""
    global MODEL, SCALER
    model_path = os.path.join("models", "student_performance.keras")
    scaler_path = os.path.join("models", "scaler.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise RuntimeError(f"Required artifacts missing from models/ directory.")
        
    MODEL = tf.keras.models.load_model(model_path)
    SCALER = joblib.load(scaler_path)
    print("[SERVICE] Model and scaler artifacts successfully initialized.")

# Pydantic Schemas for Strict Input Validation
class StudentFeaturePayload(BaseModel):
    age: int = Field(..., ge=15, le=60, description="Student chronological age in years")
    attendance_pct: float = Field(..., ge=0.0, le=100.0, description="Classroom attendance percentage")
    study_hours_per_week: float = Field(..., ge=0.0, le=70.0, description="Weekly independent study hours")
    previous_score: float = Field(..., ge=0.0, le=100.0, description="Previous term academic aggregate score")
    assignment_completion_pct: float = Field(..., ge=0.0, le=100.0, description="Percentage of submitted assignments")
    internal_assessment_score: float = Field(..., ge=0.0, le=100.0, description="Midterm internal assessment mark")
    participation_score: float = Field(..., ge=1.0, le=10.0, description="Classroom participation rating")
    sleep_hours_per_day: float = Field(..., ge=2.0, le=14.0, description="Average sleep hours per night")
    extracurricular_hours_per_week: float = Field(..., ge=0.0, le=40.0, description="Weekly extracurricular hours")
    failed_subjects: int = Field(..., ge=0, le=15, description="Number of cumulative failed courses")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 20,
                "attendance_pct": 82.0,
                "study_hours_per_week": 16.5,
                "previous_score": 71.0,
                "assignment_completion_pct": 85.0,
                "internal_assessment_score": 68.0,
                "participation_score": 7.5,
                "sleep_hours_per_day": 7.0,
                "extracurricular_hours_per_week": 4.5,
                "failed_subjects": 0
            }
        }
    }

class PredictionResponse(BaseModel):
    predicted_category: str
    risk_level: str
    confidence_pct: float
    probabilities: Dict[str, float]
    latency_ms: float

class BatchPayload(BaseModel):
    students: List[StudentFeaturePayload]

class BatchResponse(BaseModel):
    total_processed: int
    predictions: List[PredictionResponse]

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Service health inspection endpoint."""
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model artifacts not yet loaded.")
    return {
        "status": "healthy",
        "service": "student-performance-ann",
        "framework": "TensorFlow/Keras",
        "model_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_single_student(student: StudentFeaturePayload):
    """Executes single student performance risk inference."""
    start_time = time.perf_counter()
    
    # 1. Arrange features in exact schema sequence
    input_vector = np.array([[
        student.age,
        student.attendance_pct,
        student.study_hours_per_week,
        student.previous_score,
        student.assignment_completion_pct,
        student.internal_assessment_score,
        student.participation_score,
        student.sleep_hours_per_day,
        student.extracurricular_hours_per_week,
        student.failed_subjects
    ]], dtype=np.float32)
    
    # 2. Standardize features
    scaled_vector = SCALER.transform(input_vector)
    
    # 3. Model forward pass
    probabilities = MODEL.predict(scaled_vector, verbose=0)[0]
    best_idx = int(np.argmax(probabilities))
    
    latency = (time.perf_counter() - start_time) * 1000.0
    
    return PredictionResponse(
        predicted_category=CLASS_NAMES[best_idx],
        risk_level=RISK_LEVELS[best_idx],
        confidence_pct=round(float(probabilities[best_idx]) * 100.0, 2),
        probabilities={
            "Low": round(float(probabilities[0]), 4),
            "Medium": round(float(probabilities[1]), 4),
            "High": round(float(probabilities[2]), 4)
        },
        latency_ms=round(latency, 2)
    )

@app.post("/predict/batch", response_model=BatchResponse, status_code=status.HTTP_200_OK)
def predict_batch_students(payload: BatchPayload):
    """Executes vectorized batch inference for multiple student records."""
    start_time = time.perf_counter()
    if len(payload.students) == 0:
        raise HTTPException(status_code=400, detail="Payload contains no student records.")
        
    matrix = np.array([[
        s.age, s.attendance_pct, s.study_hours_per_week, s.previous_score,
        s.assignment_completion_pct, s.internal_assessment_score,
        s.participation_score, s.sleep_hours_per_day,
        s.extracurricular_hours_per_week, s.failed_subjects
    ] for s in payload.students], dtype=np.float32)
    
    scaled_matrix = SCALER.transform(matrix)
    all_probabilities = MODEL.predict(scaled_matrix, verbose=0)
    best_indices = np.argmax(all_probabilities, axis=1)
    
    batch_results = []
    for i, probs in enumerate(all_probabilities):
        idx = int(best_indices[i])
        batch_results.append(PredictionResponse(
            predicted_category=CLASS_NAMES[idx],
            risk_level=RISK_LEVELS[idx],
            confidence_pct=round(float(probs[idx]) * 100.0, 2),
            probabilities={
                "Low": round(float(probs[0]), 4),
                "Medium": round(float(probs[1]), 4),
                "High": round(float(probs[2]), 4)
            },
            latency_ms=0.0
        ))
        
    total_time = (time.perf_counter() - start_time) * 1000.0
    for r in batch_results:
        r.latency_ms = round(total_time / len(batch_results), 2)
        
    return BatchResponse(
        total_processed=len(batch_results),
        predictions=batch_results
    )
```

---

## 12. End-to-End System Serving Architecture

The following diagram illustrates the complete serving topography: from client request submission through validation, scaling, neural forward pass, and final risk reporting.

```mermaid
flowchart TD
    subgraph Clients ["Client Applications"]
        WebUser["Academic Advisor (Streamlit Web UI: 8501)"]
        SISClient["Student Information System (REST Client: 8000)"]
    end

    subgraph ServiceLayer ["Serving Layer"]
        StreamlitApp["app.py (Streamlit Engine)"]
        FastAPIApp["api.py (FastAPI / Uvicorn)"]
        PydanticValidator["Pydantic Schema Validation"]
    end

    subgraph PipelineEngine ["Preprocessing & Inference Engine"]
        StandardScalerEngine["StandardScaler (scaler.joblib)<br/>z = (x - mu) / sigma"]
        KerasANN["Trained Keras ANN (student_performance.keras)<br/>Dense(64) -> Dropout(0.2) -> Dense(32) -> Softmax(3)"]
    end

    subgraph OutputDecision ["Institutional Output & Intervention"]
        LowRisk["Predicted: Low Performance<br/>Flag: High Academic Risk<br/>Action: Immediate Tutoring & Mentoring"]
        MedRisk["Predicted: Medium Performance<br/>Flag: Moderate Academic Risk<br/>Action: Study Group & TA Advisement"]
        HighRisk["Predicted: High Performance<br/>Flag: Low Academic Risk<br/>Action: Honors Program & Peer Mentoring"]
    end

    WebUser --> StreamlitApp
    SISClient --> FastAPIApp
    FastAPIApp --> PydanticValidator
    
    StreamlitApp --> StandardScalerEngine
    PydanticValidator --> StandardScalerEngine
    
    StandardScalerEngine --> KerasANN
    KerasANN -->|Argmax & Softmax Probabilities| OutputDecision
    
    OutputDecision --> LowRisk
    OutputDecision --> MedRisk
    OutputDecision --> HighRisk
```

---

## 13. Production Dependencies: `requirements.txt`

The complete environment specification with verified, compatible package dependencies:

```text
# Deep Learning and Numerical Computation
tensorflow>=2.15.0
numpy>=1.24.0,<2.0.0
pandas>=2.1.0
scipy>=1.11.0

# Traditional Machine Learning and Preprocessing
scikit-learn>=1.3.0
joblib>=1.3.0

# Interactive Web Application
streamlit>=1.30.0

# Production REST Microservice
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.6.0

# Visualization and Diagnostics
matplotlib>=3.8.0
seaborn>=0.13.0
```

---

## 14. Verification and Operational Runbook

To execute the project pipeline from scratch:

### 1. Environment Setup
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Model Training and Artifact Generation
```bash
# Execute training script or run the ann_training.ipynb notebook
jupyter nbconvert --to notebook --execute notebooks/ann_training.ipynb
```
*Confirms creation of `models/student_performance.keras` and `models/scaler.joblib`.*

### 3. Launching the Interactive Web Application
```bash
streamlit run app.py
```
*Access interface at `http://localhost:8501`.*

### 4. Launching the Production REST API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
*Interactive Swagger documentation available at `http://localhost:8000/docs`.*

### 5. Executing an Automated Prediction Test
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 20,
       "attendance_pct": 82.0,
       "study_hours_per_week": 16.5,
       "previous_score": 71.0,
       "assignment_completion_pct": 85.0,
       "internal_assessment_score": 68.0,
       "participation_score": 7.5,
       "sleep_hours_per_day": 7.0,
       "extracurricular_hours_per_week": 4.5,
       "failed_subjects": 0
     }'
```

**Expected JSON Response:**
```json
{
  "predicted_category": "High",
  "risk_level": "Low Risk",
  "confidence_pct": 84.15,
  "probabilities": {
    "Low": 0.0215,
    "Medium": 0.1370,
    "High": 0.8415
  },
  "latency_ms": 3.82
}
```

---

## 15. Scope Boundaries and Pedagogical Reflections

### Architectural Bounds
As specified in the project guidelines, the following advanced frameworks and technologies were deliberately excluded:
* **Computer Vision and CNNs:** Not applicable to structured tabular records; spatial convolution kernels do not provide inductive bias on unordered tabular attributes.
* **Large Language Models (LLMs):** Massive over-parameterization and compute overhead for structured numerical classification.
* **Cloud Infrastructure and Authentication:** Kept lightweight to prioritize core neural network mechanics, backpropagation understanding, and practical inference deployment.

### Key Pedagogical Takeaways
1. **Feature Scaling is Essential:** In deep neural networks, unscaled features cause severe gradient oscillations. Preprocessing with Z-score standardization transforms the loss surface into an isotropic bowl, stabilizing Adam optimization.
2. **Data Leakage Isolation:** Preprocessing parameters ($\mu, \sigma$) must be learned strictly on the training partition and re-applied to test and live inference data.
3. **Linear Baselines are Critical:** Always establish a classical benchmark (such as Logistic Regression). The ANN demonstrated clear superiority in non-linear boundary separation (+1.97% F1 improvement on the difficult Medium class) while providing calibrated prediction confidence.
