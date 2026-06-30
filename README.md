# SmartCart: E-Commerce Customer Segmentation

This project uses unsupervised machine learning to group e-commerce customers based on their demographics, spending habits, and how they interact with the store. 

By applying PCA (Principal Component Analysis) to simplify the data, and using K-Means and Hierarchical Clustering, the system classifies customers into 4 distinct groups. This helps marketing teams run highly targeted campaigns instead of blasting the same generic email to everyone on their list.

---

## Why This Matters (The Business Case)
A "one-size-fits-all" marketing approach is expensive and usually gets low conversion rates. By segmenting the customer base, we can make smarter marketing decisions:
*   **Maximize VIP Revenue:** Target high-income, high-spending couples with premium items (like wine and gold products).
*   **Boost Engagement:** Send budget-friendly bundle deals to larger families who visit the web store often but spend less per visit.
*   **Optimize Spend:** Stop sending expensive catalog mailers to groups that only buy online, and vice versa.

---

## How the Pipeline Works
Here is the step-by-step data science workflow used to clean the data and build the clusters:

1.  **Data Cleaning & Preprocessing:** 
    *   Missing income entries were filled using the median to prevent skewing the data.
    *   Customer sign-up dates were converted into a relative metric: `Customer_Tenure_Days`.
    *   Ages were calculated using birth years relative to 2026.
2.  **Feature Engineering:**
    *   Combined individual item spending (wines, fruits, meats, fish, sweets, and gold) into a single `Total_Spending` feature.
    *   Combined household kids and teens into a single `Total_Children` metric.
    *   Dropped collinear and redundant columns (like ID, sign-up dates, and birth years) to clean up the feature set.
3.  **Outlier Removal:**
    *   Filtered out a few noise data points (customers older than 90 or with an annual income over $600k) to keep the clustering boundaries clean. This left us with a highly clean dataset of **2,236** customers.
4.  **Encoding & Scaling:**
    *   One-hot encoded categorical variables (`Education` and `Living_With`).
    *   Standardized the dataset using `StandardScaler` so that features with larger ranges (like income) wouldn't dominate the distance calculations.
5.  **PCA (Dimensionality Reduction):**
    *   Reduced the dataset to 3 principal components to compress the data. These 3 components capture **44.96%** of the total variance in the dataset.
6.  **Optimal Cluster Selection ($K$):**
    *   Used K-Means to evaluate WCSS (Within-Cluster Sum of Squares) and Silhouette Scores for $K$ from 1 to 10. The elbow method (validated by the `KneeLocator` package) pointed to **4 clusters** as the sweet spot for clean, interpretable groups.

---

## Performance & Model Evaluation

Here is the breakdown of the cluster metrics computed during the notebook run:

### PCA Component Variance
*   **Component 1 (PC1):** 23.16% variance explained
*   **Component 2 (PC2):** 11.39% variance explained
*   **Component 3 (PC3):** 10.41% variance explained
*   **Cumulative:** 44.96% of the original dataset variance captured in 3 dimensions.

### Finding $K$ (WCSS vs. Silhouette Scores)
| Clusters ($K$) | WCSS (Inertia) | Silhouette Score |
| :---: | :---: | :---: |
| 1 | 18093.26 | *N/A* |
| 2 | 10760.84 | 0.3716 |
| 3 | 8830.29 | 0.3077 |
| **4 (Selected)** | **6650.97** | **0.3581** |
| 5 | 5006.16 | 0.4000 |
| 6 | 4396.31 | 0.3993 |
| 7 | 3857.63 | 0.4026 |
| 8 | 3207.06 | 0.4051 |
| 9 | 3025.22 | 0.4012 |
| 10 | 2651.44 | 0.4029 |

---

## Customer Personas & Marketing Playbook

Using Hierarchical Agglomerative Clustering on the 3D PCA space, the customer base naturally split into 4 clear personas:

### Cluster Centroid Averages
| Metric | Cluster 0: Partnered / Mid-Income | Cluster 1: Partnered / High-Income | Cluster 2: Single / Low-Income | Cluster 3: Single / High-Income |
| :--- | :---: | :---: | :---: | :---: |
| **% of Customer Base** | 40.7% | 23.9% | 19.9% | 15.5% |
| **Average Income** | $39,680 | **$72,808** | $36,960 | **$70,722** |
| **Average Age** | 55.7 years | 59.5 years | 55.7 years | 58.9 years |
| **Total Children** | 1.24 | 0.51 | 1.27 | 0.46 |
| **Total Spending** | $221.96 | **$1,236.59** | $165.70 | **$1,190.39** |
| **Monthly Web Visits** | 6.3 | 3.6 | 6.7 | 3.7 |
| **Household Status** | 100% Partnered | 100% Partnered | 99.3% Single | 100% Single |

### Actionable Strategies

```carousel
#### 👨‍👩‍👧‍👦 Cluster 0: Partnered / Mid-Income Families (40.7% of customers)
*   **Profile**: Married or cohabiting couples with multiple kids. They have moderate incomes, visit the site frequently, but keep their average spending low.
*   **Strategy**: Focus on value-driven offers. Promote family bundle deals, discount campaigns (like back-to-school), and loyalty rewards to convert their frequent web visits into purchases.
<!-- slide -->
#### 💎 Cluster 1: High-Income Couples (23.9% of customers)
*   **Profile**: Partnered households with few or no children at home. They have the highest average income and are the biggest spenders, buying heavily from physical stores and catalogs.
*   **Strategy**: Enroll them in a premium VIP loyalty tier. Target them with high-end wines and gold products. Offer premium delivery, exclusive early access, and personalized concierge offers.
<!-- slide -->
#### 🙋‍♂️ Cluster 2: Single Budget Shoppers (19.9% of customers)
*   **Profile**: Single-parent or solo households with lower incomes and very low spending, but the highest web visit rates.
*   **Strategy**: Target them with budget-focused retargeting ads, free shipping codes, and flexible payment options (like Buy Now, Pay Later) to reduce cart abandonment.
<!-- slide -->
#### 🕶️ Cluster 3: Affluent Singles (15.5% of customers)
*   **Profile**: Solo buyers with high incomes, low children count, and very high spending. They also show the highest conversion rate on marketing campaigns.
*   **Strategy**: Target them with direct digital channels (personalized email/mobile recommendations). Run time-limited flash sales on high-end gadgets and boutique items to play into their high campaign response rate.
```

---

## Future Roadmap: Ideas to Improve Performance

If I were taking this model to production, here are the main steps I would take next to improve both clustering quality and execution speed:

### 1. Model & Feature Improvements
*   **Soft Clustering with GMMs:** K-Means forces every customer into a single cluster. Using Gaussian Mixture Models (GMM) would give us "soft clustering," representing a customer's probability of belonging to multiple groups (which is much closer to real shopping behaviors).
*   **DBSCAN for Auto-Outlier Detection:** Rather than setting hard, manual cutoffs for age and income, running DBSCAN or HDBSCAN would automatically detect noise points and outliers based on density.
*   **UMAP for Projection:** PCA is linear. If the customer relationships are non-linear, projecting down to 3D space using UMAP (Uniform Manifold Approximation and Projection) or t-SNE would likely reveal cleaner, more distinct visual boundaries.
*   **Better Financial Ratios:** I'd like to create features like "Discretionary Spending Ratio" (spending relative to income) or "Web Conversion Rate" (purchases divided by monthly visits) to give the models stronger signals to group on.

### 2. Code & Architecture Optimizations
*   **Full Pipeline Implementation:** Wrapping the preprocessing, scaling, and PCA steps into a scikit-learn `Pipeline` or `ColumnTransformer`. This prevents data leakage and makes it incredibly easy to deploy the model to a production API endpoint.
*   **Reduce Memory Usage:** Downcasting numeric data types (e.g., converting `float64` to `float32` and `int64` to `int32`). This is a quick win to minimize the memory footprint if the dataset scales to millions of rows.
*   **Parallel Computing:** Adding `n_jobs=-1` to K-Means configuration so scikit-learn uses all available CPU cores when calculating cluster distances.

---

## Local Setup & Run Guide

To run this project on your machine, follow these steps:

### 1. Prerequisites
Make sure you have **Python 3.10+** installed.

### 2. Setup the Environment
Clone the folder, create a virtual environment, and install the libraries:

```bash
# Navigate to project root
cd SmartCart_E-Commerce--A_Customer_Segmentation_System

# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

# Install the required packages
pip install pandas numpy matplotlib seaborn scikit-learn kneed jupyter
```

### 3. Select the Interpreter in your IDE (VS Code)
1. Open the project folder in VS Code.
2. Open `SmartCart_Clustering_System.ipynb`.
3. In the top-right corner, click on the **Kernel Selector** -> **Python Environments**.
4. Select the python environment inside your local `.venv` folder:
   ```text
   .venv/Scripts/python.exe
   ```
*(Note: If VS Code automatically creates a `.vscode/` settings folder, you can safely delete it or add it to your `.gitignore` to keep commits clean).*

### 4. Running the Notebook
To run all cells and update the notebook file in-place:
```bash
jupyter nbconvert --to notebook --execute --inplace SmartCart_Clustering_System.ipynb
```
Or open it interactively:
```bash
jupyter notebook
```