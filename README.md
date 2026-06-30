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
This segment represents couples with multiple kids at home. They have moderate household incomes, visit the web store frequently, but keep their average spending low per visit. The best way to engage them is through value-driven marketing: promote family bundle deals, seasonal discounts (such as back-to-school), and loyalty programs that incentivize converting their frequent browsing into purchases.
<!-- slide -->
#### 💎 Cluster 1: High-Income Couples (23.9% of customers)
These are partnered households with few or no children at home. They represent our highest income group and are the biggest spenders, buying heavily from physical stores and catalogs. Because they value quality and convenience over discounts, we should target them with a premium VIP loyalty tier, market premium items like wine and gold products directly, and offer concierge services or exclusive early access previews.
<!-- slide -->
#### 🙋‍♂️ Cluster 2: Single Budget Shoppers (19.9% of customers)
This segment is made up of single-parent or solo households with lower incomes. They spend the least per visit but have the highest web visit rates. To lower their cart abandonment rates, we should use retargeting ads highlighting budget-friendly entry products, free shipping codes, or flexible checkout options like 'Buy Now, Pay Later'.
<!-- slide -->
#### 🕶️ Cluster 3: Affluent Singles (15.5% of customers)
These are solo buyers with high disposable incomes, few/no kids, and very high spending. Notably, they show the highest conversion rate on marketing campaigns. To capture their attention, we should use direct digital channels with personalized email recommendations, digital flash sales, and targeted social media ads for high-end or trending items.
```

---

## Future Roadmap: Ideas to Improve Performance

If I were taking this project further, here is my planned roadmap to improve the customer segments and optimize the code performance:

### Analytical Roadmap
*   **Transition to Soft Clustering with GMMs:** K-Means enforces hard boundaries, forcing every customer into a single cluster. In reality, shopper profiles are fluid. Implementing a Gaussian Mixture Model (GMM) would assign membership probabilities across multiple segments, giving a much more realistic picture of customer habits.
*   **Density-Based Clustering (DBSCAN):** Rather than setting hard manual thresholds to filter outliers (like age < 90 or income < $600k), DBSCAN or HDBSCAN would automatically treat low-density noise points as outliers, making the boundaries cleaner and more statistically sound.
*   **UMAP for Projection:** PCA is a linear projection and can miss complex, non-linear relationships. Switching to UMAP (Uniform Manifold Approximation and Projection) or t-SNE would help preserve local structures and could yield clearer cluster separations.
*   **Engaging Financial Ratios:** Adding engineered indicators like a "Discretionary Spending Ratio" (total spend relative to income) or "Purchase Conversion Rate" (purchases divided by web visits) would give the model stronger indicators of buyer intent.

### Code & System Roadmap
*   **Unified Pipeline Implementation:** I want to wrap the entire preprocessing, scaling, and PCA steps into a scikit-learn `Pipeline` or `ColumnTransformer`. This prevents data leakage during validation and makes it simple to wrap the model in a clean web API endpoint.
*   **Reduce Memory Footprint:** If the customer database scales to millions of rows, downcasting numeric columns (like converting `float64` to `float32` and `int64` to `int32`) will keep the memory usage low.
*   **Distributing Computation:** Setting `n_jobs=-1` on distance-based models would run operations across all available CPU cores, boosting speed during hyperparameter searches.

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