# 🏠 House Price Prediction Project - Complete Documentation

## Table of Contents
1. [Project Summary](#1-project-summary)
2. [Technology Stack](#2-technology-stack)
3. [Project Architecture](#3-project-architecture)
4. [Data Pipeline & EDA Process](#4-data-pipeline--eda-process)
5. [How to Execute](#5-how-to-execute)
6. [How to Test the Flow](#6-how-to-test-the-flow)
7. [Questions & Answers](#7-questions--answers)

---

## 1. Project Summary

### Overview
**PrimeEstate** is a Machine Learning-based web application that predicts house prices in Bengaluru (Bangalore), India. The system uses a **Random Forest Regressor** trained on the Bengaluru House Data dataset to estimate property values based on key features like location, square footage, number of bedrooms (BHK), and bathrooms.

### Key Features
- 🏡 **Real-time Price Prediction**: Enter property details and get instant price estimates
- 📊 **Market Comparison**: Visual comparison with similar properties
- 💡 **Price Insights**: Automatic analysis showing if a property is underpriced, fairly priced, or overpriced
- 🎨 **Modern UI**: Clean, responsive interface built with Streamlit

### Project Files Structure
```
House Price Prediction/
├── app.py                      # Main Streamlit web application
├── eda.ipynb                   # Jupyter notebook (EDA + Model Training)
├── Bengaluru_House_Data.csv    # Raw dataset (~13,000+ records)
├── cleaned_df.csv              # Processed/cleaned dataset
├── rf_model.joblib             # Trained Random Forest model
├── model_columns.joblib        # Feature columns used by model
├── requirements.txt            # Python dependencies
├── house_logo.png              # App logo
└── design.streamlit/
    └── config.toml             # Streamlit theme configuration
```

---

## 2. Technology Stack

| Technology | Version | Purpose | Why Used |
|------------|---------|---------|----------|
| **Python** | 3.x | Core programming language | Industry standard for ML/Data Science, extensive library ecosystem |
| **Pandas** | Latest | Data manipulation & analysis | Powerful DataFrames for handling tabular data, essential for EDA |
| **NumPy** | Latest | Numerical computations | Fast array operations, foundation for scientific computing |
| **Scikit-learn** | Latest | Machine Learning | Industry-standard ML library with RandomForest, GridSearchCV, metrics |
| **Streamlit** | Latest | Web application framework | Rapid prototyping of ML apps, no frontend code required |
| **Matplotlib** | Latest | Data visualization | Static charts for market comparison histograms |
| **Seaborn** | Latest | Statistical visualization | Enhanced visualizations for EDA (boxplots, etc.) |
| **Joblib** | Latest | Model serialization | Efficient saving/loading of trained ML models |

### Why These Technologies?

#### 1. **Streamlit** (Web Framework)
- ✅ **Zero frontend knowledge required** - Pure Python
- ✅ **Built-in widgets** for ML apps (sliders, selectboxes, etc.)
- ✅ **Hot reloading** - See changes instantly
- ✅ **Easy deployment** - Can deploy to Streamlit Cloud for free

#### 2. **Random Forest Regressor** (ML Algorithm)
- ✅ **Handles non-linear relationships** - Real estate prices aren't linear
- ✅ **Feature importance** - Know which features matter most
- ✅ **Robust to outliers** - Real estate data has many outliers
- ✅ **No feature scaling required** - Works with raw features
- ✅ **Ensemble method** - Combines multiple decision trees for better accuracy

#### 3. **GridSearchCV** (Hyperparameter Tuning)
- ✅ **Automated hyperparameter search** - Finds optimal parameters
- ✅ **Cross-validation built-in** - Prevents overfitting
- ✅ **Exhaustive search** - Tests all parameter combinations

#### 4. **Joblib** (Model Persistence)
- ✅ **Faster than pickle** for large numpy arrays
- ✅ **Compression support** - Smaller file sizes
- ✅ **Standard in scikit-learn** - Native integration

---

## 3. Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    (Streamlit Web App)                          │
│   ┌─────────────┬──────────────┬──────────────┬───────────┐    │
│   │  Location   │  Square Ft   │   Bathrooms  │    BHK    │    │
│   │  Dropdown   │   Input      │   Dropdown   │  Dropdown │    │
│   └─────────────┴──────────────┴──────────────┴───────────┘    │
│                         ↓ [Predict Button]                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT PREPARATION                            │
│  • Create feature dictionary with all model columns = 0         │
│  • Fill numerical values (sqft, bath, bhk)                     │
│  • One-hot encode selected location (set to 1)                 │
│  • Convert to DataFrame                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ML MODEL PREDICTION                          │
│           Random Forest Regressor (rf_model.joblib)            │
│                                                                 │
│  Parameters: n_estimators=200, max_depth=7 (tuned via GridCV) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT & INSIGHTS                            │
│  • Predicted Price (in Lakhs → converted to INR)               │
│  • Price Insight (Under/Over/Fairly priced)                    │
│  • Market Comparison Histogram                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow
```
Raw Data → Cleaning → Feature Engineering → Encoding → Training → Model → Prediction
    ↓          ↓              ↓                ↓          ↓
  CSV    Missing Values   price_per_sqft   One-Hot    GridSearchCV
         Outliers         bhk extraction    Encoding   RandomForest
         Duplicates       Location grouping
```

---

## 4. Data Pipeline & EDA Process

### 4.1 Data Loading
```python
df = pd.read_csv('Bengaluru_House_Data.csv')
# Original: ~13,000+ records with 9 columns
```

### 4.2 Feature Selection
**Columns Removed:**
- `area_type` - Not useful for prediction
- `availability` - Date info not needed
- `society` - Too many unique values, sparse
- `balcony` - Missing many values

**Columns Kept:**
- `location`, `size`, `total_sqft`, `bath`, `price`

### 4.3 Handling Missing Values
| Column | Strategy | Reason |
|--------|----------|--------|
| `location` | Fill with "Sarjapur Road" | Most common location |
| `size` | Fill with "2 BHK" | Most common size |
| `bath` | Fill with median | Numeric column, median avoids outlier impact |

### 4.4 Feature Engineering
1. **BHK Extraction**: Convert "2 BHK" → 2 (extract number)
2. **Square Feet Cleaning**: Handle ranges like "1200-1500" → 1350 (average)
3. **Price per Sqft**: `price * 100000 / total_sqft` (for outlier detection)
4. **Location Grouping**: Locations with ≤10 properties → "others"

### 4.5 Outlier Removal
| Rule | Logic |
|------|-------|
| Sqft per BHK | Remove if `total_sqft/bhk < 300` |
| BHK limit | Keep only BHK ≤ 6 |
| Bathroom limit | Remove if `bath > bhk + 2` |
| Price per Sqft | IQR method (0.5 multiplier) |

### 4.6 Encoding
- **One-Hot Encoding** for `location` column
- Creates ~230+ binary columns (one per location)

### 4.7 Model Training
```python
# Hyperparameter Grid
params = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 4, 5, 6, 7]
}

# GridSearchCV with 5-fold cross-validation
grid = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params, cv=5)
grid.fit(Xtrain, ytrain)

# Best Parameters Found:
# n_estimators: 200
# max_depth: 7
```

### 4.8 Model Performance
| Metric | Value |
|--------|-------|
| Training R² | ~0.85-0.90 |
| Testing R² | ~0.75-0.80 |
| Mean Absolute Error | ~15-20 Lakhs |

---

## 5. How to Execute

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

---

### 🍎 macOS Installation

#### Step 1: Install Python (if not installed)
```bash
# Check if Python is installed
python3 --version

# If not installed, install via Homebrew
brew install python3
```

#### Step 2: Navigate to Project Directory
```bash
cd ~/Downloads/House\ Price\ Prediction
```

#### Step 3: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

#### Step 4: Install Dependencies
```bash
pip3 install -r requirements.txt

# Or using python module directly
python3 -m pip install -r requirements.txt
```

#### Step 5: Run the Application
```bash
# If streamlit is in PATH
streamlit run app.py

# Or using python module
python3 -m streamlit run app.py
```

#### Step 6: Access the App
- Browser opens automatically at `http://localhost:8501`
- Press `Ctrl+C` in terminal to stop the app

---

### 🪟 Windows 11 Installation (Fresh Setup)

#### Step 1: Install Python

**Option A: Download from Python.org (Recommended)**
1. Go to https://www.python.org/downloads/
2. Download **Python 3.11** or later (click "Download Python 3.x.x")
3. Run the installer
4. **IMPORTANT:** Check ✅ **"Add Python to PATH"** at the bottom
5. Click "Install Now"
6. Click "Disable path length limit" when prompted

**Option B: Install via Microsoft Store**
1. Open Microsoft Store
2. Search for "Python 3.11"
3. Click "Get" to install

**Verify Installation:**
```powershell
# Open PowerShell or Command Prompt
python --version
pip --version
```

#### Step 2: Download/Clone the Project

**Option A: Download ZIP**
1. Download the project ZIP file
2. Extract to `C:\Users\YourName\Downloads\House Price Prediction`

**Option B: Using Git**
```powershell
# Install Git first from https://git-scm.com/download/win
cd C:\Users\YourName\Downloads
git clone <repository-url>
```

#### Step 3: Open Terminal in Project Folder

**Method 1:** 
- Open File Explorer
- Navigate to project folder
- Right-click → "Open in Terminal"

**Method 2:**
```powershell
cd "C:\Users\YourName\Downloads\House Price Prediction"
```

#### Step 4: Create Virtual Environment (Recommended)
```powershell
# Create virtual environment
python -m venv venv

# Activate it (PowerShell)
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# For Command Prompt (CMD), use:
venv\Scripts\activate.bat
```

#### Step 5: Install Dependencies
```powershell
# Upgrade pip first (recommended)
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

#### Step 6: Run the Application
```powershell
streamlit run app.py
```

#### Step 7: Access the App
- Browser opens automatically at `http://localhost:8501`
- Press `Ctrl+C` in terminal to stop the app

---

### 🐧 Linux (Ubuntu/Debian) Installation

#### Step 1: Install Python
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### Step 2: Navigate & Setup
```bash
cd ~/Downloads/House\ Price\ Prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 3: Run the Application
```bash
streamlit run app.py
```

---

### 🔧 Troubleshooting Common Issues

| Issue | Solution |
|-------|----------|
| `pip: command not found` (macOS) | Use `pip3` instead of `pip` |
| `python: command not found` (macOS) | Use `python3` instead of `python` |
| `streamlit: command not found` | Use `python -m streamlit run app.py` |
| PowerShell script execution error | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `ModuleNotFoundError: No module named 'xyz'` | Run `pip install xyz` |
| Port 8501 already in use | Use `streamlit run app.py --server.port 8502` |
| Permission denied (Linux/macOS) | Use `sudo` or fix folder permissions |

---

### Alternative: Run the Notebook (For Training/EDA)

**Using Jupyter:**
```bash
# Install Jupyter
pip install jupyter

# Run notebook
jupyter notebook eda.ipynb
```

**Using VS Code:**
1. Install "Jupyter" extension in VS Code
2. Open `eda.ipynb`
3. Select Python interpreter (your venv)
4. Run cells with `Shift+Enter`

---

## 6. How to Test the Flow

### Test Case 1: Basic Prediction
| Input | Value |
|-------|-------|
| Location | Whitefield |
| Square Feet | 1200 |
| Bathrooms | 2 |
| BHK | 2 |

**Expected**: Price around ₹40-60 Lakhs

### Test Case 2: Premium Location
| Input | Value |
|-------|-------|
| Location | Indira Nagar |
| Square Feet | 1500 |
| Bathrooms | 3 |
| BHK | 3 |

**Expected**: Price around ₹100-150 Lakhs (premium area)

### Test Case 3: Large Property
| Input | Value |
|-------|-------|
| Location | Electronic City |
| Square Feet | 2500 |
| Bathrooms | 4 |
| BHK | 4 |

**Expected**: Price around ₹80-120 Lakhs

### Testing Checklist
- [ ] App launches without errors
- [ ] All dropdowns populate correctly
- [ ] Location dropdown shows sorted locations
- [ ] Predict button generates price
- [ ] Price displayed in proper format (₹ XX,XX,XXX)
- [ ] Price insight shows (Underpriced/Fairly/Overpriced)
- [ ] Histogram displays for similar properties
- [ ] Warning shows when insufficient similar data

### Validation Steps
1. **Unit Test** - Verify model loads correctly:
```python
import joblib
model = joblib.load("rf_model.joblib")
print(model)  # Should print RandomForestRegressor details
```

2. **Integration Test** - Test prediction function:
```python
import pandas as pd
import joblib

model = joblib.load("rf_model.joblib")
features = joblib.load("model_columns.joblib")

# Create test input
test_input = {col: 0 for col in features}
test_input['total_sqft'] = 1200
test_input['bath'] = 2
test_input['bhk'] = 2
test_input['location_Whitefield'] = 1

df = pd.DataFrame([test_input])
prediction = model.predict(df)
print(f"Predicted Price: ₹{prediction[0]*100000:,.0f}")
```

---

## 7. Questions & Answers

### General Questions

**Q1: What problem does this project solve?**
> This project helps home buyers, sellers, and real estate agents estimate property prices in Bengaluru based on key property features, removing guesswork from pricing decisions.

**Q2: What is the dataset used?**
> Bengaluru House Data from Kaggle containing ~13,000+ property records with features like location, size, total square feet, bathrooms, balconies, and price.

**Q3: Why Bengaluru specifically?**
> The dataset is specific to Bengaluru's real estate market. The model is trained on local data and understands Bengaluru's location-based pricing patterns.

---

### Technical Questions

**Q4: Why Random Forest instead of Linear Regression?**
> Real estate prices have non-linear relationships (e.g., premium locations don't scale linearly). Random Forest handles:
> - Non-linear patterns
> - Feature interactions
> - Outliers (common in real estate)
> - High-dimensional data (230+ location columns)

**Q5: Why One-Hot Encoding for locations?**
> Location is a **categorical** variable with no ordinal relationship. One-hot encoding creates binary columns allowing the model to learn location-specific price patterns without assuming any ordering.

**Q6: Why drop area_type and balcony?**
> - `area_type`: High cardinality with unclear impact on price
> - `balcony`: Too many missing values, and balcony count has minimal price impact compared to location/size

**Q7: What is GridSearchCV and why use it?**
> GridSearchCV automatically searches through specified hyperparameter combinations with cross-validation to find the best model configuration, preventing manual trial-and-error and overfitting.

**Q8: Why use IQR method for outliers?**
> The IQR (Interquartile Range) method is robust to extreme values and removes statistically anomalous data points without making assumptions about data distribution.

**Q9: Why multiply price by 100,000?**
> The dataset stores prices in **Lakhs** (Indian numbering: 1 Lakh = 100,000). Multiplying converts to absolute INR for display.

---

### Data Processing Questions

**Q10: Why group locations with ≤10 properties as "others"?**
> - Prevents overfitting on rare locations
> - Reduces model dimensionality
> - Locations with few samples don't provide reliable patterns

**Q11: Why require sqft/bhk ≥ 300?**
> A typical bedroom requires at least 300 sqft. Values below this are data entry errors (e.g., 1000 sqft for 10 BHK is unrealistic).

**Q12: Why limit bath to bhk+2?**
> Realistically, a property won't have more than 2 extra bathrooms beyond bedrooms. Values exceeding this are outliers/errors.

---

### Model Performance Questions

**Q13: What is R² score and what does 0.80 mean?**
> R² (coefficient of determination) measures how well the model explains variance in prices. 0.80 means 80% of price variation is explained by the model features.

**Q14: What is MAE and why track it?**
> MAE (Mean Absolute Error) is the average prediction error in Lakhs. If MAE = 15, predictions are off by ₹15 Lakhs on average.

**Q15: How can the model be improved?**
> - Add more features (amenities, age, floor, furnishing)
> - Use ensemble methods (XGBoost, LightGBM)
> - Train location-specific models
> - Include macro-economic indicators

---

### Application Questions

**Q16: How does the Price Insight feature work?**
> It compares your predicted price against similar properties (same BHK, ±20% sqft, ±1 bathroom) and calculates the percentage difference from their average price.

**Q17: What does "Not enough similar data" mean?**
> The cleaned dataset has fewer than 5 properties matching your criteria, making statistical comparison unreliable.

**Q18: Can I deploy this app online?**
> Yes! Options:
> - **Streamlit Cloud**: Free hosting at share.streamlit.io
> - **Heroku**: Free tier available
> - **AWS/GCP/Azure**: More control, but requires setup

---

### Troubleshooting

**Q19: App crashes on startup - what to do?**
> 1. Check if `rf_model.joblib` exists
> 2. Verify `model_columns.joblib` exists
> 3. Ensure `cleaned_df.csv` is present
> 4. Run `pip install -r requirements.txt`

**Q20: Predictions seem wrong - why?**
> - Model is trained on 2017-2018 data; prices have changed
> - Location might be in "others" category
> - Extreme values (very small/large sqft) may be outside training range

**Q21: How to retrain the model?**
> 1. Open `eda.ipynb`
> 2. Run all cells sequentially
> 3. New `rf_model.joblib` will be saved automatically

---

## Quick Reference Commands

### macOS / Linux
```bash
# Create & activate virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Run the app
python3 -m streamlit run app.py

# Run notebook (for retraining)
jupyter notebook eda.ipynb

# Check Python version
python3 --version

# Check installed packages
pip3 list

# Deactivate virtual environment
deactivate
```

### Windows 11 (PowerShell)
```powershell
# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Run notebook (for retraining)
jupyter notebook eda.ipynb

# Check Python version
python --version

# Check installed packages
pip list

# Deactivate virtual environment
deactivate
```

### Windows 11 (Command Prompt)
```cmd
# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Author
Developed as part of Machine Learning (Data Science) Project | CS Engineering

---

*Document Generated: May 2026*
