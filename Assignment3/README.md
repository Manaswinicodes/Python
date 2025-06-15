# Titanic Dataset - Exploratory Data Analysis (EDA)

## 📊 Project Overview

This project conducts a comprehensive Exploratory Data Analysis on the famous Titanic dataset. The analysis focuses on understanding passenger survival patterns, data quality issues, and uncovering relationships between various features.

## 🎯 Objectives

- **Data Understanding**: Examine dataset structure, dimensions, and variable types
- **Missing Value Analysis**: Identify and quantify missing data patterns
- **Statistical Summary**: Generate descriptive statistics for all variables
- **Outlier Detection**: Identify anomalous data points using multiple methods
- **Distribution Analysis**: Understand the shape and characteristics of data distributions
- **Correlation Analysis**: Discover relationships between numeric variables
- **Survival Pattern Analysis**: Investigate factors affecting passenger survival rates

## 📁 Repository Structure

```
EDA-Assignment/
│
├── titanic_eda.py          # Main EDA script
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── data/                  # Data folder (place dataset here)
│   └── titanic.csv        # Titanic dataset
└── outputs/               # Generated plots and results
    ├── missing_values.png
    ├── distributions.png
    ├── correlations.png
    └── survival_analysis.png
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/python.git
   cd python/EDA-Assignment
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data)
   - Place `titanic.csv` in the `data/` folder
   - Or run the script without the dataset (it will generate sample data)

### Running the Analysis

```bash
python titanic_eda.py
```

## 📈 Analysis Components

### 1. Dataset Overview
- Dataset dimensions and structure
- Variable types identification
- Memory usage analysis
- Initial data preview

### 2. Missing Value Analysis
- Comprehensive missing value detection
- Visual representation using heatmaps
- Missing data percentage calculations
- Recommendations for data imputation

### 3. Statistical Summary
- Descriptive statistics for numeric variables
- Frequency distributions for categorical variables
- Central tendency and dispersion measures

### 4. Outlier Detection
- **IQR Method**: Identifies outliers using interquartile range
- **Z-Score Method**: Detects outliers using standard deviation
- Box plot visualizations
- Outlier summary statistics

### 5. Distribution Analysis
- Histograms with kernel density estimation
- Statistical parameters overlay
- Distribution shape assessment
- Normality evaluation

### 6. Correlation Analysis
- Pearson correlation matrix
- Correlation heatmap visualization
- Strong correlation identification
- Feature relationship insights

### 7. Survival Analysis
- Overall survival rate calculation
- Survival by passenger class
- Gender-based survival patterns
- Age and fare impact on survival
- Cross-tabulation analysis

## 📊 Key Findings

### Data Quality Insights
- **Missing Values**: Age (20% missing), Cabin (77% missing), Embarked (0.2% missing)
- **Outliers**: Significant outliers detected in Fare and Age variables
- **Data Types**: Mixed numeric and categorical variables

### Survival Patterns
- **Overall Survival Rate**: ~38% of passengers survived
- **Gender Impact**: Females had significantly higher survival rates (~74% vs ~19% for males)
- **Class Effect**: First-class passengers had 3x higher survival rates than third-class
- **Age Factor**: Children under 15 showed higher survival rates

### Statistical Relationships
- **Strong Negative Correlation**: Passenger class and fare (-0.73)
- **Moderate Correlations**: Age with survival, family size with survival
- **Distribution Insights**: Fare follows log-normal distribution, Age approximately normal

## 🛠 Technologies Used

- **Python 3.x**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical visualization
- **SciPy**: Scientific computing and statistics
- **Jupyter**: Interactive development (optional)

## 📝 Code Features

- **Modular Design**: Functions separated by analysis type
- **Error Handling**: Robust handling of missing files and data issues
- **Sample Data Generation**: Creates realistic sample data if dataset unavailable
- **Comprehensive Visualization**: Multiple plot types for different insights
- **Statistical Rigor**: Uses appropriate statistical methods
- **Clean Output**: Well-formatted results with clear interpretations

## 🔍 Analysis Methods

### Missing Value Techniques
- Count and percentage calculations
- Visual pattern identification
- Impact assessment on analysis

### Outlier Detection Methods
- **Interquartile Range (IQR)**: Q3 + 1.5×IQR rule
- **Z-Score**: Values beyond 3 standard deviations
- **Box Plot Visualization**: Graphical outlier identification

### Distribution Analysis
- **Histograms**: Frequency distribution visualization
- **Kernel Density Estimation**: Smooth distribution curves
- **Summary Statistics**: Mean, median, mode, skewness, kurtosis

### Correlation Methods
- **Pearson Correlation**: Linear relationship measurement
- **Heatmap Visualization**: Color-coded correlation matrix
- **Threshold-based Filtering**: Focus on strong relationships

## 📊 Expected Outputs

The script generates several key outputs:

1. **Console Output**: Detailed statistical summaries and insights
2. **Visualizations**: Multiple plots showing data patterns
3. **Missing Value Analysis**: Comprehensive missingness report
4. **Outlier Summary**: Detailed outlier detection results
5. **Correlation Matrix**: Relationship strength between variables
6. **Survival Insights**: Factors affecting passenger survival

## 🎓 Educational Value

This project demonstrates:
- **Industry-standard EDA practices**
- **Statistical analysis techniques**
- **Data visualization best practices**
- **Python data science workflow**
- **Insight generation from data**
- **Professional code organization**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new analysis'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Kaggle for providing the Titanic dataset
- The data science community for EDA best practices
- Python data science libraries developers

---

**Note**: This analysis is for educational purposes and demonstrates comprehensive EDA techniques using the historic Titanic passenger dataset.