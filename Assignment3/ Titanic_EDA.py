import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

def load_and_explore_data():
    """Load Titanic dataset and perform initial exploration"""
    
    # Load the dataset
    try:
        df = pd.read_csv('titanic.csv')
        print("✓ Dataset loaded successfully")
    except FileNotFoundError:
        print("Creating sample Titanic dataset...")
        df = create_sample_data()
    
    print(f"\n{'='*60}")
    print("DATASET OVERVIEW")
    print(f"{'='*60}")
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage().sum() / 1024:.2f} KB")
    
    print(f"\nColumn Information:")
    print(df.info())
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return df

def create_sample_data():
    """Create sample Titanic dataset for demonstration"""
    np.random.seed(42)
    n_samples = 891
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.normal(29.7, 14.5, n_samples),
        'SibSp': np.random.poisson(0.5, n_samples),
        'Parch': np.random.poisson(0.4, n_samples),
        'Fare': np.random.lognormal(3.2, 1.0, n_samples),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic missing values
    missing_age_idx = np.random.choice(df.index, size=int(0.2 * len(df)), replace=False)
    df.loc[missing_age_idx, 'Age'] = np.nan
    
    missing_embarked_idx = np.random.choice(df.index, size=2, replace=False)
    df.loc[missing_embarked_idx, 'Embarked'] = np.nan
    
    # Ensure age is positive and reasonable
    df['Age'] = df['Age'].clip(0, 80)
    
    return df

def analyze_missing_values(df):
    """Comprehensive missing value analysis"""
    
    print(f"\n{'='*60}")
    print("MISSING VALUE ANALYSIS")
    print(f"{'='*60}")
    
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_count.index,
        'Missing_Count': missing_count.values,
        'Missing_Percentage': missing_percent.values
    }).sort_values('Missing_Count', ascending=False)
    
    print("Missing Values Summary:")
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Visualize missing values
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    missing_df_plot = missing_df[missing_df['Missing_Count'] > 0]
    if not missing_df_plot.empty:
        bars = plt.bar(missing_df_plot['Column'], missing_df_plot['Missing_Count'], 
                      color='lightcoral', alpha=0.7)
        plt.title('Missing Values Count by Column', fontsize=14, fontweight='bold')
        plt.xlabel('Columns')
        plt.ylabel('Missing Count')
        plt.xticks(rotation=45)
        
        for bar, count in zip(bars, missing_df_plot['Missing_Count']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(int(count)), ha='center', va='bottom')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis', alpha=0.8)
    plt.title('Missing Value Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return missing_df

def statistical_summary(df):
    """Generate comprehensive statistical summary"""
    
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    print(f"Numeric columns: {list(numeric_cols)}")
    print(f"Categorical columns: {list(categorical_cols)}")
    
    print(f"\nDescriptive Statistics for Numeric Variables:")
    print(df[numeric_cols].describe().round(2))
    
    print(f"\nCategorical Variables Summary:")
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts())
    
    return numeric_cols, categorical_cols

def detect_outliers(df, numeric_cols):
    """Detect and visualize outliers using multiple methods"""
    
    print(f"\n{'='*60}")
    print("OUTLIER DETECTION")
    print(f"{'='*60}")
    
    outlier_summary = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols[:4]):
        if col in df.columns and df[col].notna().sum() > 0:
            data = df[col].dropna()
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > 3]
            
            outlier_summary[col] = {
                'IQR_outliers': len(iqr_outliers),
                'Z_score_outliers': len(z_outliers),
                'Total_values': len(data)
            }
            
            # Box plot
            if i < len(axes):
                axes[i].boxplot(data, vert=True)
                axes[i].set_title(f'{col} - Box Plot')
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Outlier Summary:")
    outlier_df = pd.DataFrame(outlier_summary).T
    print(outlier_df)
    
    return outlier_summary

def distribution_analysis(df, numeric_cols):
    """Analyze distributions of numeric variables"""
    
    print(f"\n{'='*60}")
    print("DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols[:4]):
        if col in df.columns and df[col].notna().sum() > 0:
            data = df[col].dropna()
            
            if i < len(axes):
                # Histogram with KDE
                axes[i].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
                
                # Add KDE curve
                try:
                    sns.kdeplot(data=data, ax=axes[i], color='red', linewidth=2)
                except:
                    pass
                
                axes[i].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = data.mean()
                median_val = data.median()
                std_val = data.std()
                
                stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
                axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(df, numeric_cols):
    """Analyze correlations between numeric variables"""
    
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, mask=mask, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Find strong correlations
    strong_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_corr.append({
                    'Variable_1': correlation_matrix.columns[i],
                    'Variable_2': correlation_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if strong_corr:
        print(f"\nStrong Correlations (|r| > 0.5):")
        strong_corr_df = pd.DataFrame(strong_corr)
        print(strong_corr_df.round(3))
    else:
        print("\nNo strong correlations found (|r| > 0.5)")
    
    return correlation_matrix

def survival_analysis(df):
    """Analyze survival patterns in Titanic dataset"""
    
    if 'Survived' not in df.columns:
        print("Survival column not found. Skipping survival analysis.")
        return
    
    print(f"\n{'='*60}")
    print("SURVIVAL ANALYSIS")
    print(f"{'='*60}")
    
    survival_rate = df['Survived'].mean()
    print(f"Overall Survival Rate: {survival_rate:.3f} ({survival_rate*100:.1f}%)")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Survival by Gender
    if 'Sex' in df.columns:
        survival_by_sex = df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean']).round(3)
        survival_by_sex.columns = ['Total', 'Survived', 'Survival_Rate']
        print(f"\nSurvival by Gender:")
        print(survival_by_sex)
        
        survival_by_sex['Survival_Rate'].plot(kind='bar', ax=axes[0,0], color=['lightcoral', 'lightblue'])
        axes[0,0].set_title('Survival Rate by Gender', fontweight='bold')
        axes[0,0].set_ylabel('Survival Rate')
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # Survival by Class
    if 'Pclass' in df.columns:
        survival_by_class = df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean']).round(3)
        survival_by_class.columns = ['Total', 'Survived', 'Survival_Rate']
        print(f"\nSurvival by Class:")
        print(survival_by_class)
        
        survival_by_class['Survival_Rate'].plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Survival Rate by Passenger Class', fontweight='bold')
        axes[0,1].set_ylabel('Survival Rate')
        axes[0,1].set_xlabel('Passenger Class')
    
    # Age distribution by survival
    if 'Age' in df.columns:
        survived = df[df['Survived'] == 1]['Age'].dropna()
        not_survived = df[df['Survived'] == 0]['Age'].dropna()
        
        axes[1,0].hist([survived, not_survived], bins=20, alpha=0.7, 
                      label=['Survived', 'Not Survived'], color=['green', 'red'])
        axes[1,0].set_title('Age Distribution by Survival Status', fontweight='bold')
        axes[1,0].set_xlabel('Age')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
    
    # Fare distribution by survival
    if 'Fare' in df.columns:
        survived_fare = df[df['Survived'] == 1]['Fare'].dropna()
        not_survived_fare = df[df['Survived'] == 0]['Fare'].dropna()
        
        axes[1,1].boxplot([survived_fare, not_survived_fare], 
                         labels=['Survived', 'Not Survived'])
        axes[1,1].set_title('Fare Distribution by Survival Status', fontweight='bold')
        axes[1,1].set_ylabel('Fare')
    
    plt.tight_layout()
    plt.show()

def generate_insights(df, correlation_matrix, outlier_summary):
    """Generate key insights from the analysis"""
    
    print(f"\n{'='*60}")
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print(f"{'='*60}")
    
    insights = []
    
    # Data quality insights
    missing_percent = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_percent[missing_percent > 20]
    if not high_missing.empty:
        insights.append(f"⚠️  High missing values detected in: {list(high_missing.index)}")
    
    # Outlier insights
    total_outliers = sum([info['IQR_outliers'] for info in outlier_summary.values()])
    if total_outliers > 0:
        insights.append(f"📊 {total_outliers} outliers detected across numeric variables")
    
    # Correlation insights
    if not correlation_matrix.empty:
        high_corr = correlation_matrix.abs() > 0.7
        high_corr_pairs = []
        for i in range(len(high_corr.columns)):
            for j in range(i+1, len(high_corr.columns)):
                if high_corr.iloc[i, j]:
                    high_corr_pairs.append((high_corr.columns[i], high_corr.columns[j]))
        
        if high_corr_pairs:
            insights.append(f"🔗 Strong correlations found between: {high_corr_pairs}")
    
    # Survival insights (if applicable)
    if 'Survived' in df.columns:
        survival_rate = df['Survived'].mean()
        insights.append(f"🚢 Overall survival rate: {survival_rate:.1%}")
        
        if 'Sex' in df.columns:
            gender_survival = df.groupby('Sex')['Survived'].mean()
            insights.append(f"Gender survival gap: {gender_survival.max() - gender_survival.min():.1%}")
    
    # Recommendations
    recommendations = [
        "Consider imputing missing values using appropriate methods",
        "Investigate outliers to determine if they are data errors or valid extreme values",
        "Focus on variables with strong correlations for predictive modeling",
        "Consider feature engineering based on identified patterns",
        "Standardize numeric variables if planning to use distance-based algorithms"
    ]
    
    print("Key Insights:")
    for insight in insights:
        print(f"  {insight}")
    
    print(f"\nRecommendations:")
    for rec in recommendations:
        print(f"  {rec}")

def main():
    """Main function to run complete EDA"""
    
    print("🚢 TITANIC DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Missing value analysis
    missing_df = analyze_missing_values(df)
    
    # Statistical summary
    numeric_cols, categorical_cols = statistical_summary(df)
    
    # Outlier detection
    outlier_summary = detect_outliers(df, numeric_cols)
    
    # Distribution analysis
    distribution_analysis(df, numeric_cols)
    
    # Correlation analysis
    correlation_matrix = correlation_analysis(df, numeric_cols)
    
    # Survival analysis (specific to Titanic)
    survival_analysis(df)
    
    # Generate insights
    generate_insights(df, correlation_matrix, outlier_summary)
    
    print(f"\n{'='*80}")
    print("EXPLORATORY DATA ANALYSIS COMPLETED")
    print("All visualizations have been displayed")
    print("Check the analysis results above for insights")
    print("=" * 80)

if __name__ == "__main__":
    main()
