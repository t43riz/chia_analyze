import os
import re
import json
import holidays
import spacy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from itertools import combinations
from collections import Counter
from textblob import TextBlob
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.tsa.seasonal import seasonal_decompose
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
class TimeAnalyzer:
    def __init__(self, content_analyzer):
        """Initialize with ContentAnalyzer instance"""
        self.df = content_analyzer.df.copy()
        # Ensure datetime
        self.df['sent_at'] = pd.to_datetime(self.df['sent_at'])
        # Create additional time features
        self._create_time_features()
        
    def _create_time_features(self):
        """Create additional time-based features for analysis"""
        # Basic time components
        self.df['hour'] = self.df['sent_at'].dt.hour
        self.df['day_of_week'] = self.df['sent_at'].dt.dayofweek
        self.df['month'] = self.df['sent_at'].dt.month
        self.df['quarter'] = self.df['sent_at'].dt.quarter
        self.df['year'] = self.df['sent_at'].dt.year
        self.df['day_name'] = self.df['sent_at'].dt.day_name()
        
        # US Holidays
        us_holidays = holidays.US()
        self.df['is_holiday'] = self.df['sent_at'].dt.date.map(lambda x: x in us_holidays)
        
        # Time of day categories
        time_categories = pd.cut(self.df['hour'], 
                               bins=[0, 6, 12, 17, 24],
                               labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        self.df['time_of_day'] = time_categories

    def analyze_temporal_patterns(self) -> Dict[str, pd.DataFrame]:
        """Analyze performance patterns across different time periods"""
        metrics = ['click_rate', 'open_rate', 'total_revenue']
        temporal_patterns = {}
        
        # Hourly patterns
        hourly = self.df.groupby('hour')[metrics].mean()
        temporal_patterns['hourly'] = hourly
        
        # Daily patterns
        daily = self.df.groupby('day_of_week')[metrics].agg(['mean', 'std', 'count'])
        temporal_patterns['daily'] = daily
        
        # Monthly patterns
        monthly = self.df.groupby('month')[metrics].agg(['mean', 'std', 'count'])
        temporal_patterns['monthly'] = monthly
        
        # Time of day patterns
        time_of_day = self.df.groupby('time_of_day')[metrics].agg(['mean', 'std', 'count'])
        temporal_patterns['time_of_day'] = time_of_day
        
        # Holiday vs non-holiday
        holiday = self.df.groupby('is_holiday')[metrics].agg(['mean', 'std', 'count'])
        temporal_patterns['holiday'] = holiday
        
        return temporal_patterns

    def detect_anomalies(self, window_size: int = 7, threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect anomalies in performance metrics using rolling statistics
        
        Args:
            window_size: Size of the rolling window (in days)
            threshold: Number of standard deviations for anomaly detection
        """
        metrics = ['click_rate', 'open_rate', 'total_revenue']
        
        # Create daily aggregates
        daily_metrics = self.df.groupby(self.df['sent_at'].dt.date)[metrics].mean()
        
        anomalies = pd.DataFrame()
        for metric in metrics:
            # Calculate rolling statistics
            rolling_mean = daily_metrics[metric].rolling(window=window_size).mean()
            rolling_std = daily_metrics[metric].rolling(window=window_size).std()
            
            # Define bounds
            upper_bound = rolling_mean + (threshold * rolling_std)
            lower_bound = rolling_mean - (threshold * rolling_std)
            
            # Detect anomalies
            anomalies[f'{metric}_anomaly'] = (
                (daily_metrics[metric] > upper_bound) | 
                (daily_metrics[metric] < lower_bound)
            )
            
            # Add contextual information
            anomalies[f'{metric}_value'] = daily_metrics[metric]
            anomalies[f'{metric}_mean'] = rolling_mean
            anomalies[f'{metric}_upper'] = upper_bound
            anomalies[f'{metric}_lower'] = lower_bound
        
        return anomalies

    def analyze_seasonality(self, metric: str = 'click_rate', period: int = 7) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components
        
        Args:
            metric: Metric to analyze ('click_rate', 'open_rate', or 'total_revenue')
            period: Number of periods for seasonal decomposition
        """
        # Create daily series
        daily_metric = self.df.groupby(self.df['sent_at'].dt.date)[metric].mean()
        
        # Perform decomposition
        decomposition = seasonal_decompose(daily_metric, period=period)
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }

    def create_temporal_visualizations(self):
        """Create visualizations for temporal patterns"""
        os.makedirs('time_analysis_output', exist_ok=True)
        
        # 1. Daily Performance Heatmap
        daily_hourly = self.df.pivot_table(
            values='click_rate',
            index=self.df['sent_at'].dt.dayofweek,
            columns=self.df['sent_at'].dt.hour,
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=daily_hourly.values,
            x=daily_hourly.columns,
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Click Rate Heatmap by Day and Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week'
        )
        fig.write_html('time_analysis_output/daily_heatmap.html')
        
        # 2. Time Series with Anomalies
        anomalies = self.detect_anomalies()
        metrics = ['click_rate', 'open_rate', 'total_revenue']
        
        for metric in metrics:
            fig = go.Figure()
            
            # Add main metric line
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies[f'{metric}_value'],
                mode='lines',
                name=metric
            ))
            
            # Add anomaly points
            anomaly_dates = anomalies[anomalies[f'{metric}_anomaly']].index
            anomaly_values = anomalies.loc[anomaly_dates, f'{metric}_value']
            
            fig.add_trace(go.Scatter(
                x=anomaly_dates,
                y=anomaly_values,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
            
            fig.update_layout(
                title=f'{metric} Time Series with Anomalies',
                xaxis_title='Date',
                yaxis_title=metric
            )
            
            fig.write_html(f'time_analysis_output/{metric}_anomalies.html')
        
        # 3. Seasonal Decomposition Plots
        for metric in metrics:
            decomposition = self.analyze_seasonality(metric)
            
            fig = make_subplots(rows=4, cols=1, subplot_titles=(
                'Original', 'Trend', 'Seasonal', 'Residual'
            ))
            
            # Original
            fig.add_trace(
                go.Scatter(x=decomposition['trend'].index, 
                          y=decomposition['trend'].values + decomposition['seasonal'].values,
                          name='Original'),
                row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(x=decomposition['trend'].index, 
                          y=decomposition['trend'].values,
                          name='Trend'),
                row=2, col=1
            )
            
            # Seasonal
            fig.add_trace(
                go.Scatter(x=decomposition['seasonal'].index, 
                          y=decomposition['seasonal'].values,
                          name='Seasonal'),
                row=3, col=1
            )
            
            # Residual
            fig.add_trace(
                go.Scatter(x=decomposition['residual'].index, 
                          y=decomposition['residual'].values,
                          name='Residual'),
                row=4, col=1
            )
            
            fig.update_layout(height=900, title=f'{metric} Decomposition')
            fig.write_html(f'time_analysis_output/{metric}_decomposition.html')

    def print_temporal_insights(self):
        """Print insights about temporal patterns"""
        patterns = self.analyze_temporal_patterns()
        
        print("\nTEMPORAL ANALYSIS INSIGHTS")
        
        # Best performing times
        print("\nBest Performing Times:")
        best_hours = patterns['hourly']['click_rate'].nlargest(3)
        print(f"Top hours for click rate: {best_hours.index.tolist()}")
        
        best_days = patterns['daily']['click_rate']['mean'].nlargest(3)
        print(f"Top days for click rate: {best_days.index.tolist()}")
        
        # Holiday impact
        holiday_impact = patterns['holiday']['click_rate']['mean']
        print("\nHoliday Impact:")
        print(f"Holiday click rate: {holiday_impact[True]:.4f}")
        print(f"Non-holiday click rate: {holiday_impact[False]:.4f}")
        
        # Time of day performance
        tod_performance = patterns['time_of_day']['click_rate']['mean']
        print("\nTime of Day Performance:")
        for time, rate in tod_performance.items():
            print(f"{time}: {rate:.4f}")
            
class CampaignAnalyzer:
    def __init__(self, lgd_path: str, chia_path: str):
        """Initialize with paths to both CSV files"""
        print("Loading LGD campaigns...")
        self.lgd_df = pd.read_csv(lgd_path)
        print("LGD columns:", self.lgd_df.columns.tolist())
        
        print("\nLoading CHIA campaigns...")
        self.chia_df = pd.read_csv(chia_path, low_memory=False)
        print("CHIA columns:", self.chia_df.columns.tolist())
        
        # Merge datasets
        print("\nMerging datasets...")
        self.merged_df = self.merge_datasets()
        print("Available metrics:", self.merged_df.columns.tolist())

    def merge_datasets(self) -> pd.DataFrame:
        """Merge LGD and CHIA datasets"""
        merged = self.chia_df.merge(
            self.lgd_df,
            left_on='id',
            right_on='campaign_id',
            how='inner'
        )
        print(f"Merged dataset size: {len(merged)} rows")
        return merged

    def analyze_by_theme(self) -> pd.DataFrame:
        """Analyze performance metrics by theme"""
        theme_analysis = self.merged_df.groupby('theme_level1').agg({
            'open_rate': ['mean', 'median', 'std', 'count'],
            'click_rate': ['mean', 'median', 'std'],
            'total_revenue': ['mean', 'sum'],
            'unique_clicks': ['mean', 'sum']
        }).round(4)
        
        # Add derived metrics
        theme_analysis['revenue_per_click'] = (
            theme_analysis[('total_revenue', 'sum')] / 
            theme_analysis[('unique_clicks', 'sum')]
        ).round(4)
        
        print("\nTheme Analysis:")
        print(theme_analysis)
        return theme_analysis

    def analyze_by_account(self) -> pd.DataFrame:
        """Analyze performance metrics by account"""
        account_analysis = self.merged_df.groupby('account_name').agg({
            'open_rate': ['mean', 'median', 'std'],
            'click_rate': ['mean', 'median', 'std'],
            'total_revenue': ['mean', 'sum'],
            'unique_clicks': ['sum'],
            'id': 'count'
        }).round(4)
        
        # Calculate revenue per click
        account_analysis['revenue_per_click'] = (
            account_analysis[('total_revenue', 'sum')] / 
            account_analysis[('unique_clicks', 'sum')]
        ).round(4)
        
        print("\nAccount Analysis:")
        print(account_analysis)
        return account_analysis

    def analyze_performance_metrics(self):
        """Analyze key performance metrics"""
        metrics = ['open_rate', 'click_rate', 'total_revenue', 'unique_clicks']
        
        # Calculate correlations
        correlations = self.merged_df[metrics].corr()
        
        # Calculate summary statistics
        summary_stats = self.merged_df[metrics].describe()
        
        # Calculate revenue per click overall
        total_revenue = self.merged_df['total_revenue'].sum()
        total_clicks = self.merged_df['unique_clicks'].sum()
        revenue_per_click = total_revenue / total_clicks if total_clicks > 0 else 0
        
        print("\nOverall Metrics:")
        print(f"Total Revenue: ${total_revenue:,.2f}")
        print(f"Total Clicks: {total_clicks:,}")
        print(f"Revenue per Click: ${revenue_per_click:.4f}")
        
        print("\nMetric Correlations:")
        print(correlations.round(3))
        
        print("\nMetric Summary Statistics:")
        print(summary_stats.round(3))
        
        return {
            'correlations': correlations,
            'summary_stats': summary_stats,
            'overall_metrics': {
                'total_revenue': total_revenue,
                'total_clicks': total_clicks,
                'revenue_per_click': revenue_per_click
            }
        }

    def plot_visualizations(self):
        """Create and save visualizations"""
        os.makedirs('analysis_output', exist_ok=True)
        
        # Theme performance
        theme_metrics = self.merged_df.groupby('theme_level1').agg({
            'total_revenue': 'mean',
            'click_rate': 'mean',
            'unique_clicks': 'sum'
        }).round(4)
        
        fig = px.bar(
            theme_metrics.reset_index(),
            x='theme_level1',
            y='total_revenue',
            title='Average Revenue by Theme',
            color='click_rate',
            hover_data=['unique_clicks']
        )
        fig.write_html('analysis_output/theme_revenue.html')
        
        # Account performance scatter
        account_metrics = self.merged_df.groupby('account_name').agg({
            'total_revenue': 'mean',
            'click_rate': 'mean',
            'open_rate': 'mean',
            'unique_clicks': 'sum'
        }).reset_index()
        
        fig = px.scatter(
            account_metrics,
            x='click_rate',
            y='total_revenue',
            size='unique_clicks',
            color='open_rate',
            hover_data=['account_name'],
            title='Account Performance Matrix'
        )
        fig.write_html('analysis_output/account_performance.html')

class ContentAnalyzer:
    """Analyzes body and subject content patterns for performance"""
    
    def __init__(self, lgd_path: str, chia_path: str):
        print("Loading datasets...")
        self.lgd_df = pd.read_csv(lgd_path)
        self.chia_df = pd.read_csv(chia_path, low_memory=False)
        
        # Filter for DRAFT_SUCCESS and merge
        self.chia_df = self.chia_df[self.chia_df['draft_status'] == 'DRAFT_SUCCESS']
        self.df = self.merge_data()
        
        # Parse body content
        self.df['parsed_body'] = self.df.apply(self.parse_body_content, axis=1)
        
    def merge_data(self) -> pd.DataFrame:
        """Merge and prepare datasets with detailed date error diagnosis"""
        # Merge datasets
        df = self.chia_df.merge(
            self.lgd_df,
            left_on='id',
            right_on='campaign_id',
            how='inner'
        )
        
        # First check for null values
        null_dates = df['sent_at'].isna().sum()
        
        # Analyze date issues
        def diagnose_date(date_str):
            if pd.isna(date_str):
                return 'null'
            try:
                pd.to_datetime(date_str)
                return 'valid'
            except pd.errors.OutOfBoundsDatetime:
                return 'out_of_bounds'
            except ValueError as e:
                if 'unconverted data remains' in str(e):
                    return 'bad_format'
                else:
                    return 'other_error'

        # Diagnose each date
        df['date_status'] = df['sent_at'].apply(diagnose_date)
        
        # Get counts for each type of issue
        issue_counts = df['date_status'].value_counts()
        
        print("\nDate Quality Analysis:")
        print("-----------------------")
        for status, count in issue_counts.items():
            if status != 'valid':
                print(f"Found {count} dates with issue: {status}")
                # Show some examples of problematic dates
                examples = df[df['date_status'] == status]['sent_at'].head(3).tolist()
                print(f"Example values: {examples}\n")

        # Get sample of out of bounds dates to show their values
        out_of_bounds = df[df['date_status'] == 'out_of_bounds']
        if not out_of_bounds.empty:
            print("Sample of out of bounds dates:")
            for _, row in out_of_bounds.head(3).iterrows():
                print(f"Value: {row['sent_at']}")
            print()

        # Now convert dates with coerce option
        df['sent_at'] = pd.to_datetime(df['sent_at'], errors='coerce')
        
        # Keep only valid dates
        original_size = len(df)
        df = df[df['date_status'] == 'valid']
        rows_removed = original_size - len(df)
        
        print(f"\nSummary:")
        print(f"Total rows initially: {original_size}")
        print(f"Rows with valid dates: {len(df)}")
        print(f"Total rows removed: {rows_removed}")
        
        # Drop the diagnostic column as it's no longer needed
        df = df.drop('date_status', axis=1)
        
        return df
    def parse_body_content(self, row) -> str:
        """Parse body content from body_data and data_tokens"""
        try:
            body_data = json.loads(row['body_data'])
            data_tokens = json.loads(row['data_tokens'])
            return '\n\n'.join(body_data.get(token, '') for token in data_tokens)
        except:
            return ''

class EnhancedContentAnalyzer:
    def __init__(self, content_analyzer):
        """Initialize with existing ContentAnalyzer instance"""
        self.df = content_analyzer.df
        self.nlp = spacy.load('en_core_web_sm')
        
    def analyze_subject_linguistics(self):
        """Analyze linguistic patterns in subject lines"""
        subjects = self.df[['creative_subject_id', 'subject', 'open_rate', 'click_rate']].drop_duplicates()
        
        # Enhanced pattern analysis
        subjects['word_count'] = subjects['subject'].str.split().str.len()
        subjects['char_count'] = subjects['subject'].str.len()
        subjects['avg_word_length'] = subjects['char_count'] / subjects['word_count']
        
        # Sentiment analysis
        subjects['sentiment'] = subjects['subject'].apply(lambda x: TextBlob(x).sentiment.polarity)
        subjects['subjectivity'] = subjects['subject'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        
        # POS patterns using spaCy
        def get_pos_patterns(text):
            doc = self.nlp(text)
            return {
                'num_verbs': len([token for token in doc if token.pos_ == 'VERB']),
                'num_nouns': len([token for token in doc if token.pos_ == 'NOUN']),
                'num_adj': len([token for token in doc if token.pos_ == 'ADJ']),
                'has_number': any(token.like_num for token in doc),
                'first_word_pos': doc[0].pos_ if len(doc) > 0 else None
            }
        
        pos_patterns = pd.DataFrame(subjects['subject'].apply(get_pos_patterns).tolist())
        subjects = pd.concat([subjects, pos_patterns], axis=1)
        
        # Analyze performance by pattern
        pattern_cols = ['word_count', 'avg_word_length', 'sentiment', 'subjectivity', 
                       'num_verbs', 'num_nouns', 'num_adj']
        
        performance_by_pattern = {}
        for col in pattern_cols:
            try:
                # First try standard quartile binning
                subjects[f'{col}_bin'] = pd.qcut(subjects[col], 
                                               q=4, 
                                               labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                               duplicates='drop')
            except ValueError as e:
                print(f"\nNotice: Alternative binning strategy used for {col} due to value distribution")
                # If that fails, use an alternative approach
                # First check the unique value count
                unique_values = subjects[col].nunique()
                if unique_values <= 4:
                    # If very few unique values, use them directly
                    subjects[f'{col}_bin'] = subjects[col].astype(str)
                else:
                    # Use regular cut with equal-width bins instead
                    subjects[f'{col}_bin'] = pd.cut(subjects[col],
                                                  bins=4,
                                                  labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            # Calculate performance metrics by bin
            performance = subjects.groupby(f'{col}_bin', observed=True).agg({
                'open_rate': ['mean', 'std', 'count'],
                'click_rate': ['mean', 'std']
            }).round(4)
            
            # Add value ranges for context
            value_ranges = subjects.groupby(f'{col}_bin', observed=True)[col].agg(['min', 'max'])
            performance[('value_range', '')] = value_ranges.apply(
                lambda x: f"{x['min']:.2f} to {x['max']:.2f}", axis=1
            )
            
            performance_by_pattern[col] = performance
            
        return {
            'detailed_metrics': subjects,
            'performance_by_pattern': performance_by_pattern
        }
    def extract_topics(self, n_topics=5):
        """Extract topics from body content using NMF"""
        # Prepare text data
        bodies = self.df[['creative_body_id', 'parsed_body', 'click_rate']].drop_duplicates()
        
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        text_features = tfidf.fit_transform(bodies['parsed_body'])
        
        # Extract topics
        nmf = NMF(n_components=n_topics, random_state=42)
        topic_features = nmf.fit_transform(text_features)
        
        # Get top words for each topic
        feature_names = tfidf.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(nmf.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topics[f'topic_{topic_idx}'] = top_words
            
            # Add topic scores to bodies DataFrame
            bodies[f'topic_{topic_idx}_score'] = topic_features[:, topic_idx]
        
        # Analyze performance by topic dominance
        topic_performance = {}
        for i in range(n_topics):
            topic_col = f'topic_{i}_score'
            try:
                # First try standard quartile binning with duplicate handling
                bodies[f'{topic_col}_bin'] = pd.qcut(bodies[topic_col], 
                                                   q=4, 
                                                   labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                                   duplicates='drop')
            except ValueError as e:
                print(f"\nNotice: Alternative binning strategy used for {topic_col} due to value distribution")
                # If that fails, check unique values
                unique_values = bodies[topic_col].nunique()
                if unique_values <= 4:
                    # For very few unique values, use them directly
                    bodies[f'{topic_col}_bin'] = bodies[topic_col].astype(str)
                else:
                    # Use regular cut with equal-width bins
                    bodies[f'{topic_col}_bin'] = pd.cut(bodies[topic_col],
                                                      bins=4,
                                                      labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            # Calculate performance metrics
            performance = bodies.groupby(f'{topic_col}_bin', observed=True).agg({
                'click_rate': ['mean', 'std', 'count']
            }).round(4)
            
            # Add value ranges for context
            value_ranges = bodies.groupby(f'{topic_col}_bin', observed=True)[topic_col].agg(['min', 'max'])
            performance[('value_range', '')] = value_ranges.apply(
                lambda x: f"{x['min']:.4f} to {x['max']:.4f}", axis=1
            )
            
            topic_performance[f'topic_{i}'] = performance
        
        # Add diagnostic information about topic distributions
        topic_stats = {}
        for i in range(n_topics):
            topic_col = f'topic_{i}_score'
            stats = {
                'mean': bodies[topic_col].mean(),
                'std': bodies[topic_col].std(),
                'unique_values': bodies[topic_col].nunique(),
                'zero_values': (bodies[topic_col] == 0).sum(),
                'non_zero_values': (bodies[topic_col] > 0).sum()
            }
            topic_stats[f'topic_{i}'] = stats
        
        return {
            'topics': topics,
            'topic_performance': topic_performance,
            'body_topics': bodies,
            'topic_statistics': topic_stats
        }
    
    def analyze_email_structure(self):
        """Analyze structural elements of email bodies"""
        bodies = self.df[['creative_body_id', 'parsed_body', 'click_rate']].drop_duplicates()
        
        def extract_structure(text):
            return {
                'num_paragraphs': len(re.split(r'\n\n+', text)),
                'avg_paragraph_length': np.mean([len(p) for p in re.split(r'\n\n+', text)]),
                'num_links': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
                'has_bullet_points': bool(re.search(r'[•\-\*]', text)),
                'num_calls_to_action': len(re.findall(r'click|sign up|register|buy|shop|learn more|discover', text, re.IGNORECASE)),
                'has_price': bool(re.search(r'\$\d+', text)),
                'has_percentage': bool(re.search(r'\d+%', text)),
                'reading_time': len(text.split()) / 200  # Assuming 200 words per minute reading speed
            }
        
        structure_metrics = pd.DataFrame(bodies['parsed_body'].apply(extract_structure).tolist())
        bodies = pd.concat([bodies, structure_metrics], axis=1)
        
        # Analyze performance by structural elements
        structure_cols = structure_metrics.columns
        structure_performance = {}
        
        for col in structure_cols:
            if bodies[col].dtype in ['int64', 'float64']:
                try:
                    # First try standard quartile binning with duplicate handling
                    bodies[f'{col}_bin'] = pd.qcut(bodies[col], 
                                                q=4, 
                                                labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                                duplicates='drop')
                except ValueError as e:
                    print(f"\nNotice: Alternative binning strategy used for {col} due to value distribution")
                    # If that fails, check unique values
                    unique_values = bodies[col].nunique()
                    if unique_values <= 4:
                        # For very few unique values, use them directly
                        bodies[f'{col}_bin'] = bodies[col].astype(str)
                    else:
                        # Use regular cut with equal-width bins
                        bodies[f'{col}_bin'] = pd.cut(bodies[col],
                                                   bins=4,
                                                   labels=['Q1', 'Q2', 'Q3', 'Q4'])
                group_col = f'{col}_bin'
            else:
                # For boolean or categorical columns, use as is
                group_col = col
            
            # Calculate performance metrics
            performance = bodies.groupby(group_col, observed=True).agg({
                'click_rate': ['mean', 'std', 'count']
            }).round(4)
            
            # Add value ranges for numeric columns
            if bodies[col].dtype in ['int64', 'float64']:
                value_ranges = bodies.groupby(group_col, observed=True)[col].agg(['min', 'max'])
                performance[('value_range', '')] = value_ranges.apply(
                    lambda x: f"{x['min']:.2f} to {x['max']:.2f}", axis=1
                )
            
            structure_performance[col] = performance
        
        # Add distribution statistics
        structure_stats = {}
        for col in structure_cols:
            if bodies[col].dtype in ['int64', 'float64']:
                stats = {
                    'mean': bodies[col].mean(),
                    'std': bodies[col].std(),
                    'unique_values': bodies[col].nunique(),
                    'most_common_values': bodies[col].value_counts().head(3).to_dict()
                }
            else:
                stats = {
                    'value_counts': bodies[col].value_counts().to_dict()
                }
            structure_stats[col] = stats
        
        return {
            'detailed_metrics': bodies,
            'structure_performance': structure_performance,
            'structure_statistics': structure_stats
        }
class TopPerformerAnalyzer:
    def __init__(self, content_analyzer, percentile_threshold=90):
        """
        Initialize with ContentAnalyzer instance and performance threshold
        percentile_threshold: percentile to consider as top performers (e.g., 90 for top 10%)
        """
        self.df = content_analyzer.df
        self.percentile_threshold = percentile_threshold
        self.nlp = spacy.load('en_core_web_sm')
        
    def identify_top_performers(self):
        """Identify top performing content based on multiple metrics"""
        # Calculate performance thresholds
        click_threshold = np.percentile(self.df['click_rate'], self.percentile_threshold)
        open_threshold = np.percentile(self.df['open_rate'], self.percentile_threshold)
        revenue_threshold = np.percentile(self.df['total_revenue'], self.percentile_threshold)
        
        # Identify top performers by different metrics
        top_performers = {
            'clicks': self.df[self.df['click_rate'] >= click_threshold],
            'opens': self.df[self.df['open_rate'] >= open_threshold],
            'revenue': self.df[self.df['total_revenue'] >= revenue_threshold],
            'consistent': self.df[
                (self.df['click_rate'] >= click_threshold) &
                (self.df['open_rate'] >= open_threshold) &
                (self.df['total_revenue'] >= revenue_threshold)
            ]
        }
        
        return top_performers

    def analyze_phrase_patterns(self, text_series):
        """Analyze common phrases and their contexts"""
        # Extract common bigrams and trigrams
        def get_ngrams(text, n):
            tokens = text.lower().split()
            return list(zip(*[tokens[i:] for i in range(n)]))
            
        all_bigrams = []
        all_trigrams = []
        
        for text in text_series:
            all_bigrams.extend(get_ngrams(text, 2))
            all_trigrams.extend(get_ngrams(text, 3))
        
        # Count and sort by frequency
        bigram_freq = Counter(all_bigrams).most_common(20)
        trigram_freq = Counter(all_trigrams).most_common(20)
        
        return {
            'common_bigrams': bigram_freq,
            'common_trigrams': trigram_freq
        }

    def find_content_clusters(self, text_series, n_clusters=5):
        """Find clusters of similar content using text features"""
        # Check if we have any data
        if text_series.empty:
            print("\nWarning: No text data available for clustering")
            return {
                'cluster_assignments': [],
                'cluster_characteristics': pd.DataFrame()
            }

        # Remove any null values
        text_series = text_series.dropna()
        if len(text_series) == 0:
            print("\nWarning: No valid text data after removing null values")
            return {
                'cluster_assignments': [],
                'cluster_characteristics': pd.DataFrame()
            }

        print(f"\nProcessing {len(text_series)} texts for clustering...")
            
        # Create text features
        def extract_text_features(text):
            try:
                doc = self.nlp(str(text))
                blob = TextBlob(str(text))
                
                return {
                    'word_count': len(text.split()),
                    'avg_word_length': sum(len(word) for word in text.split()) / (len(text.split()) or 1),
                    'sentiment': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity,
                    'num_entities': len(doc.ents),
                    'num_verbs': len([token for token in doc if token.pos_ == 'VERB']),
                    'num_nouns': len([token for token in doc if token.pos_ == 'NOUN']),
                    'num_adj': len([token for token in doc if token.pos_ == 'ADJ'])
                }
            except Exception as e:
                print(f"\nWarning: Error processing text: {str(e)}")
                return None
            
        # Extract features for clustering
        features_list = []
        for text in text_series:
            features = extract_text_features(text)
            if features is not None:
                features_list.append(features)
            
        if not features_list:
            print("\nWarning: No valid features extracted for clustering")
            return {
                'cluster_assignments': [],
                'cluster_characteristics': pd.DataFrame()
            }
            
        features_df = pd.DataFrame(features_list)
        
        # Verify we have enough data for the requested number of clusters
        n_clusters = min(n_clusters, len(features_df))
        if n_clusters < 2:
            print("\nWarning: Not enough data for meaningful clustering")
            return {
                'cluster_assignments': [],
                'cluster_characteristics': features_df.mean().to_frame().T
            }
        
        # Scale features and perform clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Analyze cluster characteristics
        features_df['cluster'] = clusters
        cluster_analysis = features_df.groupby('cluster').mean()
        
        print(f"\nClustering complete: Found {n_clusters} clusters")
        print("Cluster sizes:")
        print(features_df['cluster'].value_counts().sort_index())
        
        return {
            'cluster_assignments': clusters,
            'cluster_characteristics': cluster_analysis,
            'feature_stats': {
                'total_texts': len(text_series),
                'processed_texts': len(features_list),
                'features_per_cluster': features_df['cluster'].value_counts().to_dict()
            }
        }
    def analyze_structure_patterns(self, top_performers):
        """Analyze structural patterns in top performing content"""
        def extract_structure_features(text):
            paragraphs = text.split('\n\n')
            sentences = [sent.text.strip() for sent in self.nlp(text).sents]
            
            return {
                'num_paragraphs': len(paragraphs),
                'avg_paragraph_length': np.mean([len(p.split()) for p in paragraphs]),
                'num_sentences': len(sentences),
                'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences]),
                'has_list': bool(re.search(r'^\s*[-•*]\s', text, re.MULTILINE)),
                'has_numbers': bool(re.search(r'\d+', text)),
                'has_question': bool(re.search(r'\?', text)),
                'has_exclamation': bool(re.search(r'!', text)),
                'has_price': bool(re.search(r'\$\d+', text)),
                'link_count': len(re.findall(r'http[s]?://', text)),
                'has_caps_words': bool(re.search(r'\b[A-Z]{2,}\b', text))
            }
            
        # Analyze structure for different types of top performers
        structure_patterns = {}
        for perf_type, df in top_performers.items():
            structures = []
            for text in df['parsed_body'].unique():
                structures.append(extract_structure_features(text))
            
            structure_patterns[perf_type] = pd.DataFrame(structures).mean()
            
        return structure_patterns

    def find_common_elements(self, top_performers):
        """Find elements that frequently appear together in top performing content"""
        def extract_elements(text):
            return {
                'has_price': bool(re.search(r'\$\d+', text)),
                'has_percentage': bool(re.search(r'\d+%', text)),
                'has_date': bool(re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}\b', text)),
                'has_cta': bool(re.search(r'\b(?:click|sign up|register|buy|shop|learn more|get started)\b', text, re.IGNORECASE)),
                'has_urgency': bool(re.search(r'\b(?:limited time|hurry|ends soon|last chance|don\'t miss|only)\b', text, re.IGNORECASE)),
                'has_personalization': bool(re.search(r'\b(?:you|your|you\'ll|you\'re)\b', text, re.IGNORECASE)),
                'has_benefit': bool(re.search(r'\b(?:free|save|discount|offer|deal|exclusive)\b', text, re.IGNORECASE)),
                'has_social_proof': bool(re.search(r'\b(?:popular|bestselling|trending|recommended|loved)\b', text, re.IGNORECASE))
            }
            
        # Create element co-occurrence network
        G = nx.Graph()
        
        for perf_type, df in top_performers.items():
            for text in df['parsed_body'].unique():
                elements = extract_elements(text)
                present_elements = [k for k, v in elements.items() if v]
                
                # Add edges between co-occurring elements
                for elem1, elem2 in combinations(present_elements, 2):
                    if G.has_edge(elem1, elem2):
                        G[elem1][elem2]['weight'] += 1
                    else:
                        G.add_edge(elem1, elem2, weight=1)
        
        return {
            'network': G,
            'common_pairs': sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        }

    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report of top performers"""
        # Get top performers
        top_performers = self.identify_top_performers()
        
        if not top_performers['consistent'].empty:
            # Analyze patterns
            phrase_patterns = self.analyze_phrase_patterns(top_performers['consistent']['subject'])
            content_clusters = self.find_content_clusters(top_performers['consistent']['parsed_body'])
            structure_patterns = self.analyze_structure_patterns(top_performers)
            common_elements = self.find_common_elements(top_performers)
            
            # Compile insights
            insights = {
                'performance_metrics': {
                    perf_type: {
                        'count': len(df),
                        'avg_click_rate': df['click_rate'].mean(),
                        'avg_open_rate': df['open_rate'].mean(),
                        'avg_revenue': df['total_revenue'].mean()
                    } for perf_type, df in top_performers.items()
                },
                'common_phrases': phrase_patterns,
                'content_clusters': content_clusters,
                'structure_patterns': structure_patterns,
                'element_combinations': common_elements
            }
        else:
            print("\nWarning: No consistent top performers found")
            insights = {
                'performance_metrics': {},
                'common_phrases': {'common_bigrams': [], 'common_trigrams': []},
                'content_clusters': {'cluster_assignments': [], 'cluster_characteristics': pd.DataFrame()},
                'structure_patterns': {},
                'element_combinations': {'network': nx.Graph(), 'common_pairs': []}
            }
        
        return insights
    def print_actionable_insights(self, insights):
        """Print actionable insights in a readable format"""
        print("\nTOP PERFORMER ANALYSIS INSIGHTS")
        print("\n1. Performance Metrics:")
        for perf_type, metrics in insights['performance_metrics'].items():
            print(f"\n{perf_type.title()} Performers:")
            print(f"- Count: {metrics['count']}")
            print(f"- Avg Click Rate: {metrics['avg_click_rate']:.2%}")
            print(f"- Avg Open Rate: {metrics['avg_open_rate']:.2%}")
            print(f"- Avg Revenue: ${metrics['avg_revenue']:.2f}")
        
        print("\n2. Common Phrases:")
        print("\nTop Bigrams:")
        for bigram, count in insights['common_phrases']['common_bigrams'][:5]:
            print(f"- {' '.join(bigram)}: {count} occurrences")
            
        print("\n3. Content Clusters:")
        for cluster, chars in insights['content_clusters']['cluster_characteristics'].iterrows():
            print(f"\nCluster {cluster} characteristics:")
            for feature, value in chars.items():
                print(f"- {feature}: {value:.2f}")
        
        print("\n4. Common Element Combinations:")
        for elem1, elem2, data in insights['element_combinations']['common_pairs'][:5]:
            print(f"- {elem1} + {elem2}: {data['weight']} occurrences")

def main():
    """Main execution function"""
    # Initialize all analyzers
    campaign_analyzer = CampaignAnalyzer('lgd_campaigns.csv', 'chia_campaigns.csv')
    content_analyzer = ContentAnalyzer('lgd_campaigns.csv', 'chia_campaigns.csv')
    enhanced_analyzer = EnhancedContentAnalyzer(content_analyzer)
    top_performer_analyzer = TopPerformerAnalyzer(content_analyzer)
    time_analyzer = TimeAnalyzer(content_analyzer)

    # Generate comprehensive insights
    print("\nGenerating Campaign Performance Analysis...")
    campaign_metrics = campaign_analyzer.analyze_performance_metrics()
    campaign_analyzer.plot_visualizations()

    print("\nGenerating Enhanced Content Analysis...")
    subject_analysis = enhanced_analyzer.analyze_subject_linguistics()
    topic_analysis = enhanced_analyzer.extract_topics()
    structure_analysis = enhanced_analyzer.analyze_email_structure()

    print("\nGenerating Top Performer Analysis...")
    top_performer_insights = top_performer_analyzer.generate_comprehensive_report()
    top_performer_analyzer.print_actionable_insights(top_performer_insights)

    print("\nGenerating Temporal Analysis...")
    temporal_patterns = time_analyzer.analyze_temporal_patterns()
    time_analyzer.create_temporal_visualizations()
    time_analyzer.print_temporal_insights()
    
    # Generate anomaly detection and seasonality analysis
    anomalies = time_analyzer.detect_anomalies()
    seasonality_analysis = time_analyzer.analyze_seasonality()

    # Save results
    os.makedirs('analysis_results', exist_ok=True)
    
    # Create summary report
    with open('analysis_results/summary_report.txt', 'w') as f:
        f.write("Campaign Analysis Summary Report\n")
        f.write("==============================\n\n")
        
        # Overall Performance Metrics
        f.write("Overall Performance Metrics:\n")
        f.write(f"Total Revenue: ${campaign_metrics['overall_metrics']['total_revenue']:,.2f}\n")
        f.write(f"Total Clicks: {campaign_metrics['overall_metrics']['total_clicks']:,}\n")
        f.write(f"Revenue per Click: ${campaign_metrics['overall_metrics']['revenue_per_click']:.4f}\n\n")
        
        # Top Performing Content Characteristics
        f.write("Top Performing Content Characteristics:\n")
        for cluster, chars in top_performer_insights['content_clusters']['cluster_characteristics'].iterrows():
            f.write(f"\nContent Cluster {cluster}:\n")
            for feature, value in chars.items():
                f.write(f"- {feature}: {value:.2f}\n")

        # Temporal Insights
        f.write("\nTemporal Performance Insights:\n")
        best_hours = temporal_patterns['hourly']['click_rate'].nlargest(3)
        f.write(f"Top performing hours: {best_hours.index.tolist()}\n")
        
        # Holiday Performance
        holiday_impact = temporal_patterns['holiday']['click_rate']['mean']
        f.write("\nHoliday vs Non-Holiday Performance:\n")
        f.write(f"Holiday click rate: {holiday_impact[True]:.4f}\n")
        f.write(f"Non-holiday click rate: {holiday_impact[False]:.4f}\n")
        
        # Anomalies Summary
        f.write("\nPerformance Anomalies:\n")
        anomaly_dates = anomalies[anomalies['click_rate_anomaly']].index
        f.write(f"Number of anomalous days: {len(anomaly_dates)}\n")
        if len(anomaly_dates) > 0:
            f.write("Notable anomaly dates:\n")
            for date in list(anomaly_dates)[:5]:
                f.write(f"- {date.strftime('%Y-%m-%d')}\n")
        
        # Seasonality Insights
        f.write("\nSeasonality Analysis:\n")
        f.write("Seasonal patterns detected in:\n")
        for metric in ['click_rate', 'open_rate', 'total_revenue']:
            seasonal = seasonality_analysis.get(metric, {}).get('seasonal')
            if seasonal is not None:
                f.write(f"- {metric}: {seasonal.std():.4f} standard deviation\n")

    print("\nAnalysis complete! Results saved in 'analysis_results' directory.")
    print("Visualizations saved in:")
    print("- 'analysis_output': Campaign visualizations")
    print("- 'time_analysis_output': Temporal analysis visualizations")

if __name__ == "__main__":
    main()
