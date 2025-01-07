import os
import re
import json
import holidays
import spacy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any  # Add this import
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

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from itertools import combinations
import spacy
import os
import plotly.graph_objects as go


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
    def __init__(self, content_analyzer, percentile_thresholds=None):
        """
        Initialize with ContentAnalyzer instance and configurable thresholds
        percentile_thresholds: dict with thresholds for each metric (e.g., {'clicks': 80, 'opens': 75, 'revenue': 70})
        """
        self.df = content_analyzer.df
        self.percentile_thresholds = percentile_thresholds or {
            'clicks': 80,
            'opens': 75,
            'revenue': 70
        }
        self.nlp = spacy.load('en_core_web_sm')
        self.top_performers = {}
        
    def identify_top_performers(self):
        """Identify top performing content based on individual metrics"""
        # First clean the dataframe
        clean_df = self.df.copy()
        
        # Remove rows where all key metrics are NaN
        clean_df = clean_df.dropna(subset=['click_rate', 'open_rate', 'total_revenue'], how='all')
        
        print("\nData quality check:")
        print(f"Original rows: {len(self.df)}")
        print(f"Clean rows: {len(clean_df)}")
        
        # Print summary statistics for verification
        print("\nClean data summary:")
        print(clean_df[['click_rate', 'open_rate', 'total_revenue']].describe())
        
        metric_configs = {
            'clicks': {
                'metric': 'click_rate',
                'threshold': self.percentile_thresholds['clicks']
            },
            'opens': {
                'metric': 'open_rate',
                'threshold': self.percentile_thresholds['opens']
            },
            'revenue': {
                'metric': 'total_revenue',
                'threshold': self.percentile_thresholds['revenue']
            }
        }
        
        # Store thresholds and campaigns for each metric
        performance_data = {}
        
        for category, config in metric_configs.items():
            metric = config['metric']
            percentile = config['threshold']
            
            # Calculate threshold on non-NaN values
            valid_values = clean_df[metric].dropna()
            if len(valid_values) > 0:
                threshold = np.percentile(valid_values, percentile)
                
                # Identify top performers (avoiding NaN comparisons)
                top_campaigns = clean_df[clean_df[metric].notna() & (clean_df[metric] >= threshold)].copy()
            else:
                print(f"Warning: No valid data for {metric}")
                threshold = np.nan
                top_campaigns = pd.DataFrame()
            
            performance_data[category] = {
                'threshold': threshold,
                'campaigns': top_campaigns,
                'metric_name': metric,
                'count': len(top_campaigns),
                'mean_performance': top_campaigns[metric].mean()
            }
            
            print(f"\nAnalysis for {category}:")
            print(f"Threshold ({percentile}th percentile): {threshold:.4f}")
            print(f"Number of top campaigns: {len(top_campaigns)}")
            print(f"Mean {metric}: {top_campaigns[metric].mean():.4f}")
        
        # Find overlapping top performers
        all_metrics = set(metric_configs.keys())
        for r in range(2, len(all_metrics) + 1):
            for metrics_combo in combinations(all_metrics, r):
                combo_name = '_and_'.join(metrics_combo)
                
                # Find campaigns that are top performers in all selected metrics
                combined_mask = pd.Series(True, index=self.df.index)
                for metric in metrics_combo:
                    config = metric_configs[metric]
                    metric_name = config['metric']
                    threshold = performance_data[metric]['threshold']
                    combined_mask &= (self.df[metric_name] >= threshold)
                
                overlap_campaigns = self.df[combined_mask].copy()
                
                performance_data[combo_name] = {
                    'campaigns': overlap_campaigns,
                    'count': len(overlap_campaigns),
                    'metrics_included': metrics_combo
                }
                
                print(f"\nOverlap Analysis for {combo_name}:")
                print(f"Number of campaigns excelling in all metrics: {len(overlap_campaigns)}")
                
        self.top_performers = performance_data
        return performance_data

    def analyze_content_patterns(self, category):
        """Analyze content patterns for a specific category of top performers"""
        if category not in self.top_performers:
            print(f"No data available for category: {category}")
            return None
            
        campaigns = self.top_performers[category]['campaigns']
        if len(campaigns) == 0:
            print(f"No campaigns found for category: {category}")
            return None
            
        # Print data quality information
        print(f"\nAnalyzing {category}:")
        print(f"Number of campaigns: {len(campaigns)}")
        print("Available metrics:", campaigns.columns.tolist())
            
        campaigns = self.top_performers[category]['campaigns']
        
        # Analyze subjects
        subjects = campaigns['subject'].dropna()
        
        # Basic text stats
        subject_stats = {
            'avg_length': subjects.str.len().mean(),
            'word_count': subjects.str.split().str.len().mean(),
            'has_number': subjects.str.contains(r'\d').mean(),
            'has_question': subjects.str.contains(r'\?').mean(),
            'has_exclamation': subjects.str.contains(r'!').mean(),
        }
        
        # Common words and phrases
        all_words = ' '.join(subjects).lower().split()
        word_freq = Counter(all_words).most_common(10)
        
        # Analyze body content
        bodies = campaigns['parsed_body'].dropna()
        
        # Extract key content features
        def extract_content_features(text):
            return {
                'length': len(text),
                'paragraphs': len(text.split('\n\n')),
                'has_list': bool(re.search(r'^\s*[-•*]\s', text, re.MULTILINE)),
                'has_price': bool(re.search(r'\$\d+', text)),
                'has_date': bool(re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}\b', text)),
                'has_cta': bool(re.search(r'\b(?:click|sign up|register|buy|shop|learn more|get started)\b', text, re.IGNORECASE)),
                'link_count': len(re.findall(r'http[s]?://', text))
            }
        
        content_features = [extract_content_features(text) for text in bodies]
        feature_stats = pd.DataFrame(content_features).mean()
        
        # Get performance metrics for this category
        performance_metrics = {
            'click_rate': campaigns['click_rate'].mean(),
            'open_rate': campaigns['open_rate'].mean(),
            'total_revenue': campaigns['total_revenue'].mean(),
            'campaign_count': len(campaigns)
        }
        
        return {
            'subject_stats': subject_stats,
            'common_words': word_freq,
            'content_features': feature_stats,
            'performance_metrics': performance_metrics
        }

    def generate_insights_report(self):
        """Generate comprehensive insights about top performers across categories"""
        if not self.top_performers:
            self.identify_top_performers()
        
        report = []
        
        # Analyze each category
        for category in self.top_performers.keys():
            insights = self.analyze_content_patterns(category)
            if insights:
                report.append(f"\nInsights for {category} Top Performers:")
                report.append(f"Number of Campaigns: {insights['performance_metrics']['campaign_count']}")
                
                # Performance metrics
                report.append("\nPerformance Metrics:")
                for metric, value in insights['performance_metrics'].items():
                    if metric != 'campaign_count':
                        report.append(f"- Average {metric}: {value:.4f}")
                
                # Subject line patterns
                report.append("\nSubject Line Patterns:")
                for metric, value in insights['subject_stats'].items():
                    report.append(f"- {metric}: {value:.2f}")
                
                report.append("\nMost Common Words in Subject Lines:")
                for word, count in insights['common_words']:
                    report.append(f"- {word}: {count} occurrences")
                
                # Content features
                report.append("\nContent Characteristics:")
                for feature, value in insights['content_features'].items():
                    report.append(f"- {feature}: {value:.2f}")
        
        return '\n'.join(report)

    def plot_performance_distributions(self):
        """Create visualizations comparing performance distributions"""
        if not self.top_performers:
            self.identify_top_performers()
            
        # Create visualization directory if it doesn't exist
        os.makedirs('top_performer_analysis', exist_ok=True)
        
        # Plot performance distributions for each metric
        metrics = ['click_rate', 'open_rate', 'total_revenue']
        
        for metric in metrics:
            fig = go.Figure()
            
            # Add overall distribution
            fig.add_trace(go.Histogram(
                x=self.df[metric],
                name='All Campaigns',
                opacity=0.7,
                nbinsx=50
            ))
            
            # Add distributions for top performers
            for category, data in self.top_performers.items():
                if isinstance(data.get('campaigns'), pd.DataFrame):
                    fig.add_trace(go.Histogram(
                        x=data['campaigns'][metric],
                        name=f'Top {category}',
                        opacity=0.7,
                        nbinsx=50
                    ))
            
            fig.update_layout(
                title=f'{metric} Distribution Comparison',
                xaxis_title=metric,
                yaxis_title='Count',
                barmode='overlay'
            )
            
            fig.write_html(f'top_performer_analysis/{metric}_distribution.html')



class WordPerformanceAnalyzer:
    def __init__(self, content_analyzer):
        """Initialize with ContentAnalyzer instance"""
        self.df = content_analyzer.df.copy()
        # Clean and prepare data
        self.df = self.df.dropna(subset=['subject', 'parsed_body', 'click_rate', 'open_rate', 'total_revenue'])
        self.nlp = spacy.load('en_core_web_sm')
        print(f"Initialized with {len(self.df)} valid campaigns")
        
    def analyze_word_performance(self, min_occurrences=10):
        """
        Analyze how individual words correlate with performance metrics
        
        Args:
            min_occurrences: Minimum number of times a word must appear to be included
        """
        # Combine subject and body text
        self.df['full_text'] = self.df['subject'] + ' ' + self.df['parsed_body']
        
        # Process text and extract meaningful words
        word_performances = []
        
        # Track all words and their occurrences
        word_counts = Counter()
        campaign_words = {}
        
        print("Processing campaign text...")
        for idx, row in self.df.iterrows():
            doc = self.nlp(row['full_text'].lower())
            # Only keep meaningful words (no stopwords, punctuation, etc.)
            meaningful_words = [
                token.text for token in doc 
                if not token.is_stop and not token.is_punct and not token.is_space
                and len(token.text) > 1  # Avoid single characters
            ]
            word_counts.update(meaningful_words)
            campaign_words[idx] = set(meaningful_words)  # Use set to count word presence, not frequency
        
        print(f"Analyzing {len(word_counts)} unique words...")
        # Analyze performance for each word that appears enough times
        for word, count in word_counts.items():
            if count >= min_occurrences:
                # Get campaigns with and without this word
                with_word = [
                    idx for idx in campaign_words 
                    if word in campaign_words[idx]
                ]
                
                if len(with_word) > 0:
                    # Calculate metrics
                    campaigns_with_word = self.df.loc[with_word]
                    campaigns_without_word = self.df.loc[~self.df.index.isin(with_word)]
                    
                    metrics = {
                        'word': word,
                        'occurrences': count,
                        'campaigns_with_word': len(campaigns_with_word),
                        'campaigns_without_word': len(campaigns_without_word),
                        
                        # Click rate metrics
                        'click_rate_with': campaigns_with_word['click_rate'].mean(),
                        'click_rate_without': campaigns_without_word['click_rate'].mean(),
                        'click_rate_lift': (
                            campaigns_with_word['click_rate'].mean() / 
                            campaigns_without_word['click_rate'].mean() - 1
                        ) if campaigns_without_word['click_rate'].mean() > 0 else 0,
                        
                        # Open rate metrics
                        'open_rate_with': campaigns_with_word['open_rate'].mean(),
                        'open_rate_without': campaigns_without_word['open_rate'].mean(),
                        'open_rate_lift': (
                            campaigns_with_word['open_rate'].mean() / 
                            campaigns_without_word['open_rate'].mean() - 1
                        ) if campaigns_without_word['open_rate'].mean() > 0 else 0,
                        
                        # Total Revenue metrics
                        'total_revenue_with': campaigns_with_word['total_revenue'].mean(),
                        'total_revenue_without': campaigns_without_word['total_revenue'].mean(),
                        'total_revenue_lift': (
                            campaigns_with_word['total_revenue'].mean() / 
                            campaigns_without_word['total_revenue'].mean() - 1
                        ) if campaigns_without_word['total_revenue'].mean() > 0 else 0
                    }
                    
                    # Add statistical significance using t-test
                    for metric in ['click_rate', 'open_rate', 'total_revenue']:
                        try:
                            _, p_value = stats.ttest_ind(
                                campaigns_with_word[metric].dropna(),
                                campaigns_without_word[metric].dropna()
                            )
                            metrics[f'{metric}_p_value'] = p_value
                            
                            # Add diagnostic print
                            print(f"Successfully added p-value for {metric}: {p_value}")
                        except Exception as e:
                            print(f"Error calculating p-value for {metric}: {str(e)}")
                            metrics[f'{metric}_p_value'] = 1.0  # Set to 1 if test fails
                    
                    word_performances.append(metrics)
        
        # Convert to DataFrame and sort by different metrics
        results = pd.DataFrame(word_performances)
        
        # Create different views of the data
        performance_views = {}
        
        for metric in ['click_rate', 'open_rate', 'total_revenue']:
            # Map the metric name for total_revenue
            p_value_col = f"{metric}_p_value"
            lift_col = f"{metric}_lift"
            
            # Print diagnostic information
            print(f"\nAnalyzing {metric}:")
            print(f"Available columns: {results.columns.tolist()}")
            
            try:
                # Filter for statistical significance
                significant_results = results[
                    (results[p_value_col] < 0.05) &  # Statistically significant
                    (results['occurrences'] >= min_occurrences) &  # Enough occurrences
                    (results['campaigns_with_word'] >= 5)  # Minimum campaign threshold
                ].copy()
                
                # Sort by lift
                performance_views[f'{metric}_top_words'] = (
                    significant_results
                    .sort_values(lift_col, ascending=False)
                    .head(20)
                )
                
                # Print insights for this metric
                df = performance_views[f'{metric}_top_words']
                if not df.empty:
                    print(f"\nTop words for {metric} performance (p < 0.05):")
                    for _, row in df.head().iterrows():
                        print(f"\nWord: '{row['word']}'")
                        print(f"Occurrences: {row['occurrences']}")
                        print(f"Lift: {row[f'{metric}_lift']:.2%}")
                        print(f"With word: {row[f'{metric}_with']:.4f}")
                        print(f"Without word: {row[f'{metric}_without']:.4f}")
                        print(f"P-value: {row[f'{metric}_p_value']:.4f}")
                else:
                    print(f"\nNo significant words found for {metric}")
                    
            except KeyError as e:
                print(f"Error processing {metric}: {str(e)}")
                performance_views[f'{metric}_top_words'] = pd.DataFrame()
        
        return performance_views

    def analyze_word_combinations(self, top_words_per_metric=10):
        """
        Analyze how combinations of high-performing words work together
        
        Args:
            top_words_per_metric: Number of top words to consider for combinations
        """
        print("\nAnalyzing word combinations...")
        # First get individual word performance
        performance_views = self.analyze_word_performance()
        
        # Get top words for each metric
        top_words = {}
        for metric in ['click_rate', 'open_rate', 'total_revenue']:
            top_words[metric] = set(
                performance_views[f'{metric}_top_words']['word'].head(top_words_per_metric)
            )
            print(f"\nTop {len(top_words[metric])} words for {metric}:")
            print(', '.join(top_words[metric]))
        
        # Analyze combinations
        combination_results = []
        
        for metric in ['click_rate', 'open_rate', 'total_revenue']:
            words = list(top_words[metric])
            print(f"\nAnalyzing combinations for {metric}...")
            
            # Look at pairs of words
            for word1, word2 in combinations(words, 2):
                # Find campaigns with both words
                campaigns_with_both = self.df[
                    self.df['full_text'].str.contains(word1, case=False) &
                    self.df['full_text'].str.contains(word2, case=False)
                ]
                
                # Find campaigns with neither word
                campaigns_with_neither = self.df[
                    ~self.df['full_text'].str.contains(word1, case=False) &
                    ~self.df['full_text'].str.contains(word2, case=False)
                ]
                
                if len(campaigns_with_both) >= 5:  # Minimum sample size
                    result = {
                        'metric': metric,
                        'word1': word1,
                        'word2': word2,
                        'campaigns_with_both': len(campaigns_with_both),
                        f'{metric}_with_both': campaigns_with_both[metric].mean(),
                        f'{metric}_with_neither': campaigns_with_neither[metric].mean(),
                        'lift': (
                            campaigns_with_both[metric].mean() /
                            campaigns_with_neither[metric].mean() - 1
                        ) if campaigns_with_neither[metric].mean() > 0 else 0
                    }
                    
                    # Add statistical significance
                    try:
                        _, p_value = stats.ttest_ind(
                            campaigns_with_both[metric].dropna(),
                            campaigns_with_neither[metric].dropna()
                        )
                        result['p_value'] = p_value
                    except Exception as e:
                        print(f"Error calculating p-value for combination {word1}+{word2}: {str(e)}")
                        result['p_value'] = 1.0  # Set to 1 if test fails
                    
                    combination_results.append(result)
        
        # Convert to DataFrame and filter significant results
        combinations_df = pd.DataFrame(combination_results)
        significant_combinations = combinations_df[
            (combinations_df['p_value'] < 0.05) &
            (combinations_df['campaigns_with_both'] >= 5)
        ]
        
        # Print insights
        for metric in ['click_rate', 'open_rate', 'total_revenue']:
            metric_combinations = (
                significant_combinations[significant_combinations['metric'] == metric]
                .sort_values('lift', ascending=False)
                .head(5)
            )
            
            print(f"\nTop word combinations for {metric}:")
            for _, row in metric_combinations.iterrows():
                print(f"\nWords: '{row['word1']}' + '{row['word2']}'")
                print(f"Lift: {row['lift']:.2%}")
                print(f"Campaigns using both: {row['campaigns_with_both']}")
                print(f"P-value: {row['p_value']:.4f}")
        
        return significant_combinations

    def plot_word_impact(self, performance_views):
        """
        Create visualizations of word impact on performance metrics
        
        Args:
            performance_views: Output from analyze_word_performance()
        """
        print("\nGenerating visualizations...")
        os.makedirs('word_analysis', exist_ok=True)
        
        for metric in ['click_rate', 'open_rate', 'total_revenue']:
            # Get top words for this metric
            top_words = performance_views[f'{metric}_top_words']
            
            if len(top_words) == 0:
                print(f"No significant words found for {metric}")
                continue
            
            # Create bar chart of lifts
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=top_words['word'],
                y=top_words[f'{metric}_lift'],
                text=[f"{x:.1%}" for x in top_words[f'{metric}_lift']],
                textposition='auto',
            ))
            
            fig.update_layout(
                title=f'Impact on {metric} by Word',
                xaxis_title='Word',
                yaxis_title='Lift',
                yaxis_tickformat=',.0%',
                height=600,
                showlegend=False
            )
            
            fig.write_html(f'word_analysis/{metric}_word_impact.html')
            
            # Create scatter plot of performance with/without word
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=top_words['occurrences'],
                y=top_words[f'{metric}_lift'],
                mode='markers+text',
                text=top_words['word'],
                textposition="top center",
                marker=dict(
                    size=top_words['occurrences'].apply(lambda x: np.sqrt(x) * 2),
                    sizemin=8,
                    color=top_words[f'{metric}_p_value'],
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title='P-value')
                )
            ))
            
            fig.update_layout(
                title=f'{metric} Lift vs Word Frequency',
                xaxis_title='Number of Occurrences',
                yaxis_title='Lift',
                yaxis_tickformat=',.0%',
                height=600
            )
            
            fig.write_html(f'word_analysis/{metric}_frequency_impact.html')
        
        print("Visualizations saved in 'word_analysis' directory")


class EnhancedSubjectLineAnalyzer:
    def __init__(self, content_analyzer):
        """Initialize with ContentAnalyzer instance"""
        self.df = content_analyzer.df.copy()
        self.nlp = spacy.load('en_core_web_sm')
        
        # Data validation and cleaning
        self._validate_and_clean_data()
        self.baseline_metrics = self._calculate_baseline_metrics()
        
    def _validate_and_clean_data(self):
        """Validate and clean the dataset"""
        initial_count = len(self.df)
        
        # Remove invalid entries
        self.df = self.df.dropna(subset=['subject', 'open_rate'])
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=['subject'])
        
        # Filter out invalid open rates
        self.df = self.df[
            (self.df['open_rate'] >= 0) & 
            (self.df['open_rate'] <= 1)
        ]
        
        # Remove subjects that are too short or too long (potential data issues)
        self.df = self.df[
            (self.df['subject'].str.len() >= 10) &
            (self.df['subject'].str.len() <= 200)
        ]
        
        print("\nData Validation Report:")
        print(f"Initial campaigns: {initial_count}")
        print(f"After cleaning: {len(self.df)}")
        print(f"Removed: {initial_count - len(self.df)} campaigns")
        
        # Calculate and print basic statistics
        self._print_basic_stats()
        
    def _print_basic_stats(self):
        """Print basic statistics about the dataset"""
        stats = {
            'Total Campaigns': len(self.df),
            'Average Subject Length': self.df['subject'].str.len().mean(),
            'Median Subject Length': self.df['subject'].str.len().median(),
            'Average Open Rate': self.df['open_rate'].mean(),
            'Median Open Rate': self.df['open_rate'].median(),
            'Open Rate Std Dev': self.df['open_rate'].std()
        }
        
        print("\nBasic Statistics:")
        for metric, value in stats.items():
            print(f"{metric}: {value:.2f}")
            
    def _calculate_baseline_metrics(self):
        """Calculate baseline metrics for comparison"""
        return {
            'mean_open_rate': self.df['open_rate'].mean(),
            'median_open_rate': self.df['open_rate'].median(),
            'open_rate_std': self.df['open_rate'].std(),
            'quartiles': self.df['open_rate'].quantile([0.25, 0.5, 0.75]).to_dict()
        }
        
    def _extract_word_features(self, text):
        """Extract detailed features about words in text"""
        doc = self.nlp(text.lower())
        
        features = {
            'words': [],
            'pos_tags': Counter(),
            'word_length': [],
            'is_capitalized': [],
            'entities': [],
            'sentence_count': len(list(doc.sents)),
            'has_number': any(token.like_num for token in doc),
            'has_currency': any(token.like_num and '$' in text for token in doc),
            'has_percentage': any(token.like_num and '%' in text for token in doc),
            'sentiment': TextBlob(text).sentiment.polarity
        }
        
        for token in doc:
            if not token.is_punct and not token.is_space:
                features['words'].append(token.text)
                features['pos_tags'][token.pos_] += 1
                features['word_length'].append(len(token.text))
                features['is_capitalized'].append(token.text[0].isupper())
                
        for ent in doc.ents:
            features['entities'].append((ent.text, ent.label_))
            
        return features

    def analyze_subject_patterns(self, min_occurrences=5):
        """Analyze patterns in subject lines with enhanced statistical analysis"""
        word_stats = {}
        pattern_stats = {
            'length_impact': {},
            'pos_patterns': {},
            'entity_patterns': {},
            'word_combinations': {},
            'special_features': {
                'has_number': {'count': 0, 'open_rates': []},
                'has_currency': {'count': 0, 'open_rates': []},
                'has_percentage': {'count': 0, 'open_rates': []},
                'sentiment': {'positive': [], 'negative': [], 'neutral': []}
            }
        }
        
        # Process each subject line
        for _, row in self.df.iterrows():
            subject = row['subject']
            features = self._extract_word_features(subject)
            
            # Track special features
            if features['has_number']:
                pattern_stats['special_features']['has_number']['count'] += 1
                pattern_stats['special_features']['has_number']['open_rates'].append(row['open_rate'])
                
            if features['has_currency']:
                pattern_stats['special_features']['has_currency']['count'] += 1
                pattern_stats['special_features']['has_currency']['open_rates'].append(row['open_rate'])
                
            if features['has_percentage']:
                pattern_stats['special_features']['has_percentage']['count'] += 1
                pattern_stats['special_features']['has_percentage']['open_rates'].append(row['open_rate'])
                
            # Track sentiment
            if features['sentiment'] > 0.1:
                pattern_stats['special_features']['sentiment']['positive'].append(row['open_rate'])
            elif features['sentiment'] < -0.1:
                pattern_stats['special_features']['sentiment']['negative'].append(row['open_rate'])
            else:
                pattern_stats['special_features']['sentiment']['neutral'].append(row['open_rate'])
            
            # Analyze individual words
            unique_words = set(features['words'])
            for word in unique_words:
                if word not in word_stats:
                    word_stats[word] = {
                        'count': 1,
                        'open_rates': [row['open_rate']],
                        'contexts': [subject],
                        'positions': [features['words'].index(word)],
                        'capitalized_count': 1 if word[0].isupper() else 0
                    }
                else:
                    word_stats[word]['count'] += 1
                    word_stats[word]['open_rates'].append(row['open_rate'])
                    word_stats[word]['contexts'].append(subject)
                    word_stats[word]['positions'].append(features['words'].index(word))
                    word_stats[word]['capitalized_count'] += 1 if word[0].isupper() else 0
            
            # Analyze word combinations (bigrams and trigrams)
            words = features['words']
            for i in range(len(words) - 1):
                bigram = (words[i], words[i + 1])
                if bigram not in pattern_stats['word_combinations']:
                    pattern_stats['word_combinations'][bigram] = {
                        'count': 1,
                        'open_rates': [row['open_rate']],
                        'contexts': [subject]
                    }
                else:
                    pattern_stats['word_combinations'][bigram]['count'] += 1
                    pattern_stats['word_combinations'][bigram]['open_rates'].append(row['open_rate'])
                    pattern_stats['word_combinations'][bigram]['contexts'].append(subject)
            
            # Length analysis
            length_bin = len(subject) // 10 * 10  # Group by 10-character intervals
            if length_bin not in pattern_stats['length_impact']:
                pattern_stats['length_impact'][length_bin] = {
                    'count': 1,
                    'open_rates': [row['open_rate']],
                    'examples': [subject]
                }
            else:
                pattern_stats['length_impact'][length_bin]['count'] += 1
                pattern_stats['length_impact'][length_bin]['open_rates'].append(row['open_rate'])
                pattern_stats['length_impact'][length_bin]['examples'].append(subject)
            
            # POS patterns
            for pos, count in features['pos_tags'].items():
                if pos not in pattern_stats['pos_patterns']:
                    pattern_stats['pos_patterns'][pos] = {
                        'count': 1,
                        'open_rates': [row['open_rate']],
                        'examples': [subject]
                    }
                else:
                    pattern_stats['pos_patterns'][pos]['count'] += 1
                    pattern_stats['pos_patterns'][pos]['open_rates'].append(row['open_rate'])
                    pattern_stats['pos_patterns'][pos]['examples'].append(subject)
            
            # Entity patterns
            for entity, label in features['entities']:
                if label not in pattern_stats['entity_patterns']:
                    pattern_stats['entity_patterns'][label] = {
                        'count': 1,
                        'open_rates': [row['open_rate']],
                        'examples': [(entity, subject)]
                    }
                else:
                    pattern_stats['entity_patterns'][label]['count'] += 1
                    pattern_stats['entity_patterns'][label]['open_rates'].append(row['open_rate'])
                    pattern_stats['entity_patterns'][label]['examples'].append((entity, subject))
        
        # Convert word stats to DataFrame with enhanced statistics
        word_results = []
        baseline_mean = self.baseline_metrics['mean_open_rate']
        
        for word, stats in word_stats.items():
            if stats['count'] >= min_occurrences:
                open_rates = np.array(stats['open_rates'])
                mean_open_rate = np.mean(open_rates)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt((np.std(open_rates)**2 + self.baseline_metrics['open_rate_std']**2) / 2)
                effect_size = (mean_open_rate - baseline_mean) / pooled_std if pooled_std > 0 else 0
                
                # Perform statistical test
                other_campaigns = self.df[~self.df['subject'].str.contains(word, case=False, regex=False)]
                _, p_value = stats.ttest_ind(
                    open_rates,
                    other_campaigns['open_rate'].values,
                    equal_var=False  # Use Welch's t-test for unequal variances
                )
                
                # Calculate average position and position impact
                avg_position = np.mean(stats['positions'])
                position_impact = stats.get('capitalized_count', 0) / stats['count']
                
                word_results.append({
                    'word': word,
                    'count': stats['count'],
                    'mean_open_rate': mean_open_rate,
                    'median_open_rate': np.median(open_rates),
                    'open_rate_std': np.std(open_rates),
                    'effect_size': effect_size,
                    'p_value': p_value,
                    'open_rate_impact': mean_open_rate - baseline_mean,
                    'contexts': stats['contexts'][:5],  # Store up to 5 example contexts
                    'avg_position': avg_position,
                    'capitalization_rate': position_impact,
                    'confidence_interval': stats.t.interval(
                        0.95, 
                        len(open_rates)-1, 
                        loc=mean_open_rate, 
                        scale=stats.sem(open_rates)
                    ) if len(open_rates) > 1 else (0, 0)
                })
        
        return pd.DataFrame(word_results), pattern_stats

    def create_enhanced_visualizations(self, word_results_df, pattern_stats):
        """Create comprehensive visualizations with additional insights"""
        os.makedirs('subject_analysis_output', exist_ok=True)
        
        # 1. Word Impact Bubble Chart
        significant_words = word_results_df[
            (word_results_df['p_value'] < 0.05) & 
            (abs(word_results_df['effect_size']) > 0.2)
        ].head(30)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=significant_words['count'],
            y=significant_words['open_rate_impact'] * 100,
            mode='markers+text',
            text=significant_words['word'],
            textposition="top center",
            marker=dict(
                size=significant_words['effect_size'].abs() * 20,
                sizemode='area',
                sizeref=2.*max(significant_words['effect_size'].abs())*20/(40.**2),
                sizemin=4,
                color=significant_words['p_value'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title='P-value')
            )
        ))
        
        fig.update_layout(
            title='Word Impact Analysis (Size = Effect Size)',
            xaxis_title='Number of Occurrences',
            yaxis_title='Open Rate Impact (%)',
            height=800,
            showlegend=False
        )
        
        fig.write_html('subject_analysis_output/word_impact_bubble.html')
        
        # 2. Subject Length Impact
        length_data = []
        for length_bin, stats in pattern_stats['length_impact'].items():
            if stats['count'] >= 5:  # Minimum sample size
                length_data.append({
                    'length_bin': length_bin,
                    'avg_open_rate': np.mean(stats['open_rates']),
                    'count': stats['count']
                })
        
        length_df = pd.DataFrame(length_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=length_df['length_bin'],
            y=length_df['avg_open_rate'] * 100,
            mode='lines+markers',
            name='Average Open Rate',
            text=length_df['count'].apply(lambda x: f'n={x}'),
            hovertemplate='Length: %{x}<br>Open Rate: %{y:.1f}%<br>%{text}'
        ))
        
        fig.update_layout(
            title='Impact of Subject Line Length',
            xaxis_title='Subject Length (characters)',
            yaxis_title='Average Open Rate (%)',
            height=500
        )
        
        fig.write_html('subject_analysis_output/length_impact.html')
        
        # 3. Part of Speech Impact
        pos_data = []
        for pos, stats in pattern_stats['pos_patterns'].items():
            if stats['count'] >= 5:
                pos_data.append({
                    'pos': pos,
                    'avg_open_rate': np.mean(stats['open_rates']),
                    'count': stats['count']
                })
        
        pos_df = pd.DataFrame(pos_data)
        
        fig = go.Figure(go.Bar(
            x=pos_df['pos'],
            y=pos_df['avg_open_rate'] * 100,
            text=pos_df['count'].apply(lambda x: f'n={x}'),
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Impact of Parts of Speech',
            xaxis_title='Part of Speech',
            yaxis_title='Average Open Rate (%)',
            height=500
        )
        
        fig.write_html('subject_analysis_output/pos_impact.html')
        
        # 4. Special Features Impact
        special_features = []
        for feature, data in pattern_stats['special_features'].items():
            if feature != 'sentiment':
                if data['count'] >= 5:
                    special_features.append({
                        'feature': feature,
                        'avg_open_rate': np.mean(data['open_rates']),
                        'count': data['count']
                    })
        
        # Add sentiment analysis
        sentiment_data = pattern_stats['special_features']['sentiment']
        for sentiment_type in ['positive', 'negative', 'neutral']:
            if len(sentiment_data[sentiment_type]) >= 5:
                special_features.append({
                    'feature': f'sentiment_{sentiment_type}',
                    'avg_open_rate': np.mean(sentiment_data[sentiment_type]),
                    'count': len(sentiment_data[sentiment_type])
                })
        
        special_df = pd.DataFrame(special_features)
        
        fig = go.Figure(go.Bar(
            x=special_df['feature'],
            y=special_df['avg_open_rate'] * 100,
            text=special_df['count'].apply(lambda x: f'n={x}'),
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Impact of Special Features',
            xaxis_title='Feature',
            yaxis_title='Average Open Rate (%)',
            height=500
        )
        
        fig.write_html('subject_analysis_output/special_features_impact.html')
        
        # 5. Word Position Analysis
        position_impact = go.Figure()
        
        position_impact.add_trace(go.Scatter(
            x=word_results_df['avg_position'],
            y=word_results_df['open_rate_impact'] * 100,
            mode='markers',
            marker=dict(
                size=word_results_df['count'],
                sizemode='area',
                sizeref=2.*max(word_results_df['count'])/(40.**2),
                sizemin=4,
                color=word_results_df['effect_size'],
                colorscale='RdYlBu',
                showscale=True
            ),
            text=word_results_df['word'],
            hovertemplate='Word: %{text}<br>Position: %{x:.1f}<br>Impact: %{y:.1f}%'
        ))
        
        position_impact.update_layout(
            title='Word Impact by Position in Subject Line',
            xaxis_title='Average Word Position',
            yaxis_title='Open Rate Impact (%)',
            height=600
        )
        
        position_impact.write_html('subject_analysis_output/position_impact.html')

    def generate_comprehensive_report(self, word_results_df, pattern_stats):
        """Generate a detailed analytical report"""
        report = []
        report.append("Comprehensive Subject Line Analysis Report")
        report.append("=======================================")
        
        # 1. Overall Statistics
        report.append("\n1. Dataset Overview")
        report.append("-----------------")
        report.append(f"Total Campaigns Analyzed: {len(self.df):,}")
        report.append(f"Baseline Open Rate: {self.baseline_metrics['mean_open_rate']*100:.2f}%")
        report.append(f"Open Rate Standard Deviation: {self.baseline_metrics['open_rate_std']*100:.2f}%")
        
        # 2. Word Analysis
        report.append("\n2. Word Impact Analysis")
        report.append("---------------------")
        
        # High-impact words
        significant_positive = word_results_df[
            (word_results_df['p_value'] < 0.05) &
            (word_results_df['effect_size'] > 0.2)
        ].head(10)
        
        report.append("\nTop Positive Impact Words:")
        for _, row in significant_positive.iterrows():
            report.append(f"\n{row['word']}:")
            report.append(f"- Impact: +{row['open_rate_impact']*100:.2f}% open rate")
            report.append(f"- Effect Size: {row['effect_size']:.2f}")
            report.append(f"- Occurrences: {row['count']}")
            report.append(f"- Average Position: {row['avg_position']:.1f}")
            report.append(f"- P-value: {row['p_value']:.4f}")
            report.append("- Example contexts:")
            for context in row['contexts'][:3]:
                report.append(f"  * {context}")
        
        # 3. Special Features Analysis
        report.append("\n3. Special Features Analysis")
        report.append("---------------------------")
        
        for feature, data in pattern_stats['special_features'].items():
            if feature != 'sentiment':
                if data['count'] >= 5:
                    avg_rate = np.mean(data['open_rates'])
                    impact = avg_rate - self.baseline_metrics['mean_open_rate']
                    report.append(f"\n{feature.replace('_', ' ').title()}:")
                    report.append(f"- Count: {data['count']}")
                    report.append(f"- Average Open Rate: {avg_rate*100:.2f}%")
                    report.append(f"- Impact: {impact*100:+.2f}%")
        
        # Sentiment Analysis
        report.append("\nSentiment Analysis:")
        sentiment_data = pattern_stats['special_features']['sentiment']
        for sentiment_type in ['positive', 'negative', 'neutral']:
            if len(sentiment_data[sentiment_type]) >= 5:
                avg_rate = np.mean(sentiment_data[sentiment_type])
                report.append(f"- {sentiment_type.title()} Sentiment: {avg_rate*100:.2f}% "
                            f"(n={len(sentiment_data[sentiment_type])})")
        
        # 4. Length Analysis
        report.append("\n4. Subject Length Analysis")
        report.append("------------------------")
        
        length_impacts = []
        for length_bin, stats in pattern_stats['length_impact'].items():
            if stats['count'] >= 5:
                avg_rate = np.mean(stats['open_rates'])
                length_impacts.append((length_bin, avg_rate, stats['count']))
        
        length_impacts.sort(key=lambda x: x[1], reverse=True)
        report.append("\nOptimal Length Ranges:")
        for length_bin, rate, count in length_impacts[:3]:
            report.append(f"- {length_bin}-{length_bin+10} characters: {rate*100:.2f}% open rate (n={count})")
            report.append("  Example:")
            report.append(f"  * {pattern_stats['length_impact'][length_bin]['examples'][0]}")
        
        # 5. Part of Speech Analysis
        report.append("\n5. Language Pattern Analysis")
        report.append("---------------------------")
        
        pos_impacts = []
        for pos, stats in pattern_stats['pos_patterns'].items():
            if stats['count'] >= 5:
                avg_rate = np.mean(stats['open_rates'])
                pos_impacts.append((pos, avg_rate, stats['count']))
        
        pos_impacts.sort(key=lambda x: x[1], reverse=True)
        report.append("\nMost Effective Parts of Speech:")
        for pos, rate, count in pos_impacts:
            report.append(f"- {pos}: {rate*100:.2f}% open rate (n={count})")
            report.append("  Example:")
            report.append(f"  * {pattern_stats['pos_patterns'][pos]['examples'][0]}")
        
        # 6. Entity Analysis
        report.append("\n6. Named Entity Analysis")
        report.append("----------------------")
        
        entity_impacts = []
        for entity_type, stats in pattern_stats['entity_patterns'].items():
            if stats['count'] >= 5:
                avg_rate = np.mean(stats['open_rates'])
                entity_impacts.append((entity_type, avg_rate, stats['count'], stats['examples'][0]))
        
        entity_impacts.sort(key=lambda x: x[1], reverse=True)
        report.append("\nMost Effective Entity Types:")
        for entity_type, rate, count, (example_entity, example_subject) in entity_impacts:
            report.append(f"- {entity_type}: {rate*100:.2f}% open rate (n={count})")
            report.append(f"  Example: '{example_entity}' in subject:")
            report.append(f"  * {example_subject}")
        
        # Save report
        os.makedirs('subject_analysis_output', exist_ok=True)
        with open('subject_analysis_output/comprehensive_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)

    def analyze_word_combinations(self, min_occurrences=5):
        """Analyze how combinations of words perform together"""
        # Get word pairs from subject lines
        word_pairs = {}
        
        for _, row in self.df.iterrows():
            doc = self.nlp(row['subject'].lower())
            words = [token.text for token in doc if not token.is_punct and not token.is_space]
            
            # Look at adjacent word pairs
            for i in range(len(words) - 1):
                pair = (words[i], words[i+1])
                if pair not in word_pairs:
                    word_pairs[pair] = {
                        'count': 1,
                        'open_rates': [row['open_rate']],
                        'examples': [row['subject']]
                    }
                else:
                    word_pairs[pair]['count'] += 1
                    word_pairs[pair]['open_rates'].append(row['open_rate'])
                    word_pairs[pair]['examples'].append(row['subject'])
        
        # Analyze pair performance
        pair_results = []
        baseline_mean = self.baseline_metrics['mean_open_rate']
        
        for pair, stats in word_pairs.items():
            if stats['count'] >= min_occurrences:
                open_rates = np.array(stats['open_rates'])
                mean_open_rate = np.mean(open_rates)
                
                # Calculate effect size
                pooled_std = np.sqrt((np.std(open_rates)**2 + self.baseline_metrics['open_rate_std']**2) / 2)
                effect_size = (mean_open_rate - baseline_mean) / pooled_std if pooled_std > 0 else 0
                
                # Statistical test
                other_campaigns = self.df[~self.df['subject'].str.contains(f"{pair[0]}.*{pair[1]}", case=False)]
                _, p_value = stats.ttest_ind(
                    open_rates,
                    other_campaigns['open_rate'].values,
                    equal_var=False
                )
                
                pair_results.append({
                    'word_pair': f"{pair[0]} {pair[1]}",
                    'word1': pair[0],
                    'word2': pair[1],
                    'count': stats['count'],
                    'mean_open_rate': mean_open_rate,
                    'open_rate_impact': mean_open_rate - baseline_mean,
                    'effect_size': effect_size,
                    'p_value': p_value,
                    'examples': stats['examples'][:3]
                })
        
        return pd.DataFrame(pair_results)

    def create_word_combination_visualizations(self, pair_results_df):
        """Create visualizations for word combination analysis"""
        # Filter for significant combinations
        significant_pairs = pair_results_df[
            (pair_results_df['p_value'] < 0.05) &
            (abs(pair_results_df['effect_size']) > 0.2)
        ].head(20)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes and edges
        for _, row in significant_pairs.iterrows():
            G.add_edge(
                row['word1'], 
                row['word2'], 
                weight=abs(row['effect_size']),
                impact=row['open_rate_impact']
            )
        
        # Create positions for nodes
        pos = nx.spring_layout(G)
        
        # Create network visualization
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        node_sizes = [G.degree(node) * 100 for node in G.nodes()]
        
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            impact = G[edge[0]][edge[1]]['impact']
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=weight * 2,
                    color='rgba(0,0,255,0.5)' if impact > 0 else 'rgba(255,0,0,0.5)'
                ),
                hoverinfo='text',
                text=f"Impact: {impact*100:.1f}%"
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            marker=dict(size=node_sizes),
            text=list(G.nodes()),
            textposition="top center"
        ))
        
        fig.update_layout(
            title='Word Combination Network',
            showlegend=False,
            height=800,
            width=800
        )
        
        fig.write_html('subject_analysis_output/word_combinations_network.html')

        
def main():
    """Main execution function with comprehensive campaign analysis"""
    # Initialize all analyzers
    campaign_analyzer = CampaignAnalyzer('lgd_campaigns.csv', 'chia_campaigns.csv')
    content_analyzer = ContentAnalyzer('lgd_campaigns.csv', 'chia_campaigns.csv')
    enhanced_analyzer = EnhancedContentAnalyzer(content_analyzer)
    
    # Initialize top performer analyzer with custom thresholds
    top_performer_analyzer = TopPerformerAnalyzer(content_analyzer, {
        'clicks': 80,  # 80th percentile for clicks
        'opens': 75,   # 75th percentile for opens
        'revenue': 70  # 70th percentile for revenue
    })
    
    # Initialize specialized analyzers
    time_analyzer = TimeAnalyzer(content_analyzer)
    word_analyzer = WordPerformanceAnalyzer(content_analyzer)
    subject_line_analyzer = EnhancedSubjectLineAnalyzer(content_analyzer)

    # 1. Campaign Performance Analysis
    print("\nGenerating Campaign Performance Analysis...")
    campaign_metrics = campaign_analyzer.analyze_performance_metrics()
    campaign_analyzer.plot_visualizations()

    # 2. Enhanced Content Analysis
    print("\nGenerating Enhanced Content Analysis...")
    subject_analysis = enhanced_analyzer.analyze_subject_linguistics()
    topic_analysis = enhanced_analyzer.extract_topics()
    structure_analysis = enhanced_analyzer.analyze_email_structure()

    # 3. Top Performer Analysis
    print("\nGenerating Top Performer Analysis...")
    performance_data = top_performer_analyzer.identify_top_performers()
    top_performer_report = top_performer_analyzer.generate_insights_report()
    print(top_performer_report)
    top_performer_analyzer.plot_performance_distributions()

    # 4. Word Performance Analysis
    print("\nGenerating Word Performance Analysis...")
    word_performance_views = word_analyzer.analyze_word_performance(min_occurrences=10)
    word_combinations = word_analyzer.analyze_word_combinations(top_words_per_metric=10)
    word_analyzer.plot_word_impact(word_performance_views)

    # 5. Enhanced Subject Line Analysis
    print("\nGenerating Enhanced Subject Line Analysis...")
    word_results, pattern_stats = subject_line_analyzer.analyze_subject_patterns(min_occurrences=5)
    subject_line_analyzer.create_enhanced_visualizations(word_results, pattern_stats)
    subject_line_report = subject_line_analyzer.generate_comprehensive_report(word_results, pattern_stats)
    pair_results = subject_line_analyzer.analyze_word_combinations(min_occurrences=5)
    subject_line_analyzer.create_word_combination_visualizations(pair_results)

    # 6. Temporal Analysis
    print("\nGenerating Temporal Analysis...")
    temporal_patterns = time_analyzer.analyze_temporal_patterns()
    time_analyzer.create_temporal_visualizations()
    time_analyzer.print_temporal_insights()
    
    # 7. Anomaly and Seasonality Analysis
    anomalies = time_analyzer.detect_anomalies()
    seasonality_analysis = time_analyzer.analyze_seasonality()

    # Create output directory
    os.makedirs('analysis_results', exist_ok=True)
    
    # Generate comprehensive report
    print("\nGenerating comprehensive analysis report...")
    with open('analysis_results/summary_report.txt', 'w') as f:
        # Report Header
        f.write("Campaign Analysis Summary Report\n")
        f.write("==============================\n\n")
        
        # 1. Overall Performance Metrics
        f.write("1. Overall Performance Metrics\n")
        f.write("----------------------------\n")
        f.write(f"Total Revenue: ${campaign_metrics['overall_metrics']['total_revenue']:,.2f}\n")
        f.write(f"Total Clicks: {campaign_metrics['overall_metrics']['total_clicks']:,}\n")
        f.write(f"Revenue per Click: ${campaign_metrics['overall_metrics']['revenue_per_click']:.4f}\n\n")
        
        # 2. Top Performer Analysis
        f.write("2. Top Performer Analysis\n")
        f.write("------------------------\n")
        for category, data in performance_data.items():
            f.write(f"\n{category.replace('_', ' ').title()} Performance:\n")
            f.write(f"Number of campaigns: {data['count']}\n")
            
            if 'mean_performance' in data:
                metric_name = data.get('metric_name', category)
                f.write(f"Mean {metric_name}: {data['mean_performance']:.4f}\n")
            
            if 'metrics_included' in data:
                f.write(f"Metrics included: {', '.join(data['metrics_included'])}\n")
            
            f.write("\n")

        # 3. Word Performance Analysis
        f.write("3. Word Performance Analysis\n")
        f.write("---------------------------\n")
        for metric in ['click_rate', 'open_rate', 'total_revenue']:
            if f'{metric}_top_words' in word_performance_views:
                top_words = word_performance_views[f'{metric}_top_words']
                if not top_words.empty:
                    f.write(f"\nTop 5 words for {metric}:\n")
                    for _, row in top_words.head().iterrows():
                        f.write(f"- '{row['word']}': {row[f'{metric}_lift']:.2%} lift ")
                        f.write(f"(p={row[f'{metric}_p_value']:.4f}, n={row['occurrences']})\n")

        # 4. Enhanced Subject Line Analysis
        f.write("\n4. Enhanced Subject Line Analysis\n")
        f.write("--------------------------------\n")
        f.write(subject_line_report)

        # 5. Word Combinations
        if not word_combinations.empty:
            f.write("\n5. Word Combinations Analysis\n")
            f.write("----------------------------\n")
            for metric in ['click_rate', 'open_rate', 'total_revenue']:
                metric_combinations = word_combinations[
                    word_combinations['metric'] == metric
                ].sort_values('lift', ascending=False).head(3)
                
                if not metric_combinations.empty:
                    f.write(f"\nBest combinations for {metric}:\n")
                    for _, row in metric_combinations.iterrows():
                        f.write(f"- '{row['word1']}' + '{row['word2']}': ")
                        f.write(f"{row['lift']:.2%} lift (p={row['p_value']:.4f})\n")
        
        # 6. Content Analysis Highlights
        if subject_analysis:
            f.write("\n6. Content Analysis Highlights\n")
            f.write("-----------------------------\n")
            # Add relevant content analysis metrics here
            
        # 7. Temporal Performance Analysis
        f.write("\n7. Temporal Performance Analysis\n")
        f.write("--------------------------------\n")
        best_hours = temporal_patterns['hourly']['click_rate'].nlargest(3)
        f.write(f"Top performing hours: {best_hours.index.tolist()}\n")
        
        # Holiday Performance
        holiday_impact = temporal_patterns['holiday']['click_rate']['mean']
        f.write("\nHoliday vs Non-Holiday Performance:\n")
        f.write(f"Holiday click rate: {holiday_impact[True]:.4f}\n")
        f.write(f"Non-holiday click rate: {holiday_impact[False]:.4f}\n")
        
        # 8. Anomalies Analysis
        f.write("\n8. Performance Anomalies\n")
        f.write("------------------------\n")
        anomaly_dates = anomalies[anomalies['click_rate_anomaly']].index
        f.write(f"Number of anomalous days: {len(anomaly_dates)}\n")
        if len(anomaly_dates) > 0:
            f.write("Notable anomaly dates:\n")
            for date in list(anomaly_dates)[:5]:
                f.write(f"- {date.strftime('%Y-%m-%d')}\n")
        
        # 9. Seasonality Analysis
        f.write("\n9. Seasonality Analysis\n")
        f.write("----------------------\n")
        f.write("Seasonal patterns detected in:\n")
        for metric in ['click_rate', 'open_rate', 'total_revenue']:
            seasonal = seasonality_analysis.get(metric, {}).get('seasonal')
            if seasonal is not None:
                f.write(f"- {metric}: {seasonal.std():.4f} standard deviation\n")

    print("\nAnalysis complete! Results saved in 'analysis_results' directory.")
    print("\nVisualizations saved in:")
    print("- 'analysis_output': Campaign visualizations")
    print("- 'time_analysis_output': Temporal analysis visualizations")
    print("- 'top_performer_analysis': Top performer visualizations")
    print("- 'word_analysis': Word performance visualizations")
    print("- 'subject_analysis_output': Subject line analysis visualizations")

if __name__ == "__main__":
    main()
