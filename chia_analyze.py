import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import os

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
        """Merge and prepare datasets"""
        df = self.chia_df.merge(
            self.lgd_df,
            left_on='id',
            right_on='campaign_id',
            how='inner'
        )
        df['sent_at'] = pd.to_datetime(df['sent_at'])
        return df
    
    def parse_body_content(self, row) -> str:
        """Parse body content from body_data and data_tokens"""
        try:
            body_data = json.loads(row['body_data'])
            data_tokens = json.loads(row['data_tokens'])
            return '\n\n'.join(body_data.get(token, '') for token in data_tokens)
        except:
            return ''

    def analyze_subject_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in subject lines and their performance"""
        subject_analysis = pd.DataFrame()
        
        # Group by subject ID and calculate metrics
        subject_metrics = self.df.groupby('creative_subject_id').agg({
            'open_rate': ['mean', 'median', 'std', 'count'],
            'click_rate': ['mean', 'median'],
            'subject': 'first'  # Get the subject text
        }).round(4)
        
        # Analyze patterns
        patterns = {
            'length': subject_metrics[('subject', 'first')].str.len(),
            'contains_number': subject_metrics[('subject', 'first')].str.contains(r'\d+'),
            'contains_dollar': subject_metrics[('subject', 'first')].str.contains(r'\$'),
            'all_caps_words': subject_metrics[('subject', 'first')].str.count(r'\b[A-Z]{2,}\b')
        }
        
        # Add pattern columns
        for name, pattern in patterns.items():
            subject_metrics[('pattern', name)] = pattern
        
        # Calculate performance by pattern
        pattern_performance = {}
        for name in patterns.keys():
            pattern_performance[name] = (
                subject_metrics.groupby(('pattern', name))[('open_rate', 'mean')]
                .agg(['mean', 'count'])
                .round(4)
            )
        
        return {
            'metrics': subject_metrics,
            'pattern_performance': pattern_performance
        }

    def analyze_body_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in body content and their performance"""
        body_analysis = pd.DataFrame()
        
        # Group by body ID and calculate metrics
        body_metrics = self.df.groupby('creative_body_id').agg({
            'click_rate': ['mean', 'median', 'std', 'count'],
            'total_revenue': ['mean', 'sum'],
            'parsed_body': 'first'  # Get the body text
        }).round(4)
        
        # Analyze patterns
        patterns = {
            'length': body_metrics[('parsed_body', 'first')].str.len(),
            'num_paragraphs': body_metrics[('parsed_body', 'first')].str.count(r'\n\n'),
            'contains_number': body_metrics[('parsed_body', 'first')].str.contains(r'\d+'),
            'contains_dollar': body_metrics[('parsed_body', 'first')].str.contains(r'\$'),
            'contains_question': body_metrics[('parsed_body', 'first')].str.contains(r'\?'),
            'contains_exclamation': body_metrics[('parsed_body', 'first')].str.contains(r'!')
        }
        
        # Add pattern columns
        for name, pattern in patterns.items():
            body_metrics[('pattern', name)] = pattern
        
        # Calculate performance by pattern
        pattern_performance = {}
        for name in patterns.keys():
            pattern_performance[name] = (
                body_metrics.groupby(('pattern', name))[('click_rate', 'mean')]
                .agg(['mean', 'count'])
                .round(4)
            )
        
        return {
            'metrics': body_metrics,
            'pattern_performance': pattern_performance
        }

    def analyze_theme_performance(self) -> pd.DataFrame:
        """Analyze performance by theme and content type"""
        theme_analysis = self.df.groupby(['theme_level1', 'theme_level2']).agg({
            'open_rate': ['mean', 'median', 'count'],
            'click_rate': ['mean', 'median'],
            'total_revenue': ['mean', 'sum']
        }).round(4)
        
        return theme_analysis

    def get_top_performers(self, n: int = 10) -> Dict[str, pd.DataFrame]:
        """Get top performing subjects and bodies"""
        # Top subjects by open rate
        top_subjects = (
            self.df.groupby(['creative_subject_id', 'subject'])
            .agg({
                'open_rate': ['mean', 'count'],
                'click_rate': 'mean'
            })
            .sort_values(('open_rate', 'mean'), ascending=False)
            .head(n)
        )
        
        # Top bodies by CTR
        top_bodies = (
            self.df.groupby(['creative_body_id', 'parsed_body'])
            .agg({
                'click_rate': ['mean', 'count'],
                'total_revenue': 'sum'
            })
            .sort_values(('click_rate', 'mean'), ascending=False)
            .head(n)
        )
        
        return {
            'top_subjects': top_subjects,
            'top_bodies': top_bodies
        }

    def plot_pattern_impacts(self):
        """Create visualizations of pattern impacts on performance"""
        # Subject patterns
        subject_analysis = self.analyze_subject_patterns()
        subject_patterns = subject_analysis['pattern_performance']
        
        fig_subjects = go.Figure()
        for pattern, performance in subject_patterns.items():
            fig_subjects.add_trace(go.Bar(
                name=pattern,
                x=performance.index,
                y=performance['mean'],
                text=performance['count'],
                textposition='auto',
            ))
        
        fig_subjects.update_layout(
            title='Impact of Subject Line Patterns on Open Rate',
            barmode='group',
            yaxis_title='Average Open Rate',
            showlegend=True
        )
        
        # Body patterns
        body_analysis = self.analyze_body_patterns()
        body_patterns = body_analysis['pattern_performance']
        
        fig_bodies = go.Figure()
        for pattern, performance in body_patterns.items():
            fig_bodies.add_trace(go.Bar(
                name=pattern,
                x=performance.index,
                y=performance['mean'],
                text=performance['count'],
                textposition='auto',
            ))
        
        fig_bodies.update_layout(
            title='Impact of Body Patterns on Click Rate',
            barmode='group',
            yaxis_title='Average Click Rate',
            showlegend=True
        )
        
        return {
            'subject_patterns': fig_subjects,
            'body_patterns': fig_bodies
        }

def main():
    """Main execution function"""
    analyzer = ContentAnalyzer('lgd_campaigns.csv', 'chia_campaigns.csv')
    
    # Get pattern analysis
    subject_patterns = analyzer.analyze_subject_patterns()
    body_patterns = analyzer.analyze_body_patterns()
    theme_performance = analyzer.analyze_theme_performance()
    top_performers = analyzer.get_top_performers()
    
    # Print insights
    print("\nSubject Line Patterns Impact on Open Rate:")
    for pattern, perf in subject_patterns['pattern_performance'].items():
        print(f"\n{pattern}:")
        print(perf)
    
    print("\nBody Content Patterns Impact on Click Rate:")
    for pattern, perf in body_patterns['pattern_performance'].items():
        print(f"\n{pattern}:")
        print(perf)
    
    print("\nTop Performing Subject Lines:")
    print(top_performers['top_subjects'])
    
    print("\nTop Performing Bodies:")
    print(top_performers['top_bodies'])
    
    # Create and save visualizations
    plots = analyzer.plot_pattern_impacts()
    plots['subject_patterns'].write_html('subject_pattern_analysis.html')
    plots['body_patterns'].write_html('body_pattern_analysis.html')

if __name__ == "__main__":
    main()
