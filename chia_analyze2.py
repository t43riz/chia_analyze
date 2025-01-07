import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import spacy
from textblob import TextBlob
import re
from collections import Counter
import networkx as nx
from itertools import combinations
import json

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
            # Create bins for numerical values
            subjects[f'{col}_bin'] = pd.qcut(subjects[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            # Calculate performance metrics by bin
            performance = subjects.groupby(f'{col}_bin').agg({
                'open_rate': ['mean', 'std', 'count'],
                'click_rate': ['mean', 'std']
            }).round(4)
            
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
            bodies[f'{topic_col}_bin'] = pd.qcut(bodies[topic_col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            performance = bodies.groupby(f'{topic_col}_bin').agg({
                'click_rate': ['mean', 'std', 'count']
            }).round(4)
            
            topic_performance[f'topic_{i}'] = performance
        
        return {
            'topics': topics,
            'topic_performance': topic_performance,
            'body_topics': bodies
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
                bodies[f'{col}_bin'] = pd.qcut(bodies[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                group_col = f'{col}_bin'
            else:
                group_col = col
            
            performance = bodies.groupby(group_col).agg({
                'click_rate': ['mean', 'std', 'count']
            }).round(4)
            
            structure_performance[col] = performance
        
        return {
            'detailed_metrics': bodies,
            'structure_performance': structure_performance
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
        # Create text features
        def extract_text_features(text):
            doc = self.nlp(text)
            blob = TextBlob(text)
            
            return {
                'word_count': len(text.split()),
                'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()),
                'sentiment': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'num_entities': len(doc.ents),
                'num_verbs': len([token for token in doc if token.pos_ == 'VERB']),
                'num_nouns': len([token for token in doc if token.pos_ == 'NOUN']),
                'num_adj': len([token for token in doc if token.pos_ == 'ADJ'])
            }
            
        # Extract features for clustering
        features_list = []
        for text in text_series:
            features_list.append(extract_text_features(text))
            
        features_df = pd.DataFrame(features_list)
        
        # Scale features and perform clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Analyze cluster characteristics
        features_df['cluster'] = clusters
        cluster_analysis = features_df.groupby('cluster').mean()
        
        return {
            'cluster_assignments': clusters,
            'cluster_characteristics': cluster_analysis
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

    # Save results
    os.makedirs('analysis_results', exist_ok=True)
    
    # Create summary report
    with open('analysis_results/summary_report.txt', 'w') as f:
        f.write("Campaign Analysis Summary Report\n")
        f.write("==============================\n\n")
        
        f.write("Overall Performance Metrics:\n")
        f.write(f"Total Revenue: ${campaign_metrics['overall_metrics']['total_revenue']:,.2f}\n")
        f.write(f"Total Clicks: {campaign_metrics['overall_metrics']['total_clicks']:,}\n")
        f.write(f"Revenue per Click: ${campaign_metrics['overall_metrics']['revenue_per_click']:.4f}\n\n")
        
        f.write("Top Performing Content Characteristics:\n")
        for cluster, chars in top_performer_insights['content_clusters']['cluster_characteristics'].iterrows():
            f.write(f"\nContent Cluster {cluster}:\n")
            for feature, value in chars.items():
                f.write(f"- {feature}: {value:.2f}\n")

if __name__ == "__main__":
    main()
