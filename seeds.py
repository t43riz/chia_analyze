import pandas as pd

def analyze_seed_data(file_path):
    """Analyze seed-related data in the dataset"""
    # Load the data
    print("Loading dataset...")
    df = pd.read_csv(file_path, low_memory=False)
    total_rows = len(df)
    
    print("\nSeed Data Analysis")
    print("==================")
    print(f"Total rows in dataset: {total_rows:,}")
    
    # Analyze seed_list_id
    seed_list_counts = df['seed_list_id'].value_counts()
    non_null_seed_lists = df['seed_list_id'].notna()
    
    print("\n1. seed_list_id Analysis:")
    print(f"Number of rows with seed_list_id: {non_null_seed_lists.sum():,} ({(non_null_seed_lists.sum()/total_rows)*100:.2f}%)")
    print(f"Number of unique seed_list_ids: {seed_list_counts.count():,}")
    print("\nTop 5 most common seed_list_ids:")
    for id, count in seed_list_counts.head().items():
        print(f"ID: {id}, Count: {count:,} ({(count/total_rows)*100:.2f}%)")
    
    # Analyze default_seed_list_id
    default_seed_counts = df['default_seed_list_id'].value_counts()
    non_null_default_seeds = df['default_seed_list_id'].notna()
    
    print("\n2. default_seed_list_id Analysis:")
    print(f"Number of rows with default_seed_list_id: {non_null_default_seeds.sum():,} ({(non_null_default_seeds.sum()/total_rows)*100:.2f}%)")
    print(f"Number of unique default_seed_list_ids: {default_seed_counts.count():,}")
    print("\nTop 5 most common default_seed_list_ids:")
    for id, count in default_seed_counts.head().items():
        print(f"ID: {id}, Count: {count:,} ({(count/total_rows)*100:.2f}%)")
    
    # Analyze overlap
    has_both = (non_null_seed_lists & non_null_default_seeds)
    print("\n3. Overlap Analysis:")
    print(f"Rows with both seed IDs: {has_both.sum():,} ({(has_both.sum()/total_rows)*100:.2f}%)")
    print(f"Rows with any seed ID: {(non_null_seed_lists | non_null_default_seeds).sum():,} ({((non_null_seed_lists | non_null_default_seeds).sum()/total_rows)*100:.2f}%)")
    
    # Create list of rows to potentially exclude
    rows_with_seeds = non_null_seed_lists | non_null_default_seeds
    print("\n4. Impact Analysis:")
    print(f"Rows that would be removed: {rows_with_seeds.sum():,}")
    print(f"Rows that would remain: {(~rows_with_seeds).sum():,}")
    
    return {
        'total_rows': total_rows,
        'seed_list_rows': non_null_seed_lists.sum(),
        'default_seed_rows': non_null_default_seeds.sum(),
        'overlap_rows': has_both.sum(),
        'total_seed_rows': rows_with_seeds.sum(),
        'remaining_rows': (~rows_with_seeds).sum()
    }

def analyze_seed_metrics(chia_path, lgd_path):
    """Analyze if seed IDs correlate with test campaign characteristics"""
    print("Loading datasets...")
    chia_df = pd.read_csv(chia_path, low_memory=False)
    lgd_df = pd.read_csv(lgd_path)
    
    # Merge datasets
    df = chia_df.merge(
        lgd_df,
        left_on='id',
        right_on='campaign_id',
        how='inner'
    )
    
    print("\nDetailed Seed Analysis")
    print("=====================")
    print(f"Total campaigns analyzed: {len(df):,}")
    
    # Analyze metrics by seed_list_id
    print("\n1. Metrics by seed_list_id:")
    seed_metrics = df.groupby('seed_list_id').agg({
        'sent': ['mean', 'min', 'max'],
        'delivered': ['mean', 'min', 'max'],
        'unique_opens': ['mean', 'min', 'max'],
        'unique_clicks': ['mean', 'min', 'max'],
        'total_revenue': ['mean', 'min', 'max']
    }).round(2)
    
    for seed_id in df['seed_list_id'].unique():
        print(f"\nSeed List ID: {seed_id}")
        metrics = seed_metrics.loc[seed_id]
        print(f"Sent: {metrics[('sent', 'mean')]:,.0f} (range: {metrics[('sent', 'min')]:,.0f} - {metrics[('sent', 'max')]:,.0f})")
        print(f"Revenue: ${metrics[('total_revenue', 'mean')]:,.2f} (range: ${metrics[('total_revenue', 'min')]:,.2f} - ${metrics[('total_revenue', 'max')]:,.2f})")
    
    # Check for patterns in campaign names or types
    print("\n2. Campaign Patterns:")
    for seed_id in df['seed_list_id'].unique():
        seed_campaigns = df[df['seed_list_id'] == seed_id]
        print(f"\nSeed List ID: {seed_id}")
        print(f"Number of campaigns: {len(seed_campaigns):,}")
        if 'campaign_name' in seed_campaigns.columns:
            print("Sample campaign names:")
            print(seed_campaigns['campaign_name'].head().tolist())
    
    # Look for test indicators
    test_indicators = ['test', 'seed', 'dummy', 'trial']
    print("\n3. Test Campaign Analysis:")
    for indicator in test_indicators:
        if 'campaign_name' in df.columns:
            test_campaigns = df[df['campaign_name'].str.lower().str.contains(indicator, na=False)]
            print(f"\nCampaigns with '{indicator}' in name: {len(test_campaigns):,}")
            for seed_id in test_campaigns['seed_list_id'].unique():
                print(f"- Seed ID {seed_id}: {len(test_campaigns[test_campaigns['seed_list_id'] == seed_id]):,} campaigns")
    
    # Analyze performance metrics distribution
    print("\n4. Performance Distribution:")
    metrics = ['sent', 'delivered', 'unique_opens', 'unique_clicks', 'total_revenue']
    for metric in metrics:
        print(f"\n{metric.title()} Statistics:")
        stats = df.groupby('seed_list_id')[metric].describe()
        print(stats.round(2))
    
    return {
        'total_campaigns': len(df),
        'seed_metrics': seed_metrics,
        'has_test_campaigns': any(df['campaign_name'].str.lower().str.contains('|'.join(test_indicators), na=False)) if 'campaign_name' in df.columns else None
    }

if __name__ == "__main__":
    results = analyze_seed_metrics('chia_campaigns.csv', 'lgd_campaigns.csv')
