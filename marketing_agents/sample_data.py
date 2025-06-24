"""
Sample Marketing Data Generator
Generates realistic sample data for testing the marketing agents system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_marketing_data(num_records: int = 100) -> pd.DataFrame:
    """
    Generate sample marketing data with the required structure:
    PUBLISHER, TARGET, COST PER SALE, CALL COUNT, SALES RATE, QUOTE RATE, SALES, QUOTED, 
    LEVEL 3 INTENT–HIGH, LEVEL 2 INTENT–MEDIUM, LEVEL 1 INTENT–LOW, AD MISLED, AD MISLED RATE, 
    DURATION, IVR, REACHED AGENT
    """
    
    # Define sample publishers and targets
    publishers = [
        'LeadGen Pro', 'CallSource Direct', 'Premium Leads', 'Quality Connect',
        'Lead Masters', 'Direct Response', 'Call Center Plus', 'Lead Factory',
        'Response Network', 'Lead Generation Co'
    ]
    
    targets = [
        'Insurance Quotes', 'Solar Installation', 'Home Security', 'Debt Consolidation',
        'Auto Insurance', 'Health Insurance', 'Home Improvement', 'Financial Services'
    ]
    
    data = []
    
    for _ in range(num_records):
        publisher = random.choice(publishers)
        target = random.choice(targets)
        
        # Generate realistic call volumes (some publishers are bigger than others)
        if publisher in ['LeadGen Pro', 'CallSource Direct', 'Premium Leads']:
            call_count = random.randint(50, 200)  # High volume publishers
        elif publisher in ['Quality Connect', 'Lead Masters']:
            call_count = random.randint(25, 75)   # Medium volume publishers
        else:
            call_count = random.randint(5, 30)    # Lower volume publishers
        
        # Generate intent distribution (Level 3 + Level 2 + Level 1 should roughly equal call count)
        level_3_pct = random.uniform(0.15, 0.35)  # 15-35% high intent
        level_2_pct = random.uniform(0.25, 0.45)  # 25-45% medium intent
        level_1_pct = max(0.2, 1 - level_3_pct - level_2_pct)  # Rest low intent
        
        level_3_calls = int(call_count * level_3_pct)
        level_2_calls = int(call_count * level_2_pct)
        level_1_calls = call_count - level_3_calls - level_2_calls
        
        # Generate conversion metrics (higher intent should correlate with higher conversion)
        base_conversion_rate = 0.08 + (level_3_pct * 0.15) + (level_2_pct * 0.08)
        conversion_rate = max(0.02, min(0.25, base_conversion_rate + random.uniform(-0.03, 0.03)))
        
        sales = int(call_count * conversion_rate)
        
        # Quote rate should be higher than sales rate
        quote_rate = conversion_rate + random.uniform(0.05, 0.15)
        quote_rate = min(0.4, quote_rate)
        quoted = int(call_count * quote_rate)
        
        # Cost per sale varies by publisher quality and target
        if target in ['Insurance Quotes', 'Financial Services']:
            base_cost = random.uniform(80, 150)  # Higher value targets
        else:
            base_cost = random.uniform(40, 100)  # Standard targets
        
        # Quality publishers might be more expensive but more efficient
        if publisher in ['Premium Leads', 'Quality Connect']:
            cost_per_sale = base_cost * random.uniform(1.1, 1.3)
        else:
            cost_per_sale = base_cost * random.uniform(0.8, 1.2)
        
        # Ad misled rate (should negatively correlate with conversion)
        ad_misled_rate = random.uniform(0.05, 0.25)
        ad_misled = int(call_count * ad_misled_rate)
        
        # Call duration (in minutes)
        duration = random.uniform(2, 15)
        
        # IVR and agent reach rates
        ivr_rate = random.uniform(0.7, 0.95)
        agent_reach_rate = random.uniform(0.6, 0.9)
        
        record = {
            'PUBLISHER': publisher,
            'TARGET': target,
            'COST PER SALE': round(cost_per_sale, 2),
            'CALL COUNT': call_count,
            'SALES RATE': round(conversion_rate * 100, 2),
            'QUOTE RATE': round(quote_rate * 100, 2),
            'SALES': sales,
            'QUOTED': quoted,
            'LEVEL 3 INTENT–HIGH': level_3_calls,
            'LEVEL 2 INTENT–MEDIUM': level_2_calls,
            'LEVEL 1 INTENT–LOW': level_1_calls,
            'AD MISLED': ad_misled,
            'AD MISLED RATE': round(ad_misled_rate * 100, 2),
            'DURATION': round(duration, 1),
            'IVR': round(ivr_rate * 100, 2),
            'REACHED AGENT': round(agent_reach_rate * 100, 2)
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # Add some realistic variation and outliers
    # Create a few high-performing outliers
    high_performers = df.sample(n=min(3, len(df)//10))
    for idx in high_performers.index:
        df.loc[idx, 'SALES RATE'] *= 1.5
        df.loc[idx, 'SALES'] = int(df.loc[idx, 'CALL COUNT'] * df.loc[idx, 'SALES RATE'] / 100)
        df.loc[idx, 'COST PER SALE'] *= 0.8  # More efficient
    
    # Create a few poor performers
    poor_performers = df.sample(n=min(2, len(df)//15))
    for idx in poor_performers.index:
        df.loc[idx, 'SALES RATE'] *= 0.3
        df.loc[idx, 'SALES'] = int(df.loc[idx, 'CALL COUNT'] * df.loc[idx, 'SALES RATE'] / 100)
        df.loc[idx, 'AD MISLED RATE'] *= 2  # Higher ad misled rate
        df.loc[idx, 'COST PER SALE'] *= 1.5  # Less efficient
    
    return df

def save_sample_data(filename: str = 'sample_marketing_data.csv', num_records: int = 100):
    """Generate and save sample data to CSV"""
    df = generate_sample_marketing_data(num_records)
    df.to_csv(filename, index=False)
    print(f"Generated {len(df)} records and saved to {filename}")
    return df

if __name__ == "__main__":
    # Generate sample data when run directly
    sample_df = save_sample_data('marketing_agents/sample_marketing_data.csv', 150)
    print("\nSample data preview:")
    print(sample_df.head())
    print(f"\nData shape: {sample_df.shape}")
    print(f"Publishers: {sample_df['PUBLISHER'].nunique()}")
    print(f"Total calls: {sample_df['CALL COUNT'].sum():,}")
    print(f"Overall conversion rate: {(sample_df['SALES'].sum() / sample_df['CALL COUNT'].sum() * 100):.2f}%") 