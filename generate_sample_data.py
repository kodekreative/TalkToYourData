#!/usr/bin/env python3
"""
Sample Data Generator for Performance Marketing Diagnostic Tool
Creates realistic test data with various scenarios and edge cases
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_sample_data(num_records=1000):
    """Generate sample performance marketing data"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define realistic values
    publishers = ['PublisherA', 'PublisherB', 'PublisherC', 'PublisherD', 'PublisherE', 'PublisherF']
    buyers = ['BuyerX', 'BuyerY', 'BuyerZ', 'BuyerW']
    targets = ['AutoInsurance', 'HomeInsurance', 'LifeInsurance', 'HealthInsurance']
    customer_intents = ['Level 1', 'Level 2', 'Level 3', 'Negative Intent', 'Not Detected']
    
    # Publisher quality profiles (affects lead quality)
    publisher_profiles = {
        'PublisherA': {'quality': 'high', 'ad_misled_rate': 0.02, 'strong_lead_rate': 0.6},
        'PublisherB': {'quality': 'medium', 'ad_misled_rate': 0.05, 'strong_lead_rate': 0.4},
        'PublisherC': {'quality': 'low', 'ad_misled_rate': 0.15, 'strong_lead_rate': 0.2},
        'PublisherD': {'quality': 'high', 'ad_misled_rate': 0.03, 'strong_lead_rate': 0.55},
        'PublisherE': {'quality': 'medium', 'ad_misled_rate': 0.08, 'strong_lead_rate': 0.35},
        'PublisherF': {'quality': 'low', 'ad_misled_rate': 0.20, 'strong_lead_rate': 0.15}
    }
    
    # Buyer performance profiles (affects conversion)
    buyer_profiles = {
        'BuyerX': {'efficiency': 'high', 'agent_availability': 0.95, 'conversion_skill': 0.8},
        'BuyerY': {'efficiency': 'medium', 'agent_availability': 0.85, 'conversion_skill': 0.6},
        'BuyerZ': {'efficiency': 'low', 'agent_availability': 0.70, 'conversion_skill': 0.4},
        'BuyerW': {'efficiency': 'medium', 'agent_availability': 0.80, 'conversion_skill': 0.65}
    }
    
    data = []
    
    for i in range(num_records):
        # Select random publisher and buyer
        publisher = random.choice(publishers)
        buyer = random.choice(buyers)
        target = random.choice(targets)
        
        # Get profiles
        pub_profile = publisher_profiles[publisher]
        buyer_profile = buyer_profiles[buyer]
        
        # Generate customer intent based on publisher quality
        if pub_profile['quality'] == 'high':
            intent_weights = [0.2, 0.3, 0.35, 0.1, 0.05]  # More Level 2 and 3
        elif pub_profile['quality'] == 'medium':
            intent_weights = [0.4, 0.25, 0.2, 0.1, 0.05]  # Balanced
        else:
            intent_weights = [0.5, 0.2, 0.1, 0.15, 0.05]  # More Level 1 and Negative
        
        customer_intent = np.random.choice(customer_intents, p=intent_weights)
        
        # Generate ad misled based on publisher
        ad_misled = 'Yes' if random.random() < pub_profile['ad_misled_rate'] else 'No'
        
        # Generate agent availability based on buyer
        reached_agent = 'Yes' if random.random() < buyer_profile['agent_availability'] else 'No'
        
        # Generate IVR (inverse correlation with agent availability)
        ivr_rate = 1 - buyer_profile['agent_availability']
        ivr = 'Yes' if random.random() < ivr_rate else 'No'
        
        # Generate billable (poor quality leads less likely to be billable)
        if customer_intent in ['Level 2', 'Level 3']:
            billable_rate = 0.95
        elif customer_intent == 'Level 1':
            billable_rate = 0.80
        else:
            billable_rate = 0.30
        
        billable = 'Yes' if random.random() < billable_rate else 'No'
        
        # Generate duration (higher for better leads and buyers)
        base_duration = 180  # 3 minutes baseline
        
        if customer_intent == 'Level 3':
            duration_multiplier = 1.5
        elif customer_intent == 'Level 2':
            duration_multiplier = 1.2
        elif customer_intent == 'Level 1':
            duration_multiplier = 1.0
        else:
            duration_multiplier = 0.6
        
        # Buyer efficiency affects duration
        if buyer_profile['efficiency'] == 'high':
            duration_multiplier *= 1.3
        elif buyer_profile['efficiency'] == 'medium':
            duration_multiplier *= 1.0
        else:
            duration_multiplier *= 0.8
        
        duration = int(base_duration * duration_multiplier * (0.5 + random.random()))
        
        # Generate objection with no rebuttal (inverse of buyer skill)
        rebuttal_skill = buyer_profile['conversion_skill']
        objection_no_rebuttal = 'Yes' if random.random() > rebuttal_skill else 'No'
        
        # Generate sale based on multiple factors
        sale_probability = 0.1  # Base probability
        
        # Customer intent impact
        if customer_intent == 'Level 3':
            sale_probability *= 6
        elif customer_intent == 'Level 2':
            sale_probability *= 3
        elif customer_intent == 'Level 1':
            sale_probability *= 1.5
        else:
            sale_probability *= 0.2
        
        # Buyer skill impact
        sale_probability *= buyer_profile['conversion_skill'] * 2
        
        # Agent availability impact
        if reached_agent == 'No':
            sale_probability *= 0.1
        
        # Ad misled impact
        if ad_misled == 'Yes':
            sale_probability *= 0.3
        
        # Duration impact (longer calls more likely to convert)
        if duration > 300:
            sale_probability *= 1.4
        elif duration < 120:
            sale_probability *= 0.6
        
        # Cap probability
        sale_probability = min(sale_probability, 0.85)
        
        sale = 'Yes' if random.random() < sale_probability else 'No'
        
        # Generate quote (intermediate step)
        if reached_agent == 'Yes' and customer_intent in ['Level 2', 'Level 3']:
            quote_probability = 0.7
        elif reached_agent == 'Yes' and customer_intent == 'Level 1':
            quote_probability = 0.4
        else:
            quote_probability = 0.1
        
        quote = 'Yes' if random.random() < quote_probability else 'No'
        
        # Generate stage progression (if applicable)
        stage_1 = 'Yes' if reached_agent == 'Yes' else 'No'
        stage_2 = 'Yes' if stage_1 == 'Yes' and random.random() < 0.7 else 'No'
        stage_3 = 'Yes' if stage_2 == 'Yes' and random.random() < 0.5 else 'No'
        stage_4 = 'Yes' if stage_3 == 'Yes' and random.random() < 0.4 else 'No'
        stage_5 = 'Yes' if stage_4 == 'Yes' and sale == 'Yes' else 'No'
        
        # Create record
        record = {
            'PUBLISHER': publisher,
            'BUYER': buyer,
            'TARGET': target,
            'CUSTOMER_INTENT': customer_intent,
            'SALE': sale,
            'QUOTE': quote,
            'REACHED_AGENT': reached_agent,
            'AD_MISLED': ad_misled,
            'BILLABLE': billable,
            'DURATION': duration,
            'IVR': ivr,
            'OBJECTION_WITH_NO_REBUTTAL': objection_no_rebuttal,
            'STAGE_1': stage_1,
            'STAGE_2': stage_2,
            'STAGE_3': stage_3,
            'STAGE_4': stage_4,
            'STAGE_5': stage_5,
            'CALL_DATE': datetime.now() - timedelta(days=random.randint(0, 30)),
            'LEAD_ID': f"LEAD_{i+1:06d}"
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some edge cases
    
    # Add some records with extreme ad misled issues for PublisherC
    extreme_misled = df[df['PUBLISHER'] == 'PublisherC'].sample(n=min(20, len(df[df['PUBLISHER'] == 'PublisherC'])))
    df.loc[extreme_misled.index, 'AD_MISLED'] = 'Yes'
    df.loc[extreme_misled.index, 'SALE'] = 'No'
    
    # Add some Level 3 leads that didn't convert (training issue)
    level_3_missed = df[(df['CUSTOMER_INTENT'] == 'Level 3') & (df['BUYER'] == 'BuyerZ')].sample(n=min(10, len(df[(df['CUSTOMER_INTENT'] == 'Level 3') & (df['BUYER'] == 'BuyerZ')])))
    df.loc[level_3_missed.index, 'SALE'] = 'No'
    df.loc[level_3_missed.index, 'OBJECTION_WITH_NO_REBUTTAL'] = 'Yes'
    
    return df

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample performance marketing data...")
    
    sample_data = generate_sample_data(1000)
    
    # Save to Excel
    output_file = "sample_performance_data.xlsx"
    sample_data.to_excel(output_file, index=False)
    
    print(f"Sample data saved to {output_file}")
    print(f"Generated {len(sample_data)} records")
    
    # Show basic statistics
    print("\nBasic Statistics:")
    print(f"Overall Conversion Rate: {(sample_data['SALE'] == 'Yes').mean():.1%}")
    print(f"Ad Misled Rate: {(sample_data['AD_MISLED'] == 'Yes').mean():.1%}")
    print(f"Agent Availability: {(sample_data['REACHED_AGENT'] == 'Yes').mean():.1%}")
    print(f"Strong Leads (Level 2+3): {sample_data['CUSTOMER_INTENT'].isin(['Level 2', 'Level 3']).mean():.1%}")
    
    print("\nConversion Rate by Publisher:")
    print(sample_data.groupby('PUBLISHER')['SALE'].apply(lambda x: (x == 'Yes').mean()).sort_values(ascending=False))
    
    print("\nConversion Rate by Buyer:")
    print(sample_data.groupby('BUYER')['SALE'].apply(lambda x: (x == 'Yes').mean()).sort_values(ascending=False)) 