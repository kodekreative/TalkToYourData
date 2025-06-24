import os
import tempfile
import pandas as pd
import numpy as np
import json
from pathlib import Path
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import pygame

# Load business context configuration
def load_business_config():
    """Load business metrics and field mappings from config files"""
    config = {}
    
    # Load business metrics
    business_metrics_path = Path("business_metrics.json")
    if business_metrics_path.exists():
        with open(business_metrics_path, 'r') as f:
            config['business_metrics'] = json.load(f)
    
    # Load business terms
    business_terms_path = Path("config/business_terms.json")
    if business_terms_path.exists():
        with open(business_terms_path, 'r') as f:
            config['business_terms'] = json.load(f)
    
    return config

def find_column_matches(df_columns, business_config):
    """Find which business columns match actual dataframe columns using user's mappings"""
    if 'business_metrics' not in business_config:
        return {}
    
    column_mappings = business_config['business_metrics'].get('column_mappings', {})
    matches = {}
    
    df_columns_lower = [col.lower() for col in df_columns]
    
    for business_col, synonyms in column_mappings.items():
        for df_col in df_columns:
            df_col_lower = df_col.lower()
            if any(synonym.lower() in df_col_lower for synonym in synonyms):
                matches[business_col] = df_col
                break
    
    return matches

def calculate_business_metric(df, metric_name, metric_config, column_matches):
    """Calculate a business metric using the user's defined formulas"""
    try:
        required_cols = metric_config.get('required_columns', [])
        
        # Check if we have the required columns
        available_cols = {}
        for req_col in required_cols:
            if req_col in column_matches:
                available_cols[req_col] = column_matches[req_col]
            else:
                return None
        
        # Apply the metric calculation based on the formula
        if metric_name == 'conversion_rate' and 'sale' in available_cols:
            sale_col = available_cols['sale']
            yes_sales = len(df[df[sale_col].astype(str).str.lower().isin(['yes', 'true', '1', 'sale'])])
            total = len(df)
            return (yes_sales / total * 100) if total > 0 else 0
        
        elif metric_name == 'average_revenue' and 'revenue' in available_cols:
            try:
                revenue_series = df[available_cols['revenue']].astype(str).str.replace('$', '').str.replace(',', '')
                revenue_numeric = pd.to_numeric(revenue_series, errors='coerce').fillna(0)
                return revenue_numeric.mean()
            except:
                return None
        
        elif metric_name == 'total_revenue' and 'revenue' in available_cols:
            try:
                revenue_series = df[available_cols['revenue']].astype(str).str.replace('$', '').str.replace(',', '')
                revenue_numeric = pd.to_numeric(revenue_series, errors='coerce').fillna(0)
                return revenue_numeric.sum()
            except:
                return None
        
        elif metric_name == 'cost_per_lead' and 'payout' in available_cols and 'call_id' in available_cols:
            try:
                payout_series = df[available_cols['payout']].astype(str).str.replace('$', '').str.replace(',', '')
                payout_numeric = pd.to_numeric(payout_series, errors='coerce').fillna(0)
                total_payout = payout_numeric.sum()
                total_calls = len(df)
                return total_payout / total_calls if total_calls > 0 else 0
            except:
                return None
        
        # Add more metric calculations as needed based on your business_metrics.json
        
    except Exception as e:
        return None
    
    return None

# Performance Marketing Analysis Dictionary - Business Logic Reference
BUSINESS_REFERENCE = {
    "success_metrics": {
        "conversion_rate": {
            "formula": "(Number of Sales ÷ Number of Leads) × 100",
            "success_threshold": "5-15% (industry dependent)",
            "primary_kpi": True
        }
    },
    "quality_indicators": {
        "customer_intent": {
            "high_quality": ["Level 2", "Level 3"],
            "low_quality": ["Level 1", "Negative Intent", "Not Detected"],
            "critical_rule": "Level 3 non-conversions indicate sales execution problems"
        },
        "billable": {
            "good_threshold": 0.70,
            "action_trigger": "Low billable rates require publisher review"
        },
        "duration": {
            "good_threshold": 300,  # 5 minutes in seconds
            "benchmark": ">5 minutes indicates qualified interest"
        },
        "ad_misled": {
            "tolerance": "zero",
            "action": "Immediate publisher review and correction"
        }
    },
    "execution_indicators": {
        "quote_to_call_ratio": {
            "benchmark": "40-60%",
            "formula": "(QUOTE = 'Yes' count ÷ Total Calls) × 100"
        },
        "quote_to_sale_ratio": {
            "benchmark": ">30%",
            "formula": "(SALE = 'Yes' count ÷ QUOTE = 'Yes' count) × 100"
        },
        "reached_agent": {
            "critical_threshold": 0.90,
            "escalation": "Immediate management attention for low rates"
        }
    },
    "critical_thresholds": {
        "agent_availability_crisis": 0.90,
        "strong_publisher_billable": 0.70,
        "strong_publisher_intent": 0.30,
        "strong_buyer_reach": 0.90,
        "strong_buyer_quote_to_sale": 0.30,
        "ad_misled_tolerance": 0.05
    },
    "stage_progression": {
        "stage_5": "Highest quality outcome - Enrollment",
        "stage_4": "Strong engagement - Plan Detail", 
        "stage_3": "Good qualification - Needs Analysis",
        "stage_2": "Basic qualification - Eligibility",
        "stage_1": "Initial contact only - Introduction"
    }
}

def analyze_customer_intent_quality(df, intent_col):
    """Analyze customer intent using business rules"""
    high_quality_keywords = ['level 2', 'level 3', '2', '3']
    
    # Count high quality leads
    high_quality_mask = df[intent_col].astype(str).str.lower().str.contains('|'.join(high_quality_keywords), na=False)
    high_quality_count = high_quality_mask.sum()
    high_quality_rate = (high_quality_count / len(df)) * 100
    
    # Analyze Level 3 non-conversions (critical business rule)
    level_3_mask = df[intent_col].astype(str).str.lower().str.contains('level 3|3', na=False)
    
    findings = []
    findings.append(f"Lead Quality Analysis:")
    findings.append(f"  • High-intent leads (Level 2+3): {high_quality_rate:.1f}% ({high_quality_count}/{len(df)})")
    
    # Quality assessment
    if high_quality_rate >= 30:
        findings.append(f"  ✅ Strong lead quality - above 30% threshold")
    else:
        findings.append(f"  ⚠️ Below quality threshold (30%) - publisher review needed")
    
    return findings, high_quality_rate, level_3_mask

def analyze_publisher_quality_score(df, column_matches):
    """Calculate Publisher Quality Index using business formula"""
    if not all(col in column_matches for col in ['customer_intent', 'billable']):
        return None, []
    
    findings = []
    intent_col = column_matches['customer_intent']
    billable_col = column_matches['billable']
    
    # Publisher quality analysis
    if 'publisher' in column_matches:
        pub_col = column_matches['publisher']
        
        publisher_analysis = []
        for publisher in df[pub_col].unique():
            pub_data = df[df[pub_col] == publisher]
            
            # Calculate components
            intent_findings, intent_rate, _ = analyze_customer_intent_quality(pub_data, intent_col)
            billable_rate = pub_data[billable_col].astype(str).str.lower().isin(['yes', 'true', '1']).mean() * 100
            
            # Check for ad misled violations
            ad_misled_count = 0
            if 'ad_misled' in column_matches:
                ad_misled_col = column_matches['ad_misled']
                ad_misled_count = pub_data[ad_misled_col].astype(str).str.lower().isin(['yes', 'true', '1']).sum()
            
            # Quality assessment
            quality_status = "✅ Strong Publisher" if (billable_rate >= 70 and intent_rate >= 30) else "⚠️ Needs Review"
            
            publisher_analysis.append({
                'publisher': publisher,
                'billable_rate': billable_rate,
                'intent_rate': intent_rate,
                'ad_misled_count': ad_misled_count,
                'status': quality_status
            })
        
        # Sort by quality
        publisher_analysis.sort(key=lambda x: (x['billable_rate'] + x['intent_rate']), reverse=True)
        
        findings.append("Publisher Quality Ranking:")
        for pub in publisher_analysis[:5]:  # Top 5
            findings.append(f"  • {pub['publisher']}: {pub['billable_rate']:.1f}% billable, {pub['intent_rate']:.1f}% high-intent {pub['status']}")
            if pub['ad_misled_count'] > 0:
                findings.append(f"    🚨 CRITICAL: {pub['ad_misled_count']} ad misled violations")
    
    return publisher_analysis, findings

def analyze_sales_execution_problems(df, column_matches):
    """Identify sales execution vs lead quality issues"""
    findings = []
    
    if not all(col in column_matches for col in ['sale', 'customer_intent']):
        return findings
    
    sale_col = column_matches['sale']
    intent_col = column_matches['customer_intent']
    
    # Critical Level 3 analysis
    level_3_mask = df[intent_col].astype(str).str.lower().str.contains('level 3|3', na=False)
    level_3_data = df[level_3_mask]
    
    if len(level_3_data) > 0:
        level_3_sales = level_3_data[sale_col].astype(str).str.lower().isin(['yes', 'true', '1', 'sale']).sum()
        level_3_conversion = (level_3_sales / len(level_3_data)) * 100
        
        findings.append(f"🎯 Level 3 Intent Analysis (Critical):")
        findings.append(f"  • Level 3 leads: {len(level_3_data)} total")
        findings.append(f"  • Level 3 conversions: {level_3_sales} ({level_3_conversion:.1f}%)")
        
        if level_3_conversion < 50:  # Level 3 should convert at high rates
            findings.append(f"  🚨 SALES EXECUTION PROBLEM: Level 3 leads not converting properly")
        
    # Quote-to-sale analysis
    if 'quote' in column_matches:
        quote_col = column_matches['quote']
        quoted_leads = df[df[quote_col].astype(str).str.lower().isin(['yes', 'true', '1'])]
        
        if len(quoted_leads) > 0:
            quote_sales = quoted_leads[sale_col].astype(str).str.lower().isin(['yes', 'true', '1', 'sale']).sum()
            quote_to_sale_rate = (quote_sales / len(quoted_leads)) * 100
            
            findings.append(f"💼 Sales Execution Analysis:")
            findings.append(f"  • Quote-to-sale conversion: {quote_to_sale_rate:.1f}%")
            
            if quote_to_sale_rate < 30:
                findings.append(f"  ⚠️ Below benchmark (30%) - sales training needed")
    
    # Agent availability analysis
    if 'reached_agent' in column_matches:
        agent_col = column_matches['reached_agent']
        agent_reach_rate = df[agent_col].astype(str).str.lower().isin(['yes', 'true', '1']).mean() * 100
        
        findings.append(f"👥 Agent Availability:")
        findings.append(f"  • Agent reach rate: {agent_reach_rate:.1f}%")
        
        if agent_reach_rate < 90:
            findings.append(f"  🚨 CAPACITY CRISIS: Below 90% threshold - immediate attention needed")
    
    return findings

def identify_successful_combinations(df, column_matches):
    """Identify successful Publisher-Buyer combinations"""
    findings = []
    
    if not all(col in column_matches for col in ['publisher', 'buyer', 'sale']):
        return findings
    
    pub_col = column_matches['publisher']
    buyer_col = column_matches['buyer']
    sale_col = column_matches['sale']
    
    # Combination analysis
    combination_stats = []
    
    for pub in df[pub_col].unique():
        for buyer in df[buyer_col].unique():
            combo_data = df[(df[pub_col] == pub) & (df[buyer_col] == buyer)]
            
            if len(combo_data) >= 5:  # Minimum sample size
                sales = combo_data[sale_col].astype(str).str.lower().isin(['yes', 'true', '1', 'sale']).sum()
                conversion_rate = (sales / len(combo_data)) * 100
                
                # Quality indicators
                quality_score = 0
                if 'customer_intent' in column_matches:
                    intent_col = column_matches['customer_intent']
                    _, high_intent_rate, _ = analyze_customer_intent_quality(combo_data, intent_col)
                    quality_score += high_intent_rate
                
                if 'billable' in column_matches:
                    billable_col = column_matches['billable']
                    billable_rate = combo_data[billable_col].astype(str).str.lower().isin(['yes', 'true', '1']).mean() * 100
                    quality_score += billable_rate
                
                combination_stats.append({
                    'publisher': pub,
                    'buyer': buyer,
                    'conversion_rate': conversion_rate,
                    'total_leads': len(combo_data),
                    'total_sales': sales,
                    'quality_score': quality_score / 2 if quality_score > 0 else 0
                })
    
    # Sort by conversion rate
    combination_stats.sort(key=lambda x: x['conversion_rate'], reverse=True)
    
    findings.append("🏆 Successful Publisher-Buyer Combinations:")
    
    # Top performers
    for combo in combination_stats[:3]:
        if combo['conversion_rate'] > 5:  # Above minimum threshold
            findings.append(f"  • {combo['publisher']} → {combo['buyer']}: {combo['conversion_rate']:.1f}% conversion ({combo['total_sales']}/{combo['total_leads']})")
    
    # Problem combinations
    problem_combos = [c for c in combination_stats if c['conversion_rate'] < 2 and c['total_leads'] >= 10]
    if problem_combos:
        findings.append("⚠️ Combinations Needing Review:")
        for combo in problem_combos[:2]:
            findings.append(f"  • {combo['publisher']} → {combo['buyer']}: {combo['conversion_rate']:.1f}% conversion")
    
    return findings

# --- Agent Definitions ---

class Agent:
    def analyze(self, data):
        raise NotImplementedError

class BusinessContextAgent(Agent):
    """Base agent that uses the user's business context and field mappings"""
    
    def __init__(self):
        self.business_config = load_business_config()
    
    def analyze(self, data):
        if isinstance(data, pd.DataFrame):
            return self.analyze_dataframe(data)
        elif isinstance(data, dict):
            return self.analyze_dict(data)
        else:
            return {"findings": ["Unsupported data format"]}
    
    def analyze_dataframe(self, df):
        # Find column matches using user's business mappings
        self.column_matches = find_column_matches(df.columns.tolist(), self.business_config)
        return self._perform_analysis(df)
    
    def analyze_dict(self, data):
        # Original logic for demo data
        return self._perform_analysis_dict(data)
    
    def _perform_analysis(self, df):
        raise NotImplementedError
    
    def _perform_analysis_dict(self, data):
        raise NotImplementedError

class SalesPerformanceAgent(BusinessContextAgent):
    """Analyzes sales performance using comprehensive business context"""
    
    def _perform_analysis(self, df):
        findings = []
        
        if not self.column_matches:
            findings.append("No recognized business columns found. Analysis limited to generic data exploration.")
            return {"sales_findings": findings}
        
        findings.append(f"📊 Sales Performance Analysis ({len(df):,} records)")
        findings.append(f"🎯 Recognized fields: {', '.join(self.column_matches.keys())}")
        
        # Overall conversion analysis
        if 'sale' in self.column_matches:
            sale_col = self.column_matches['sale']
            total_sales = df[sale_col].astype(str).str.lower().isin(['yes', 'true', '1', 'sale']).sum()
            conversion_rate = (total_sales / len(df)) * 100
            
            findings.append(f"💰 Overall Performance:")
            findings.append(f"  • Total sales: {total_sales:,} ({conversion_rate:.1f}% conversion rate)")
            
            # Benchmark assessment
            if conversion_rate >= 15:
                findings.append(f"  ✅ Excellent conversion rate")
            elif conversion_rate >= 5:
                findings.append(f"  ✅ Good conversion rate")
            else:
                findings.append(f"  ⚠️ Below industry benchmark (5-15%)")
        
        # Revenue analysis (if available)
        if 'revenue' in self.column_matches:
            revenue_col = self.column_matches['revenue']
            try:
                # Clean and convert revenue data
                revenue_series = df[revenue_col].astype(str).str.replace('$', '').str.replace(',', '')
                # Try to convert to numeric, setting invalid values to 0
                revenue_numeric = pd.to_numeric(revenue_series, errors='coerce').fillna(0)
                total_revenue = revenue_numeric.sum()
                avg_revenue = revenue_numeric.mean()
                
                findings.append(f"💵 Revenue Analysis:")
                findings.append(f"  • Total revenue: ${total_revenue:,.2f}")
                findings.append(f"  • Average revenue per record: ${avg_revenue:.2f}")
            except Exception as e:
                findings.append(f"💵 Revenue Analysis: Unable to process revenue data (data format issues)")
        
        # Sales execution problem analysis
        execution_findings = analyze_sales_execution_problems(df, self.column_matches)
        findings.extend(execution_findings)
        
        # Successful combination analysis
        combination_findings = identify_successful_combinations(df, self.column_matches)
        findings.extend(combination_findings)
        
        return {"sales_findings": findings}
    
    def _perform_analysis_dict(self, data):
        # Original logic for demo data
        buyers = data.get("buyers", [])
        findings = []
        for buyer in buyers:
            if buyer.get("availability", 1) < 0.9:
                findings.append(
                    f"{buyer['name']} shows {int((1-buyer['availability'])*100)}% agent availability issues "
                    f"with quote-to-sale rate {buyer.get('quote_to_sale', 0)*100:.1f}%."
                )
        return {"sales_findings": findings}

class QualityAnalysisAgent(BusinessContextAgent):
    """Analyzes data quality and lead quality using comprehensive business context"""
    
    def _perform_analysis(self, df):
        findings = []
        
        # Data completeness
        total_rows = len(df)
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        findings.append(f"📋 Data Quality Assessment:")
        findings.append(f"  • Dataset: {total_rows:,} records × {len(df.columns)} columns")
        findings.append(f"  • Data completeness: {100-missing_pct:.1f}%")
        findings.append(f"  • Business fields mapped: {len(self.column_matches)}")
        
        if not self.column_matches:
            findings.append("⚠️ No business context available - limited analysis")
            return {"quality_findings": findings}
        
        # Lead quality analysis using business rules
        if 'customer_intent' in self.column_matches:
            intent_col = self.column_matches['customer_intent']
            intent_findings, intent_rate, level_3_mask = analyze_customer_intent_quality(df, intent_col)
            findings.extend(intent_findings)
        
        # Publisher quality analysis
        publisher_analysis, pub_findings = analyze_publisher_quality_score(df, self.column_matches)
        findings.extend(pub_findings)
        
        # Ad compliance violations (critical)
        if 'ad_misled' in self.column_matches:
            ad_misled_col = self.column_matches['ad_misled']
            violations = df[ad_misled_col].astype(str).str.lower().isin(['yes', 'true', '1']).sum()
            violation_rate = (violations / len(df)) * 100
            
            findings.append(f"🚨 Compliance Analysis:")
            findings.append(f"  • Ad misled violations: {violations} ({violation_rate:.1f}%)")
            
            if violations > 0:
                findings.append(f"  ⚠️ IMMEDIATE ACTION REQUIRED: Zero tolerance policy")
                
                # Identify violating publishers
                if 'publisher' in self.column_matches:
                    pub_col = self.column_matches['publisher']
                    violating_pubs = df[df[ad_misled_col].astype(str).str.lower().isin(['yes', 'true', '1'])][pub_col].value_counts()
                    findings.append(f"  • Violating publishers: {list(violating_pubs.head(3).index)}")
        
        # Duration analysis
        if 'duration' in self.column_matches:
            duration_col = self.column_matches['duration']
            try:
                # Handle duration in various formats
                duration_data = df[duration_col].dropna()
                if len(duration_data) > 0:
                    # Try different approaches for duration parsing
                    avg_duration = None
                    
                    # Try as timedelta first
                    try:
                        avg_duration = pd.to_timedelta(duration_data, errors='coerce').dt.total_seconds().mean()
                    except:
                        pass
                    
                    # If that fails, try parsing MM:SS or HH:MM:SS format
                    if avg_duration is None or pd.isna(avg_duration):
                        try:
                            # Convert time strings to seconds
                            time_seconds = []
                            for time_str in duration_data.astype(str):
                                parts = time_str.split(':')
                                if len(parts) == 2:  # MM:SS
                                    minutes, seconds = int(parts[0]), int(parts[1])
                                    time_seconds.append(minutes * 60 + seconds)
                                elif len(parts) == 3:  # HH:MM:SS
                                    hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
                                    time_seconds.append(hours * 3600 + minutes * 60 + seconds)
                            
                            if time_seconds:
                                avg_duration = sum(time_seconds) / len(time_seconds)
                        except:
                            pass
                    
                    if avg_duration and avg_duration > 0:
                        findings.append(f"⏱️ Engagement Quality:")
                        findings.append(f"  • Average call duration: {avg_duration/60:.1f} minutes")
                        
                        if avg_duration >= 300:  # 5 minutes
                            findings.append(f"  ✅ Good engagement - above 5-minute benchmark")
                        else:
                            findings.append(f"  ⚠️ Short calls - may indicate quality issues")
            except Exception as e:
                findings.append(f"⏱️ Duration analysis: Unable to process duration data")
        
        return {"quality_findings": findings}
    
    def _perform_analysis_dict(self, data):
        # Original logic for demo data
        publishers = data.get("publishers", [])
        findings = []
        for pub in publishers:
            if pub.get("ad_misled_rate", 0) > 0.1:
                findings.append(
                    f"{pub['name']} has {pub['ad_misled_rate']*100:.1f}% ad misled rate but "
                    f"{pub.get('billable_rate', 0)*100:.1f}% billable rate."
                )
        return {"quality_findings": findings}

class ComparativeAnalysisAgent(BusinessContextAgent):
    """Performs sophisticated comparative analysis using business context"""
    
    def _perform_analysis(self, df):
        findings = []
        
        if not self.column_matches:
            findings.append("Insufficient business context for comparative analysis")
            return {"comparative_findings": findings}
        
        findings.append(f"📈 Comparative Performance Analysis:")
        
        # Publisher performance comparison
        if 'publisher' in self.column_matches:
            pub_col = self.column_matches['publisher']
            
            # Performance by publisher
            pub_stats = []
            for pub in df[pub_col].unique():
                pub_data = df[df[pub_col] == pub]
                
                stats = {'publisher': pub, 'total_leads': len(pub_data)}
                
                if 'sale' in self.column_matches:
                    sale_col = self.column_matches['sale']
                    sales = pub_data[sale_col].astype(str).str.lower().isin(['yes', 'true', '1', 'sale']).sum()
                    stats['conversion_rate'] = (sales / len(pub_data)) * 100
                
                if 'customer_intent' in self.column_matches:
                    intent_col = self.column_matches['customer_intent']
                    _, intent_rate, _ = analyze_customer_intent_quality(pub_data, intent_col)
                    stats['intent_quality'] = intent_rate
                
                pub_stats.append(stats)
            
            # Sort by conversion rate
            pub_stats.sort(key=lambda x: x.get('conversion_rate', 0), reverse=True)
            
            findings.append(f"🏅 Publisher Performance Ranking:")
            for i, pub in enumerate(pub_stats[:3], 1):
                conv_rate = pub.get('conversion_rate', 0)
                findings.append(f"  {i}. {pub['publisher']}: {conv_rate:.1f}% conversion ({pub['total_leads']} leads)")
                
                # Quality indicator
                if 'intent_quality' in pub:
                    quality_indicator = "🟢" if pub['intent_quality'] >= 30 else "🟡" if pub['intent_quality'] >= 15 else "🔴"
                    findings.append(f"     {quality_indicator} Lead quality: {pub['intent_quality']:.1f}%")
        
        # Buyer efficiency comparison
        if 'buyer' in self.column_matches:
            buyer_col = self.column_matches['buyer']
            
            buyer_efficiency = []
            for buyer in df[buyer_col].unique():
                buyer_data = df[df[buyer_col] == buyer]
                
                efficiency = {'buyer': buyer, 'total_leads': len(buyer_data)}
                
                # Agent reach rate
                if 'reached_agent' in self.column_matches:
                    agent_col = self.column_matches['reached_agent']
                    reach_rate = buyer_data[agent_col].astype(str).str.lower().isin(['yes', 'true', '1']).mean() * 100
                    efficiency['agent_reach'] = reach_rate
                
                # Quote-to-sale efficiency
                if all(col in self.column_matches for col in ['quote', 'sale']):
                    quote_col = self.column_matches['quote']
                    sale_col = self.column_matches['sale']
                    
                    quoted = buyer_data[buyer_data[quote_col].astype(str).str.lower().isin(['yes', 'true', '1'])]
                    if len(quoted) > 0:
                        quote_sales = quoted[sale_col].astype(str).str.lower().isin(['yes', 'true', '1', 'sale']).sum()
                        efficiency['quote_to_sale'] = (quote_sales / len(quoted)) * 100
                
                buyer_efficiency.append(efficiency)
            
            # Sort by agent reach rate
            buyer_efficiency.sort(key=lambda x: x.get('agent_reach', 0), reverse=True)
            
            findings.append(f"🎯 Buyer Efficiency Analysis:")
            for buyer in buyer_efficiency[:3]:
                findings.append(f"  • {buyer['buyer']}: {buyer['total_leads']} leads")
                
                if 'agent_reach' in buyer:
                    reach_status = "✅" if buyer['agent_reach'] >= 90 else "⚠️"
                    findings.append(f"    {reach_status} Agent reach: {buyer['agent_reach']:.1f}%")
                
                if 'quote_to_sale' in buyer:
                    close_status = "✅" if buyer['quote_to_sale'] >= 30 else "⚠️"
                    findings.append(f"    {close_status} Closing rate: {buyer['quote_to_sale']:.1f}%")
        
        # Identify problem areas requiring immediate attention
        findings.append(f"🚨 Immediate Action Items:")
        action_items = []
        
        # Check for critical thresholds
        if 'reached_agent' in self.column_matches:
            agent_col = self.column_matches['reached_agent']
            overall_reach = df[agent_col].astype(str).str.lower().isin(['yes', 'true', '1']).mean() * 100
            if overall_reach < 90:
                action_items.append(f"Agent availability crisis: {overall_reach:.1f}% reach rate")
        
        if 'ad_misled' in self.column_matches:
            ad_col = self.column_matches['ad_misled']
            violations = df[ad_col].astype(str).str.lower().isin(['yes', 'true', '1']).sum()
            if violations > 0:
                action_items.append(f"Compliance violations: {violations} ad misled cases")
        
        if action_items:
            for item in action_items:
                findings.append(f"  • {item}")
        else:
            findings.append(f"  • No critical issues requiring immediate attention")
        
        return {"comparative_findings": findings}
    
    def _perform_analysis_dict(self, data):
        # Original logic for demo data
        leads = data.get("leads", [])
        findings = []
        for lead in leads:
            if lead.get("intent_level_2_3", 0) > 0.4 and lead.get("conversion", 0) < 0.02:
                findings.append(
                    f"{lead['name']} generates {lead['intent_level_2_3']*100:.1f}% Level 2/3 intent leads "
                    f"but converts at only {lead['conversion']*100:.1f}%."
                )
        return {"comparative_findings": findings}

class ExecutiveSummaryAgent(Agent):
    def analyze(self, data):
        return {}
    
    def synthesize(self, results):
        summary = []
        
        # Sales performance highlights
        if results.get("sales_performance", {}).get("sales_findings"):
            sales_findings = results["sales_performance"]["sales_findings"]
            if sales_findings and "No recognized business columns" not in sales_findings[0]:
                # Extract key metrics
                for finding in sales_findings[:4]:
                    if "Total sales:" in finding or "conversion rate" in finding.lower() or "revenue:" in finding:
                        summary.append(f"📊 {finding}")
        
        # Quality issues and compliance
        if results.get("quality_analysis", {}).get("quality_findings"):
            quality_findings = results["quality_analysis"]["quality_findings"]
            for finding in quality_findings:
                if "IMMEDIATE ACTION" in finding or "violations:" in finding or "CRITICAL" in finding:
                    summary.append(f"🚨 {finding}")
        
        # Performance insights
        if results.get("comparative_analysis", {}).get("comparative_findings"):
            comp_findings = results["comparative_analysis"]["comparative_findings"]
            if comp_findings and "Insufficient business context" not in comp_findings[0]:
                # Extract action items
                for finding in comp_findings:
                    if "Immediate Action" in finding or "crisis:" in finding or "violations:" in finding:
                        summary.append(f"⚠️ {finding}")
        
        if not summary:
            return "Analysis complete using Performance Marketing Analysis Dictionary. Upload data with recognized business fields (publisher, buyer, sale, customer_intent, etc.) for detailed insights following industry best practices."
        
        summary.append("💡 Executive Summary: Analysis follows Performance Marketing Analysis Dictionary standards for accurate publisher vs buyer issue identification.")
        return "\n".join(summary)

# --- Orchestrator ---

class Orchestrator:
    def __init__(self, agents):
        self.agents = agents
    
    def run_comprehensive_analysis(self, data, focus_areas=None):
        results = {}
        for name, agent in self.agents.items():
            if not focus_areas or name in focus_areas:
                results[name] = agent.analyze(data)
        return results
    
    def get_summary_for_voice(self, results):
        return self.agents['executive_summary'].synthesize(results)

# No demo data - only analyze uploaded files

# --- Web UI Integration Function ---

def run_selected_analysis(data, selected_agents):
    agents = {
        "sales_performance": SalesPerformanceAgent(),
        "quality_analysis": QualityAnalysisAgent(),  
        "comparative_analysis": ComparativeAnalysisAgent(),
        "executive_summary": ExecutiveSummaryAgent(),
    }
    
    # Map UI names to internal keys
    agent_key_map = {
        "Buyer Performance": "sales_performance",
        "Publisher Quality": "quality_analysis",
        "Comparative Analysis": "comparative_analysis",
        "Executive Summary": "executive_summary",
    }
    
    focus_areas = [agent_key_map[a] for a in selected_agents]
    orchestrator = Orchestrator(agents)
    results = orchestrator.run_comprehensive_analysis(data, focus_areas)
    summary = orchestrator.get_summary_for_voice(results)
    return results, summary 