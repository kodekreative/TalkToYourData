"""
Marketing Data Analyzer
Core business logic for analyzing performance marketing data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
import os

warnings.filterwarnings('ignore')

class MarketingDataAnalyzer:
    """Core data analysis engine for marketing performance data"""
    
    def __init__(self):
        """Initialize the analyzer with business metrics configuration"""
        # Load business metrics configuration
        config_path = os.path.join(os.path.dirname(__file__), '../../business_metrics.json')
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Fallback configuration if file not found
            self.config = {
                "column_mappings": {},
                "business_metrics": {},
                "synonyms": {}
            }
        
        self.column_mappings = self.config.get('column_mappings', {})
        self.business_metrics = self.config.get('business_metrics', {})
        self.synonyms = self.config.get('synonyms', {})
        
    def _find_column(self, df: pd.DataFrame, target_field: str) -> Optional[str]:
        """Find the actual column name in the dataframe based on mappings"""
        # First check if the exact field exists
        if target_field.upper() in df.columns:
            return target_field.upper()
        if target_field.lower() in df.columns:
            return target_field.lower()
        if target_field in df.columns:
            return target_field
            
        # Check mappings
        possible_names = self.column_mappings.get(target_field, [target_field])
        for name in possible_names:
            for col in df.columns:
                if col.lower() == name.lower():
                    return col
        return None
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names but keep original ones that exist"""
        # Just return the dataframe as-is since the existing system already has standardized names
        return df
    
    def _create_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived columns for analysis"""
        df = df.copy()
        
        # Create IS_SALE column if SALE column exists
        sale_col = self._find_column(df, 'sale')
        if sale_col and 'IS_SALE' not in df.columns:
            df['IS_SALE'] = (df[sale_col].astype(str).str.upper() == 'YES').astype(int)
        
        # Create IS_STRONG_LEAD column based on customer intent
        intent_col = self._find_column(df, 'customer_intent')
        if intent_col and 'IS_STRONG_LEAD' not in df.columns:
            df['IS_STRONG_LEAD'] = df[intent_col].astype(str).str.contains('Level 1|High', case=False, na=False).astype(int)
        
        # Create IS_QUALIFIED column
        qualified_col = self._find_column(df, 'isqualified')
        if qualified_col and 'IS_QUALIFIED' not in df.columns:
            df['IS_QUALIFIED'] = (df[qualified_col] == 1).astype(int)
        
        return df
    
    def get_summary_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get high-level summary metrics"""
        df = self._create_derived_columns(df)
        
        # Find relevant columns
        call_id_col = self._find_column(df, 'call_id') or 'index'
        publisher_col = self._find_column(df, 'publisher')
        revenue_col = self._find_column(df, 'revenue')
        payout_col = self._find_column(df, 'payout')
        
        total_calls = len(df)
        total_sales = df['IS_SALE'].sum() if 'IS_SALE' in df.columns else 0
        total_strong_leads = df['IS_STRONG_LEAD'].sum() if 'IS_STRONG_LEAD' in df.columns else 0
        
        metrics = {
            'total_calls': total_calls,
            'total_sales': total_sales,
            'total_strong_leads': total_strong_leads,
            'conversion_rate': (total_sales / total_calls * 100) if total_calls > 0 else 0,
            'strong_lead_rate': (total_strong_leads / total_calls * 100) if total_calls > 0 else 0
        }
        
        # Add revenue metrics if available
        if revenue_col and revenue_col in df.columns:
            total_revenue = df[revenue_col].sum()
            metrics['total_revenue'] = total_revenue
            metrics['revenue_per_call'] = total_revenue / total_calls if total_calls > 0 else 0
            if total_sales > 0:
                metrics['revenue_per_sale'] = total_revenue / total_sales
        
        # Add cost metrics if available
        if payout_col and payout_col in df.columns:
            total_cost = df[payout_col].sum()
            metrics['total_cost'] = total_cost
            metrics['cost_per_call'] = total_cost / total_calls if total_calls > 0 else 0
            if total_sales > 0:
                metrics['cost_per_sale'] = total_cost / total_sales
        
        # Add publisher count if available
        if publisher_col and publisher_col in df.columns:
            metrics['unique_publishers'] = df[publisher_col].nunique()
        
        return metrics
    
    def calculate_intent_percentages(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate customer intent distribution"""
        intent_col = self._find_column(df, 'customer_intent')
        if not intent_col or intent_col not in df.columns:
            return {'error': 'Customer intent column not found'}
        
        intent_counts = df[intent_col].value_counts()
        total = len(df)
        
        return {
            intent: (count / total * 100) for intent, count in intent_counts.items()
        }
    
    def analyze_publisher_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by publisher"""
        df = self._create_derived_columns(df)
        publisher_col = self._find_column(df, 'publisher')
        
        if not publisher_col or publisher_col not in df.columns:
            return pd.DataFrame({'error': ['Publisher column not found']})
        
        # Group by publisher
        grouped = df.groupby(publisher_col).agg({
            publisher_col: 'count',  # Total calls
            'IS_SALE': 'sum' if 'IS_SALE' in df.columns else lambda x: 0,
            'IS_STRONG_LEAD': 'sum' if 'IS_STRONG_LEAD' in df.columns else lambda x: 0
        }).rename(columns={publisher_col: 'total_calls'})
        
        # Calculate rates
        grouped['conversion_rate'] = (grouped['IS_SALE'] / grouped['total_calls'] * 100).round(2)
        grouped['strong_lead_rate'] = (grouped['IS_STRONG_LEAD'] / grouped['total_calls'] * 100).round(2)
        
        # Add revenue and cost metrics if available
        revenue_col = self._find_column(df, 'revenue')
        payout_col = self._find_column(df, 'payout')
        
        if revenue_col and revenue_col in df.columns:
            revenue_stats = df.groupby(publisher_col)[revenue_col].agg(['sum', 'mean'])
            grouped['total_revenue'] = revenue_stats['sum']
            grouped['revenue_per_call'] = revenue_stats['mean'].round(2)
        
        if payout_col and payout_col in df.columns:
            cost_stats = df.groupby(publisher_col)[payout_col].agg(['sum', 'mean'])
            grouped['total_cost'] = cost_stats['sum']
            grouped['cost_per_call'] = cost_stats['mean'].round(2)
        
        return grouped.reset_index()
    
    def analyze_target_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by target audience"""
        df = self._create_derived_columns(df)
        target_col = self._find_column(df, 'target')
        
        if not target_col or target_col not in df.columns:
            return pd.DataFrame({'error': ['Target column not found']})
        
        # Group by target
        grouped = df.groupby(target_col).agg({
            target_col: 'count',  # Total calls
            'IS_SALE': 'sum' if 'IS_SALE' in df.columns else lambda x: 0,
            'IS_STRONG_LEAD': 'sum' if 'IS_STRONG_LEAD' in df.columns else lambda x: 0
        }).rename(columns={target_col: 'total_calls'})
        
        # Calculate rates
        grouped['conversion_rate'] = (grouped['IS_SALE'] / grouped['total_calls'] * 100).round(2)
        grouped['strong_lead_rate'] = (grouped['IS_STRONG_LEAD'] / grouped['total_calls'] * 100).round(2)
        
        return grouped.reset_index()
    
    def analyze_cost_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cost efficiency metrics"""
        df = self._create_derived_columns(df)
        
        revenue_col = self._find_column(df, 'revenue')
        payout_col = self._find_column(df, 'payout')
        
        if not revenue_col or not payout_col:
            return {'error': 'Revenue or payout columns not found'}
        
        if revenue_col not in df.columns or payout_col not in df.columns:
            return {'error': 'Required columns not available in data'}
        
        total_revenue = df[revenue_col].sum()
        total_cost = df[payout_col].sum()
        total_calls = len(df)
        total_sales = df['IS_SALE'].sum() if 'IS_SALE' in df.columns else 0
        
        return {
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'net_profit': total_revenue - total_cost,
            'roi_percentage': ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0,
            'cost_per_call': total_cost / total_calls if total_calls > 0 else 0,
            'cost_per_sale': total_cost / total_sales if total_sales > 0 else 0,
            'revenue_per_call': total_revenue / total_calls if total_calls > 0 else 0,
            'revenue_per_sale': total_revenue / total_sales if total_sales > 0 else 0,
            'profit_margin': ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0
        }
    
    def analyze_call_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze call quality metrics"""
        quality_metrics = {}
        
        # Agent reach rate
        reached_agent_col = self._find_column(df, 'reached_agent')
        if reached_agent_col and reached_agent_col in df.columns:
            reached_count = (df[reached_agent_col].astype(str).str.upper() == 'YES').sum()
            quality_metrics['agent_reach_rate'] = (reached_count / len(df) * 100) if len(df) > 0 else 0
        
        # Billable rate
        billable_col = self._find_column(df, 'billable')
        if billable_col and billable_col in df.columns:
            billable_count = (df[billable_col].astype(str).str.upper() == 'YES').sum()
            quality_metrics['billable_rate'] = (billable_count / len(df) * 100) if len(df) > 0 else 0
        
        # Ad satisfaction rate (not misled)
        ad_misled_col = self._find_column(df, 'ad_misled')
        if ad_misled_col and ad_misled_col in df.columns:
            not_misled_count = (df[ad_misled_col].astype(str).str.upper() == 'NO').sum()
            quality_metrics['ad_satisfaction_rate'] = (not_misled_count / len(df) * 100) if len(df) > 0 else 0
        
        # Average call duration
        duration_col = self._find_column(df, 'duration')
        if duration_col and duration_col in df.columns:
            quality_metrics['avg_call_duration'] = df[duration_col].mean()
        
        # IVR abandonment rate
        ivr_col = self._find_column(df, 'ivr')
        if ivr_col and reached_agent_col and both_cols_exist(df, [ivr_col, reached_agent_col]):
            ivr_calls = df[df[ivr_col].astype(str).str.upper() == 'YES']
            if len(ivr_calls) > 0:
                abandoned = ivr_calls[ivr_calls[reached_agent_col].astype(str).str.upper() == 'NO']
                quality_metrics['ivr_abandonment_rate'] = (len(abandoned) / len(ivr_calls) * 100)
        
        return quality_metrics
    
    def analyze_conversion_funnel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the conversion funnel through different stages"""
        funnel_data = {}
        
        total_calls = len(df)
        funnel_data['total_calls'] = total_calls
        
        # Stage analysis
        stages = [
            'stage_1_introduction',
            'stage_2_eligibility', 
            'stage_3_needs_analysis',
            'stage_4_plan_detail',
            'stage_5_enrollment'
        ]
        
        for stage in stages:
            stage_col = self._find_column(df, stage)
            if stage_col and stage_col in df.columns:
                stage_count = (df[stage_col].astype(str).str.upper() == 'YES').sum()
                funnel_data[f'{stage}_count'] = stage_count
                funnel_data[f'{stage}_rate'] = (stage_count / total_calls * 100) if total_calls > 0 else 0
        
        # Quote and sale analysis
        quote_col = self._find_column(df, 'quote')
        if quote_col and quote_col in df.columns:
            quote_count = (df[quote_col].astype(str).str.upper() == 'YES').sum()
            funnel_data['quotes_given'] = quote_count
            funnel_data['quote_rate'] = (quote_count / total_calls * 100) if total_calls > 0 else 0
        
        # Final conversion
        df = self._create_derived_columns(df)
        if 'IS_SALE' in df.columns:
            sales_count = df['IS_SALE'].sum()
            funnel_data['final_sales'] = sales_count
            funnel_data['conversion_rate'] = (sales_count / total_calls * 100) if total_calls > 0 else 0
            
            # Quote to sale conversion if both exist
            if 'quotes_given' in funnel_data and funnel_data['quotes_given'] > 0:
                funnel_data['quote_to_sale_rate'] = (sales_count / funnel_data['quotes_given'] * 100)
        
        return funnel_data
    
    def analyze_time_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time-based patterns in the data"""
        date_col = self._find_column(df, 'date') or self._find_column(df, 'call_date')
        
        if not date_col or date_col not in df.columns:
            return {'error': 'Date column not found'}
        
        # Convert to datetime
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            return {'error': 'Could not parse date column'}
        
        df = self._create_derived_columns(df)
        
        patterns = {}
        
        # Daily patterns
        df['day_of_week'] = df[date_col].dt.day_name()
        daily_calls = df.groupby('day_of_week').size()
        patterns['daily_call_volume'] = daily_calls.to_dict()
        
        if 'IS_SALE' in df.columns:
            daily_conversion = df.groupby('day_of_week')['IS_SALE'].mean() * 100
            patterns['daily_conversion_rate'] = daily_conversion.to_dict()
        
        # Hourly patterns (if time info available)
        if df[date_col].dt.hour.notna().any():
            df['hour'] = df[date_col].dt.hour
            hourly_calls = df.groupby('hour').size()
            patterns['hourly_call_volume'] = hourly_calls.to_dict()
            
            if 'IS_SALE' in df.columns:
                hourly_conversion = df.groupby('hour')['IS_SALE'].mean() * 100
                patterns['hourly_conversion_rate'] = hourly_conversion.to_dict()
        
        return patterns
    
    def analyze_query(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze data based on natural language query"""
        query_lower = query.lower()
        
        # Determine what type of analysis to perform based on query
        if any(word in query_lower for word in ['intent', 'interest', 'level']):
            return {
                'analysis_type': 'Intent Analysis',
                'data': self.calculate_intent_percentages(df),
                'summary': 'Customer intent level distribution'
            }
        
        elif any(word in query_lower for word in ['publisher', 'affiliate', 'source']):
            result_df = self.analyze_publisher_performance(df)
            return {
                'analysis_type': 'Publisher Performance',
                'data': result_df.to_dict('records') if not result_df.empty else [],
                'summary': f'Performance analysis across {len(result_df)} publishers'
            }
        
        elif any(word in query_lower for word in ['target', 'audience', 'demographic']):
            result_df = self.analyze_target_performance(df)
            return {
                'analysis_type': 'Target Performance', 
                'data': result_df.to_dict('records') if not result_df.empty else [],
                'summary': f'Performance analysis across {len(result_df)} target segments'
            }
        
        elif any(word in query_lower for word in ['cost', 'efficiency', 'roi', 'profit', 'revenue']):
            return {
                'analysis_type': 'Cost Efficiency',
                'data': self.analyze_cost_efficiency(df),
                'summary': 'Financial performance and cost efficiency metrics'
            }
        
        elif any(word in query_lower for word in ['quality', 'agent', 'duration', 'billable']):
            return {
                'analysis_type': 'Call Quality',
                'data': self.analyze_call_quality(df),
                'summary': 'Call quality and operational metrics'
            }
        
        elif any(word in query_lower for word in ['funnel', 'conversion', 'stage', 'enrollment']):
            return {
                'analysis_type': 'Conversion Funnel',
                'data': self.analyze_conversion_funnel(df),
                'summary': 'Conversion funnel analysis through sales stages'
            }
        
        elif any(word in query_lower for word in ['time', 'daily', 'hourly', 'pattern', 'trend']):
            return {
                'analysis_type': 'Time Patterns',
                'data': self.analyze_time_patterns(df),
                'summary': 'Time-based patterns and trends analysis'
            }
        
        elif any(word in query_lower for word in ['summary', 'overview', 'total']):
            return {
                'analysis_type': 'Summary Metrics',
                'data': self.get_summary_metrics(df),
                'summary': 'High-level summary of key performance metrics'
            }
        
        else:
            # Default to summary metrics
            return {
                'analysis_type': 'Summary Metrics',
                'data': self.get_summary_metrics(df),
                'summary': 'High-level summary of key performance metrics'
            }

def both_cols_exist(df: pd.DataFrame, cols: List[str]) -> bool:
    """Helper function to check if both columns exist"""
    return all(col in df.columns for col in cols) 