import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import json
import traceback

# Import local modules - will handle at app level
try:
    from .nlp_handler import NLPHandler, QueryResult
    from .visualizer import Visualizer
    from .config_manager import ConfigManager
except ImportError:
    # For direct execution
    from nlp_handler import NLPHandler, QueryResult
    from visualizer import Visualizer
    from config_manager import ConfigManager

class QueryEngine:
    """Coordinates query processing, analysis, and visualization."""
    
    def __init__(self, config_path: str = "config"):
        self.nlp_handler = NLPHandler()
        self.visualizer = Visualizer(config_path)
        self.config_manager = ConfigManager(config_path)
        self.query_history = []
        self.current_context = {}
    
    def process_natural_language_query(self, 
                                     query: str, 
                                     df: pd.DataFrame, 
                                     use_context: bool = True) -> Dict[str, Any]:
        """
        Process a natural language query end-to-end.
        
        Args:
            query: Natural language question
            df: DataFrame to analyze
            use_context: Whether to use business context
            
        Returns:
            Dictionary containing results, visualization, and metadata
        """
        
        try:
            # Get business context if available
            context = None
            if use_context:
                context = self.config_manager.get_context_for_query(df)
                self.current_context = context
            
            # Process query with NLP handler
            query_result = self.nlp_handler.process_query(query, df, context)
            
            # Create visualization if data is available
            visualization = None
            if query_result.success and query_result.data is not None:
                viz_type = query_result.visualization_type or 'auto'
                visualization = self.visualizer.create_visualization(
                    query_result.data, 
                    viz_type
                )
            
            # Store in history
            self._add_to_history(query, query_result, context)
            
            # Prepare response
            response = {
                'success': query_result.success,
                'query': query,
                'data': query_result.data,
                'visualization': visualization,
                'explanation': query_result.explanation,
                'code': query_result.code,
                'follow_up_suggestions': query_result.follow_up_suggestions,
                'context_used': context is not None,
                'timestamp': pd.Timestamp.now()
            }
            
            return response
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'timestamp': pd.Timestamp.now()
            }
    
    def _add_to_history(self, query: str, result: QueryResult, context: Dict):
        """Add query and result to history."""
        
        history_entry = {
            'query': query,
            'success': result.success,
            'explanation': result.explanation,
            'data_shape': result.data.shape if result.data is not None else None,
            'visualization_type': result.visualization_type,
            'context_used': context is not None,
            'timestamp': pd.Timestamp.now()
        }
        
        self.query_history.append(history_entry)
        
        # Keep only last 50 queries
        if len(self.query_history) > 50:
            self.query_history = self.query_history[-50:]
    
    def get_query_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Get suggested queries based on data structure and business context."""
        
        suggestions = []
        
        # Get basic data-driven suggestions
        basic_suggestions = self._get_data_driven_suggestions(df)
        suggestions.extend(basic_suggestions)
        
        # Get business context suggestions
        context = self.config_manager.get_context_for_query(df)
        business_mappings = context.get('business_mappings', {})
        
        if business_mappings:
            business_suggestions = self._get_business_driven_suggestions(df, business_mappings)
            suggestions.extend(business_suggestions)
        
        # Get suggested analyses from config
        config_suggestions = self.config_manager.get_suggested_analyses()
        for category, questions in config_suggestions.items():
            suggestions.extend(questions[:2])  # Take first 2 from each category
        
        return list(set(suggestions))  # Remove duplicates
    
    def _get_data_driven_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Generate suggestions based on data structure."""
        
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Numeric column suggestions
        if numeric_cols:
            suggestions.append(f"What is the total {numeric_cols[0]}?")
            suggestions.append(f"Show me the average {numeric_cols[0]}")
            if len(numeric_cols) > 1:
                suggestions.append(f"What's the relationship between {numeric_cols[0]} and {numeric_cols[1]}?")
        
        # Categorical column suggestions
        if categorical_cols and numeric_cols:
            suggestions.append(f"Show me {numeric_cols[0]} by {categorical_cols[0]}")
            suggestions.append(f"Which {categorical_cols[0]} has the highest {numeric_cols[0]}?")
        
        # Time-based suggestions
        if datetime_cols and numeric_cols:
            suggestions.append(f"Show me {numeric_cols[0]} trends over time")
            suggestions.append("What are the monthly patterns?")
        
        return suggestions
    
    def _get_business_driven_suggestions(self, df: pd.DataFrame, mappings: Dict[str, str]) -> List[str]:
        """Generate suggestions based on business context."""
        
        suggestions = []
        
        # Revenue-related suggestions
        revenue_cols = [col for col, term in mappings.items() if term == 'revenue']
        if revenue_cols:
            suggestions.extend([
                "What is our total revenue?",
                "Show me revenue by region",
                "How has revenue changed over time?"
            ])
        
        # Customer-related suggestions
        customer_cols = [col for col, term in mappings.items() if term == 'customer']
        if customer_cols:
            suggestions.extend([
                "How many unique customers do we have?",
                "Show me customer distribution",
                "Who are our top customers?"
            ])
        
        # Product-related suggestions
        product_cols = [col for col, term in mappings.items() if term == 'product']
        if product_cols:
            suggestions.extend([
                "What are our best-selling products?",
                "Show me product performance",
                "Which products generate the most revenue?"
            ])
        
        return suggestions
    
    def create_executive_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create an executive summary of the dataset."""
        
        try:
            summary = {
                'overview': {},
                'key_metrics': {},
                'insights': [],
                'recommendations': []
            }
            
            # Basic overview
            summary['overview'] = {
                'total_records': len(df),
                'columns': len(df.columns),
                'date_range': self._get_date_range(df),
                'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            }
            
            # Key metrics
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols[:5]:  # Top 5 numeric columns
                summary['key_metrics'][col] = {
                    'total': df[col].sum(),
                    'average': df[col].mean(),
                    'max': df[col].max(),
                    'min': df[col].min()
                }
            
            # Generate insights
            insights = self._generate_insights(df)
            summary['insights'] = insights
            
            # Generate recommendations
            recommendations = self._generate_recommendations(df)
            summary['recommendations'] = recommendations
            
            return summary
            
        except Exception as e:
            st.error(f"Error creating executive summary: {str(e)}")
            return {'error': str(e)}
    
    def _get_date_range(self, df: pd.DataFrame) -> Optional[str]:
        """Get date range from the dataset."""
        
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        if len(datetime_cols) > 0:
            col = datetime_cols[0]
            min_date = df[col].min()
            max_date = df[col].max()
            return f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        return None
    
    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate automatic insights from the data."""
        
        insights = []
        
        try:
            # Missing data insight
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct > 10:
                insights.append(f"⚠️ Data has {missing_pct:.1f}% missing values - consider data quality review")
            
            # Numeric trends
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols[:3]:
                if df[col].std() > df[col].mean():
                    insights.append(f"📊 {col} shows high variability (std > mean)")
                
                # Check for outliers
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
                if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                    insights.append(f"⚡ {col} has significant outliers ({len(outliers)} records)")
            
            # Categorical insights
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols[:2]:
                unique_count = df[col].nunique()
                if unique_count < 10:
                    top_category = df[col].value_counts().index[0]
                    top_pct = (df[col].value_counts().iloc[0] / len(df)) * 100
                    insights.append(f"🎯 {top_category} dominates {col} ({top_pct:.1f}% of records)")
            
            # Correlation insights
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                
                for col1, col2, corr in high_corr[:2]:  # Top 2 correlations
                    insights.append(f"🔗 Strong correlation between {col1} and {col2} ({corr:.2f})")
            
        except Exception as e:
            insights.append(f"Error generating insights: {str(e)}")
        
        return insights
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        try:
            # Data quality recommendations
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                recommendations.append(f"📋 Address missing values in: {', '.join(missing_cols[:3])}")
            
            # Analysis recommendations
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            if datetime_cols and numeric_cols:
                recommendations.append("📈 Perform time series analysis to identify trends and patterns")
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                recommendations.append("🎯 Segment analysis by categories for deeper insights")
            
            if len(numeric_cols) >= 2:
                recommendations.append("🔍 Explore correlations between numeric variables")
            
            # Business recommendations based on context
            context = self.current_context
            if context and context.get('business_mappings'):
                mappings = context['business_mappings']
                if any(term == 'revenue' for term in mappings.values()):
                    recommendations.append("💰 Focus on revenue drivers and profitability analysis")
                if any(term == 'customer' for term in mappings.values()):
                    recommendations.append("👥 Analyze customer behavior and segmentation")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def export_analysis_report(self, df: pd.DataFrame, queries: List[str]) -> str:
        """Export a comprehensive analysis report."""
        
        try:
            report = {
                'metadata': {
                    'generated_at': pd.Timestamp.now().isoformat(),
                    'dataset_info': {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'memory_usage': df.memory_usage(deep=True).sum()
                    }
                },
                'executive_summary': self.create_executive_summary(df),
                'query_history': self.query_history,
                'business_context': self.current_context,
                'data_profile': {
                    'column_types': df.dtypes.to_dict(),
                    'missing_values': df.isnull().sum().to_dict(),
                    'summary_statistics': df.describe().to_dict()
                }
            }
            
            # Convert to JSON string
            report_json = json.dumps(report, default=str, indent=2)
            return report_json
            
        except Exception as e:
            st.error(f"Error exporting report: {str(e)}")
            return f"Error: {str(e)}"
    
    def get_conversation_context(self) -> str:
        """Get context from conversation history."""
        
        if not self.query_history:
            return "No previous queries in this session."
        
        recent_queries = self.query_history[-5:]  # Last 5 queries
        context = "Recent conversation:\n"
        
        for entry in recent_queries:
            status = "✅" if entry['success'] else "❌"
            context += f"{status} {entry['query']}\n"
        
        return context
    
    def clear_session(self):
        """Clear session data."""
        
        self.query_history = []
        self.current_context = {}
        self.nlp_handler.clear_conversation_history()
        
        # Clear streamlit session state
        for key in list(st.session_state.keys()):
            if key.startswith('user_'):
                del st.session_state[key] 