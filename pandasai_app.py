#!/usr/bin/env python3
"""
PandasAI-style TalkToYourData using OpenAI GPT-4o
Much simpler and more reliable approach
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import warnings
import re
from difflib import get_close_matches
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

def init_openai():
    """Initialize OpenAI client with GPT-4o."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != "your_openai_api_key_here":
        return OpenAI(api_key=api_key)
    return None

def get_dataframe_info(df):
    """Get comprehensive information about the dataframe."""
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "sample_data": df.head(3).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    return info

def query_with_gpt4o(query, df, client):
    """Use GPT-4o to analyze query and generate pandas code."""
    if not client:
        return None, "OpenAI client not available"
    
    try:
        df_info = get_dataframe_info(df)
        business_config = load_business_metrics()
        column_matches = find_column_matches(df.columns, business_config)
        detected_metric = detect_business_metric(query, business_config)
        
        # Check for grouping keywords
        group_by_col = None
        query_lower = query.lower()
        for col in df.columns:
            if f"by {col.lower()}" in query_lower or f"per {col.lower()}" in query_lower:
                group_by_col = col
                break
        
        # If we detected a business metric, try to generate code directly
        if detected_metric and detected_metric in business_config['business_metrics']:
            business_code = generate_business_metric_code(detected_metric, business_config, column_matches, group_by_col)
            if business_code and not business_code.startswith("# Missing columns"):
                return {
                    "code": business_code.strip(),
                    "explanation": f"Calculate {detected_metric.replace('_', ' ')} using business logic: {business_config['business_metrics'][detected_metric]['description']}",
                    "visualization_type": "bar"
                }, None
        
        # Build business context for GPT-4o
        business_context = ""
        if business_config['business_metrics']:
            business_context = "\n\nBUSINESS METRICS AVAILABLE:\n"
            for metric_name, metric_info in business_config['business_metrics'].items():
                business_context += f"- {metric_name.replace('_', ' ').title()}: {metric_info['description']}\n"
                business_context += f"  Formula: {metric_info['example']}\n"
            
            business_context += "\nCOLUMN MAPPINGS DETECTED:\n"
            for business_col, actual_col in column_matches.items():
                business_context += f"- {business_col} → {actual_col}\n"
        
        # Create a detailed prompt for GPT-4o
        prompt = f"""
You are a data analyst with business intelligence capabilities. Given a user query and dataframe information, generate Python pandas code to answer the query.

DATAFRAME INFO:
- Shape: {df_info['shape']} (rows, columns)
- Columns: {df_info['columns']}
- Data types: {df_info['dtypes']}
- Sample data (first 3 rows): {df_info['sample_data']}
- Numeric columns: {df_info['numeric_columns']}
- Categorical columns: {df_info['categorical_columns']}

{business_context}

USER QUERY: "{query}"

Generate Python pandas code that:
1. Uses the variable name 'df' for the dataframe
2. Handles case-insensitive column matching
3. Returns a result that directly answers the query
4. For groupby queries, returns a proper dataframe or series
5. For calculations, returns the actual calculated value
6. For business metrics, uses proper business logic
7. For date/time analysis, automatically detects date columns and enables time-based grouping

TIME-BASED ANALYSIS EXAMPLES:

For "sales by month":
{{
    "code": "date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]; date_col = date_cols[0] if date_cols else None; df[date_col] = pd.to_datetime(df[date_col], errors='coerce'); result = df.groupby(df[date_col].dt.to_period('M'))['SALE'].value_counts().unstack(fill_value=0); result.index = result.index.astype(str); result",
    "explanation": "Group sales data by month using date column",
    "visualization_type": "bar"
}}

For "conversion rate by day of week":
{{
    "code": "date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]; date_col = date_cols[0] if date_cols else None; df[date_col] = pd.to_datetime(df[date_col], errors='coerce'); df['day_of_week'] = df[date_col].dt.day_name(); result = df.groupby('day_of_week').apply(lambda x: (x['SALE'] == 'Yes').sum() / (x['SALE'] == 'No').sum() if (x['SALE'] == 'No').sum() > 0 else 0).reset_index(); result.columns = ['Day_of_Week', 'Conversion_Rate']; result",
    "explanation": "Calculate conversion rate by day of the week",
    "visualization_type": "bar"
}}

For "hourly call volume":
{{
    "code": "date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]; date_col = date_cols[0] if date_cols else None; df[date_col] = pd.to_datetime(df[date_col], errors='coerce'); result = df.groupby(df[date_col].dt.hour).size().reset_index(); result.columns = ['Hour', 'Call_Count']; result",
    "explanation": "Show call volume by hour of day",
    "visualization_type": "bar"
}}

BUSINESS METRIC EXAMPLES:

For "cost per sale by publisher":
{{
    "code": "revenue_col = 'REVENUE'; sale_col = 'SALE'; result = df.groupby('PUBLISHER').apply(lambda x: x[x[sale_col] == 'Yes'][revenue_col].sum() / (x[sale_col] == 'Yes').sum()).reset_index(); result.columns = ['PUBLISHER', 'Cost_Per_Sale']; result",
    "explanation": "Calculate cost per sale by publisher: total revenue per publisher divided by successful sales per publisher",
    "visualization_type": "bar"
}}

For "conversion rate by publisher":
{{
    "code": "sale_col = 'SALE'; result = df.groupby('PUBLISHER').apply(lambda x: (x[sale_col] == 'Yes').sum() / (x[sale_col] == 'No').sum() if (x[sale_col] == 'No').sum() > 0 else 0).reset_index(); result.columns = ['PUBLISHER', 'Conversion_Rate']; result",
    "explanation": "Calculate conversion rate by publisher: ratio of successful sales to declined sales (0=no successes, 1=equal successes and declines)",
    "visualization_type": "bar"
}}

PERCENTAGE CALCULATION RULES:
- For conversion rates, calculate Yes/No (successes as percentage of declines) within each group
- Focus on success rates relative to failures, not total success rates
- Use .round(1) to show one decimal place for percentages
- For cost metrics, calculate total cost/revenue per group

Return ONLY a JSON object with this EXACT format:
{{
    "code": "pandas code here",
    "explanation": "brief explanation of what the code does",
    "visualization_type": "bar"
}}

IMPORTANT: 
- For business metrics, focus on actionable insights
- Use proper business formulas (revenue/sales, not just counts)
- Handle missing data gracefully
- Make sure the JSON is valid 
- The code field contains executable pandas code
- For cost per sale, divide revenue by successful sales count
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        # Try to parse JSON response
        try:
            result = json.loads(result_text)
            # Validate that we have the required fields
            if not all(key in result for key in ['code', 'explanation', 'visualization_type']):
                return None, "Invalid response format from GPT-4o"
            return result, None
        except json.JSONDecodeError as e:
            return None, f"Failed to parse GPT-4o JSON response: {str(e)}"
                
    except Exception as e:
        return None, f"Error calling GPT-4o: {str(e)}"

def execute_code_safely(code, df):
    """Execute pandas code safely and return result."""
    try:
        # Create a safe execution environment with necessary built-ins
        import numpy as np
        
        # Include necessary built-in functions that pandas code commonly uses
        safe_builtins = {
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'any': any,
            'all': all
        }
        
        local_vars = {"df": df, "pd": pd, "np": np}
        
        # Clean the code - remove any extra whitespace
        code = code.strip()
        
        # Execute the entire code block in one go
        exec(code, {"__builtins__": safe_builtins, "pd": pd, "np": np}, local_vars)
        
        # Try to get the result
        if 'result' in local_vars:
            result = local_vars['result']
        else:
            # If no result variable, try to find the last assigned variable
            lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
            if lines:
                last_line = lines[-1]
                if '=' in last_line and not last_line.startswith(('if', 'for', 'while')):
                    var_name = last_line.split('=')[0].strip()
                    result = local_vars.get(var_name, "Code executed successfully")
                else:
                    # Last line might be an expression
                    try:
                        result = eval(last_line, {"__builtins__": safe_builtins, "pd": pd, "np": np}, local_vars)
                    except:
                        result = "Code executed successfully"
            else:
                result = "Code executed successfully"
        
        # Check if result is empty or None
        if result is None:
            return "No data found or calculation returned empty result", None
        
        # Handle empty DataFrames or Series
        if isinstance(result, pd.DataFrame) and len(result) == 0:
            return "No data found matching the criteria", None
        elif isinstance(result, pd.Series) and len(result) == 0:
            return "No data found matching the criteria", None
        
        return result, None
        
    except Exception as e:
        return None, f"Error executing code: {str(e)}"

def format_percentage_results(result, query):
    """Format results as percentages when appropriate."""
    if isinstance(result, pd.DataFrame):
        # If values already look like percentages (0-100 range), add % symbol
        numeric_cols = result.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        
        if len(numeric_cols) > 0:
            # Check if values are in percentage range (0-100) 
            for col in numeric_cols:
                if result[col].max() <= 100 and result[col].min() >= 0:
                    # If query mentions percentage, add % symbol
                    if any(word in query.lower() for word in ['percentage', 'percent', '%']):
                        result[col] = result[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%")
                        
    elif isinstance(result, pd.Series):
        # Handle series percentage formatting
        if result.max() <= 100 and result.min() >= 0:
            if any(word in query.lower() for word in ['percentage', 'percent', '%']):
                result = result.apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "0.0%")
    
    return result

def create_visualization(result, viz_type, query):
    """Create appropriate visualization based on result and type."""
    if viz_type == "none" or result is None:
        return None
    
    try:
        # Work with the original numeric data for plotting
        plot_data = result.copy() if hasattr(result, 'copy') else result
        
        if isinstance(result, (int, float)):
            # Single value result
            fig = go.Figure(go.Indicator(
                mode = "number",
                value = result,
                title = {"text": query},
                number = {"suffix": "%" if "percent" in query.lower() else ""}
            ))
            return fig
        
        elif isinstance(plot_data, pd.Series):
            if viz_type == "pie":
                fig = px.pie(values=plot_data.values, names=plot_data.index, title=query)
            else:
                fig = px.bar(x=plot_data.index, y=plot_data.values, title=query)
                if "percent" in query.lower():
                    fig.update_yaxes(title="Percentage (%)")
            return fig
        
        elif isinstance(plot_data, pd.DataFrame):
            # Handle conversion rate DataFrames (with columns like 'Yes_Rate_%', 'No_Rate_%', etc.)
            if any('Rate_%' in col or 'Conversion' in col for col in plot_data.columns):
                # Focus on rate/percentage columns for visualization
                rate_cols = [col for col in plot_data.columns if 'Rate_%' in col or 'Conversion' in col]
                if rate_cols:
                    # Create a bar chart showing the rates
                    if len(rate_cols) == 1:
                        # Single rate column (e.g., conversion rate)
                        fig = px.bar(x=plot_data.index, y=plot_data[rate_cols[0]], title=query)
                        fig.update_yaxes(title="Percentage (%)")
                        return fig
                    else:
                        # Multiple rate columns (e.g., Yes_Rate_%, No_Rate_%)
                        fig = px.bar(plot_data, x=plot_data.index.name or 'Publisher', 
                                   y=rate_cols, title=query, barmode='group')
                        fig.update_yaxes(title="Percentage (%)")
                        return fig
            
            # Standard DataFrame handling
            elif len(plot_data.columns) == 2 and viz_type == "bar":
                fig = px.bar(plot_data, x=plot_data.columns[0], y=plot_data.columns[1], title=query)
                if "percent" in query.lower():
                    fig.update_yaxes(title="Percentage (%)")
                return fig
            elif viz_type == "pie" and len(plot_data.columns) == 2:
                fig = px.pie(plot_data, values=plot_data.columns[1], names=plot_data.columns[0], title=query)
                return fig
            elif viz_type == "bar" and len(plot_data.columns) > 2:
                # For multi-column bar charts
                fig = px.bar(plot_data.reset_index(), x=plot_data.index.name or 'index', 
                           y=plot_data.columns.tolist(), title=query, barmode='group')
                if "percent" in query.lower():
                    fig.update_yaxes(title="Percentage (%)")
                return fig
        
        return None
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def load_business_metrics():
    """Load business metrics configuration from JSON file."""
    try:
        with open('business_metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return empty structure if file doesn't exist
        return {
            "business_metrics": {},
            "column_mappings": {},
            "synonyms": {}
        }

def find_column_matches(df_columns, business_config):
    """Find which business columns match actual dataframe columns."""
    column_mappings = business_config.get('column_mappings', {})
    matches = {}
    
    df_columns_lower = [col.lower() for col in df_columns]
    
    for business_col, synonyms in column_mappings.items():
        for df_col in df_columns:
            df_col_lower = df_col.lower()
            if any(synonym.lower() in df_col_lower for synonym in synonyms):
                matches[business_col] = df_col
                break
    
    return matches

def detect_business_metric(query, business_config):
    """Detect if the query is asking for a business metric."""
    query_lower = query.lower()
    
    # Check direct metric names
    for metric_name in business_config['business_metrics'].keys():
        if metric_name.replace('_', ' ') in query_lower:
            return metric_name
    
    # Check synonyms
    for term, synonyms in business_config['synonyms'].items():
        if any(synonym.lower() in query_lower for synonym in synonyms):
            return term.replace(' ', '_')
    
    return None

def generate_business_metric_code(metric_name, business_config, column_matches, group_by_col=None):
    """Generate pandas code for a business metric."""
    if metric_name not in business_config['business_metrics']:
        return None
    
    metric = business_config['business_metrics'][metric_name]
    required_cols = metric['required_columns']
    
    # Check if we have all required columns
    missing_cols = [col for col in required_cols if col not in column_matches]
    if missing_cols:
        return f"# Missing columns for {metric_name}: {missing_cols}"
    
    # Get actual column names - ensure they exist in column_matches
    sale_col = column_matches.get('sale', 'SALE')
    revenue_col = column_matches.get('revenue', 'REVENUE')
    payout_col = column_matches.get('payout', 'PAYOUT')
    
    # Debug: Add column information to the generated code
    debug_info = f"# Debug: sale_col='{sale_col}', revenue_col='{revenue_col}', payout_col='{payout_col}'"
    debug_info += f"\\n# Column matches: {column_matches}"
    
    # Convert to actual pandas code using actual column names
    if metric_name in ["sales_conversion_rate", "conversion_rate"]:
        if group_by_col:
            code = f"""{debug_info}
result = df.groupby('{group_by_col}').agg({{
    '{sale_col}': [
        lambda x: (x == 'Yes').sum(),
        lambda x: (x == 'No').sum(),
        lambda x: (x == 'Yes').sum() / (x == 'No').sum() if (x == 'No').sum() > 0 else 0
    ]
}}).round(3)
result.columns = ['Yes_Count', 'No_Count', 'Conversion_Rate']
result = result.reset_index()
result"""
        else:
            code = f"""{debug_info}
yes_count = (df['{sale_col}'] == 'Yes').sum()
no_count = (df['{sale_col}'] == 'No').sum()
conversion_rate = (yes_count / no_count) if no_count > 0 else 0
result = pd.DataFrame({{
    'Yes_Count': [yes_count],
    'No_Count': [no_count], 
    'Conversion_Rate': [round(conversion_rate, 3)]
}})
result"""
    
    elif metric_name == "cost_per_sale":
        if group_by_col:
            code = f"""{debug_info}
# Convert payout to numeric to handle text values
df['{payout_col}'] = pd.to_numeric(df['{payout_col}'], errors='coerce').fillna(0)
result = df.groupby('{group_by_col}').apply(lambda x: x['{payout_col}'].sum() / (x['{sale_col}'] == 'Yes').sum() if (x['{sale_col}'] == 'Yes').sum() > 0 else 0).reset_index()
result.columns = ['{group_by_col}', 'Cost_Per_Sale']
result"""
        else:
            code = f"""{debug_info}
# Convert payout to numeric to handle text values
df['{payout_col}'] = pd.to_numeric(df['{payout_col}'], errors='coerce').fillna(0)
total_payout = df['{payout_col}'].sum()
total_sales = (df['{sale_col}'] == 'Yes').sum()
result = total_payout / total_sales if total_sales > 0 else 0
result"""
    
    else:
        # Generic code generation for other metrics
        code = f"""{debug_info}
# Generic metric calculation for {metric_name}
# {metric['description']}
result = 'Metric calculation not implemented yet'
result"""
    
    return code

def generate_intelligent_summary(result, query, client):
    """Generate a comprehensive and well-formatted summary with quartile analysis and specific outlier details."""
    if not client or result is None:
        return None
    
    try:
        # Prepare comprehensive statistical analysis
        if isinstance(result, pd.DataFrame):
            if len(result) == 0:
                return {"error": "No data available for analysis."}
            
            # Get all numeric columns for analysis
            numeric_cols = result.select_dtypes(include=['number']).columns
            
            # Generate comprehensive statistics for each numeric column
            detailed_analysis = {}
            for col in numeric_cols:
                if col in result.columns:
                    values = result[col].dropna()
                    if len(values) > 0:
                        # Calculate comprehensive statistics
                        mean_val = values.mean()
                        median_val = values.median()
                        std_val = values.std()
                        q1 = values.quantile(0.25)
                        q3 = values.quantile(0.75)
                        iqr = q3 - q1
                        
                        # Identify specific outliers with their values
                        outliers_high = values[values > (q3 + 1.5 * iqr)]
                        outliers_low = values[values < (q1 - 1.5 * iqr)]
                        
                        # Performance tiers
                        top_quartile = values[values >= q3]
                        bottom_quartile = values[values <= q1]
                        above_median = values[values > median_val]
                        below_median = values[values <= median_val]
                        
                        # Get specific performer details (with row indices if DataFrame)
                        top_performers = []
                        bottom_performers = []
                        
                        if hasattr(result, 'index'):
                            # Get top 5 performers with their row information
                            top_indices = values.nlargest(5).index
                            for idx in top_indices:
                                performer_data = {"value": round(values[idx], 4), "index": idx}
                                if len(result.columns) > 1:
                                    # Add identifying information from other columns
                                    other_cols = [c for c in result.columns if c != col][:2]  # Take first 2 other columns
                                    for other_col in other_cols:
                                        performer_data[other_col] = result.loc[idx, other_col]
                                top_performers.append(performer_data)
                            
                            # Get bottom 5 performers
                            bottom_indices = values.nsmallest(5).index
                            for idx in bottom_indices:
                                performer_data = {"value": round(values[idx], 4), "index": idx}
                                if len(result.columns) > 1:
                                    other_cols = [c for c in result.columns if c != col][:2]
                                    for other_col in other_cols:
                                        performer_data[other_col] = result.loc[idx, other_col]
                                bottom_performers.append(performer_data)
                        
                        detailed_analysis[col] = {
                            "basic_stats": {
                                "count": len(values),
                                "mean": round(mean_val, 4),
                                "median": round(median_val, 4),
                                "std": round(std_val, 4),
                                "min": round(values.min(), 4),
                                "max": round(values.max(), 4)
                            },
                            "quartile_analysis": {
                                "q1": round(q1, 4),
                                "q3": round(q3, 4),
                                "iqr": round(iqr, 4),
                                "top_quartile_count": len(top_quartile),
                                "bottom_quartile_count": len(bottom_quartile),
                                "top_quartile_values": top_quartile.round(4).tolist()[:10],  # Show top 10
                                "bottom_quartile_values": bottom_quartile.round(4).tolist()[:10]
                            },
                            "outlier_analysis": {
                                "outliers_high_count": len(outliers_high),
                                "outliers_low_count": len(outliers_low),
                                "outliers_high_values": outliers_high.round(4).tolist(),
                                "outliers_low_values": outliers_low.round(4).tolist(),
                                "outlier_threshold_high": round(q3 + 1.5 * iqr, 4),
                                "outlier_threshold_low": round(q1 - 1.5 * iqr, 4)
                            },
                            "performance_distribution": {
                                "above_median_count": len(above_median),
                                "below_median_count": len(below_median),
                                "above_median_pct": round(len(above_median) / len(values) * 100, 1),
                                "below_median_pct": round(len(below_median) / len(values) * 100, 1)
                            },
                            "top_performers": top_performers,
                            "bottom_performers": bottom_performers
                        }
            
            summary_data = {
                "query": query,
                "data_shape": result.shape,
                "columns": result.columns.tolist(),
                "analysis": detailed_analysis
            }
            
        elif isinstance(result, pd.Series):
            values = result.dropna()
            if len(values) > 0:
                mean_val = values.mean()
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                
                outliers_high = values[values > (q3 + 1.5 * iqr)]
                outliers_low = values[values < (q1 - 1.5 * iqr)]
                
                summary_data = {
                    "query": query,
                    "type": "series",
                    "basic_stats": {
                        "count": len(values),
                        "mean": round(mean_val, 4),
                        "median": round(values.median(), 4),
                        "min": round(values.min(), 4),
                        "max": round(values.max(), 4)
                    },
                    "quartile_analysis": {
                        "q1": round(q1, 4),
                        "q3": round(q3, 4),
                        "iqr": round(iqr, 4)
                    },
                    "outliers": {
                        "high_values": outliers_high.round(4).tolist(),
                        "low_values": outliers_low.round(4).tolist()
                    },
                    "top_5": values.nlargest(5).round(4).tolist()
                }
            else:
                return {"error": "No valid data for analysis."}
        else:
            summary_data = {
                "query": query,
                "type": "scalar", 
                "value": result
            }
        
        # Enhanced prompt for structured markdown analysis
        prompt = f"""
You are a senior business intelligence analyst. Analyze this data and provide comprehensive insights in properly formatted markdown.

**ANALYSIS REQUEST:**
Query: "{query}"
Data: {summary_data}

**CRITICAL REQUIREMENT - COMPREHENSIVE EXECUTIVE SUMMARY:**
The Executive Summary MUST be extremely detailed and comprehensive, covering:
- Complete performance overview with specific metrics and values
- Detailed statistical insights with quartile breakdowns
- Comprehensive trend analysis and performance patterns
- Specific business implications and strategic insights
- Detailed comparative analysis between top and bottom performers
- Risk factors and opportunity identification
- Performance distribution analysis
- Market positioning insights based on the data

**REQUIRED OUTPUT FORMAT:**
Return a JSON object with these exact fields, each containing well-formatted markdown:

{{
  "executive_summary": "COMPREHENSIVE 8-10 paragraph detailed executive summary with complete analysis",
  "quartile_performance": "markdown content here", 
  "outlier_analysis": "markdown content here",
  "top_performers": "markdown content here",
  "performance_insights": "markdown content here",
  "recommendations": "markdown content here"
}}

**FORMATTING REQUIREMENTS:**
- Use proper markdown headers (##, ###)
- Use bullet points for lists
- Use **bold** for emphasis
- Use tables where appropriate
- Include specific values and percentages
- Executive summary should be 8-10 comprehensive paragraphs minimum
- Each section should be 3-5 well-formatted paragraphs

**EXECUTIVE SUMMARY REQUIREMENTS (MOST IMPORTANT):**
Write a comprehensive 8-10 paragraph executive summary that includes:

1. **Opening Performance Assessment**: Overall performance metrics, averages, ranges, and key statistical indicators
2. **Quartile Distribution Analysis**: Detailed breakdown of performance across quartiles with specific thresholds and counts
3. **Top Performer Analysis**: Comprehensive analysis of highest performing entities with specific values and what drives their success
4. **Bottom Performer Analysis**: Detailed examination of lowest performing entities and potential improvement areas
5. **Performance Variance Analysis**: Analysis of performance spread, volatility, and consistency patterns
6. **Outlier Impact Assessment**: How outliers affect overall performance and what they represent
7. **Comparative Performance Insights**: How different segments/groups compare against each other
8. **Strategic Business Implications**: What these performance patterns mean for business strategy and operations
9. **Risk and Opportunity Assessment**: Key risks identified and opportunities for improvement
10. **Performance Trends and Patterns**: Any notable trends, patterns, or anomalies in the data

**ANALYSIS REQUIREMENTS:**

1. **Quartile Performance**: Focus on Q1 (25th percentile) and Q3 (75th percentile) analysis, not just averages
2. **Outlier Analysis**: List specific outlier values and what they represent
3. **Top Performers**: Specific top/bottom performers with their actual values
4. **Performance Insights**: Distribution analysis and performance gaps
5. **Recommendations**: Actionable steps based on quartile and outlier analysis

**BUSINESS CONTEXT:**
- Higher conversion rates, revenue, ROAS = BETTER
- Lower cost per sale/lead = BETTER
- Focus on quartile-based performance (top 25%, bottom 25%)
- Include specific values, not just percentages
- Identify performance gaps between quartiles

Return ONLY the JSON object with properly formatted markdown in each field. Make the executive summary extremely comprehensive and detailed.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4000  # Significantly increased from 1000 to 4000 for comprehensive analysis
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Clean and parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        try:
            parsed_response = json.loads(response_text)
            parsed_response["raw_analysis"] = summary_data
            return parsed_response
        except json.JSONDecodeError:
            # Fallback with basic structure
            return {
                "executive_summary": f"## Executive Summary\n\nAnalysis completed for: **{query}**\n\n" + response_text[:300] + "...",
                "quartile_performance": "## Quartile Analysis\n\nDetailed quartile analysis available in raw data.",
                "raw_analysis": summary_data
            }
        
    except Exception as e:
        return {"error": f"Error generating comprehensive summary: {str(e)}"}

def main():
    st.set_page_config(
        page_title="Talk To Your Data - GPT-4o Powered",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Talk To Your Data")
    st.subheader("Powered by OpenAI GPT-4o")
    
    # Initialize OpenAI client
    client = init_openai()
    if not client:
        st.error("⚠️ OpenAI API key not found. Please add your API key to the .env file.")
        st.code("OPENAI_API_KEY=your_openai_api_key_here")
        return
    
    st.success("✅ Connected to OpenAI GPT-4o")
    
    # File upload
    st.sidebar.header("📁 Upload Your Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload your Excel file to start chatting with your data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_excel(uploaded_file)
            st.sidebar.success(f"✅ File loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head(10))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Columns:**")
                    for col in df.columns:
                        st.write(f"- {col} ({df[col].dtype})")
                
                with col2:
                    st.write("**Summary:**")
                    st.write(f"- Total rows: {df.shape[0]:,}")
                    st.write(f"- Total columns: {df.shape[1]}")
                    st.write(f"- Numeric columns: {len(df.select_dtypes(include=['number']).columns)}")
                    st.write(f"- Text columns: {len(df.select_dtypes(include=['object']).columns)}")
            
            # Initialize session state for chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Chat interface
            st.header("💬 Chat with Your Data")
            
            # Check for auto-generated query from business metric buttons
            initial_query = ""
            if 'auto_query' in st.session_state:
                initial_query = st.session_state['auto_query']
                del st.session_state['auto_query']
            
            # Query input
            query = st.text_input(
                "Ask me anything about your data:",
                value=initial_query,
                placeholder="e.g., What's the sales conversion rate by publisher?",
                help="Ask natural language questions about your call center performance data",
                key="main_query_input"
            )
            
            # Add manual execution button and query status
            col1, col2 = st.columns([1, 4])
            with col1:
                execute_query = st.button("🚀 Execute Query", key="execute_btn")
            with col2:
                if query:
                    st.write(f"**Ready to analyze:** {query}")
            
            # Sample queries
            st.write("**💡 Try these sample queries:**")
            col1, col2, col3 = st.columns(3)
            
            # Load business metrics for smart suggestions
            business_config = load_business_metrics()
            column_matches = find_column_matches(df.columns, business_config)
            
            sample_queries = [
                "What's the total count of each category?",
                "Show me the distribution of values",
                "Group by category and sum the amounts"
            ]
            
            # Add business metric queries if we have matching columns
            if column_matches:
                business_queries = []
                if 'revenue' in column_matches and 'sale' in column_matches:
                    business_queries.append("What is the cost per sale by publisher?")
                if 'sale' in column_matches:
                    business_queries.append("What's the conversion rate by publisher?")
                if 'revenue' in column_matches:
                    business_queries.append("Show me average revenue per sale")
                
                sample_queries = business_queries + sample_queries
            
            for i, sample in enumerate(sample_queries[:3]):
                with [col1, col2, col3][i]:
                    if st.button(f"📊 {sample}", key=f"sample_{i}"):
                        st.session_state['auto_query'] = sample
                        st.rerun()
            
            # Process query if there's input and either auto-execution or manual button press
            if query and (execute_query or initial_query):
                # Display format options
                col1, col2 = st.columns([3, 1])
                with col1:
                    format_as_percentage = st.checkbox(
                        "📊 Display as percentages (%)", 
                        value="percent" in query.lower(),
                        help="Convert decimal values to percentage format"
                    )
                
                with st.spinner("🤖 GPT-4o is analyzing your query..."):
                    # Get analysis from GPT-4o
                    gpt_result, error = query_with_gpt4o(query, df, client)
                    
                    if error:
                        st.error(f"❌ {error}")
                    elif gpt_result:
                        # Show the analysis
                        st.write("**🎯 GPT-4o Analysis:**")
                        st.write(f"**Explanation:** {gpt_result['explanation']}")
                        st.write(f"**Generated Code:**")
                        st.code(gpt_result['code'], language='python')
                        
                        # Execute the code
                        result, exec_error = execute_code_safely(gpt_result['code'], df)
                        
                        if exec_error:
                            st.error(f"❌ Execution Error: {exec_error}")
                        else:
                            # Create visualization BEFORE percentage formatting (using numeric data)
                            viz = create_visualization(result, gpt_result.get('visualization_type', 'table'), query)
                            
                            # Format percentages if requested or if query mentions percentages
                            if format_as_percentage or any(word in query.lower() for word in ['percentage', 'percent', '%']):
                                formatted_result = format_percentage_results(result, query)
                            else:
                                formatted_result = result
                            
                            # Show results
                            st.write("**📊 Results:**")
                            
                            if isinstance(formatted_result, (int, float)):
                                suffix = "%" if format_as_percentage else ""
                                st.metric("Result", f"{formatted_result:,.2f}{suffix}")
                            elif isinstance(formatted_result, pd.DataFrame):
                                st.dataframe(formatted_result)
                                
                                # Download option
                                csv = formatted_result.to_csv(index=True)
                                st.download_button(
                                    "📥 Download Results as CSV",
                                    csv,
                                    "query_results.csv",
                                    "text/csv"
                                )
                                
                                # Show summary stats for percentage data
                                if format_as_percentage:
                                    # Extract numeric values from percentage strings for summary
                                    numeric_cols = []
                                    for col in formatted_result.columns:
                                        if formatted_result[col].dtype == 'object' and formatted_result[col].astype(str).str.contains('%').any():
                                            numeric_vals = pd.to_numeric(formatted_result[col].astype(str).str.rstrip('%'), errors='coerce')
                                            if not numeric_vals.isna().all():
                                                numeric_cols.append((col, numeric_vals))
                                    
                                    if numeric_cols:
                                        st.write("**📈 Summary:**")
                                        for col_name, values in numeric_cols:
                                            st.write(f"- {col_name}: Average = {values.mean():.1f}%, Range = {values.min():.1f}% - {values.max():.1f}%")
                                
                            elif isinstance(formatted_result, pd.Series):
                                st.dataframe(formatted_result.to_frame())
                            else:
                                st.write(formatted_result)
                            
                            # Show visualization (using numeric data)
                            if viz:
                                st.plotly_chart(viz, use_container_width=True)
                            
                            # Intelligent Summary Section
                            if st.checkbox("🧠 Generate AI Summary", value=True, help="Get GPT-4o insights on performance, outliers, and top performers"):
                                with st.spinner("🤖 Generating comprehensive business intelligence summary..."):
                                    summary = generate_intelligent_summary(result, query, client)
                                    
                                    if summary:
                                        if "error" in summary:
                                            st.error(f"❌ {summary['error']}")
                                        else:
                                            # Full-width Business Intelligence Report
                                            st.markdown("---")
                                            st.markdown("# 📊 Business Intelligence Report")
                                            st.markdown("---")
                                            
                                            # Executive Summary (full width)
                                            if "executive_summary" in summary:
                                                st.markdown("## 🎯 Executive Summary")
                                                st.markdown(summary['executive_summary'])
                                                st.markdown("---")
                                            
                                            # Quartile Performance Analysis (full width)
                                            if "quartile_performance" in summary:
                                                st.markdown("## 📈 Quartile Performance Analysis")
                                                st.markdown(summary['quartile_performance'])
                                                st.markdown("---")
                                            
                                            # Outlier Analysis (full width)
                                            if "outlier_analysis" in summary:
                                                st.markdown("## ⚡ Outlier & Anomaly Analysis")
                                                st.markdown(summary['outlier_analysis'])
                                                st.markdown("---")
                                            
                                            # Top Performers (full width)
                                            if "top_performers" in summary:
                                                st.markdown("## 🏆 Top & Bottom Performers")
                                                st.markdown(summary['top_performers'])
                                                st.markdown("---")
                                            
                                            # Performance Insights (full width)
                                            if "performance_insights" in summary:
                                                st.markdown("## 💡 Performance Insights")
                                                st.markdown(summary['performance_insights'])
                                                st.markdown("---")
                                            
                                            # Strategic Recommendations (full width)
                                            if "recommendations" in summary:
                                                st.markdown("## 🚀 Strategic Recommendations")
                                                st.markdown(summary['recommendations'])
                                                st.markdown("---")
                                            
                                            # Detailed Statistical Data (full width)
                                            if "raw_analysis" in summary and isinstance(summary["raw_analysis"], dict):
                                                raw_data = summary["raw_analysis"]
                                                
                                                st.markdown("## 📊 Detailed Statistical Data & Outlier Specifics")
                                                
                                                if "analysis" in raw_data:
                                                    for col_name, analysis in raw_data["analysis"].items():
                                                        st.markdown(f"### 📈 {col_name} - Complete Statistical Analysis")
                                                        
                                                        # Basic Statistics
                                                        if "basic_stats" in analysis:
                                                            stats = analysis["basic_stats"]
                                                            st.markdown("#### 📊 Basic Statistics")
                                                            
                                                            stats_table = f"""
| Metric | Value |
|--------|-------|
| **Count** | {stats['count']:,} |
| **Mean** | {stats['mean']:.4f} |
| **Median** | {stats['median']:.4f} |
| **Standard Deviation** | {stats['std']:.4f} |
| **Minimum** | {stats['min']:.4f} |
| **Maximum** | {stats['max']:.4f} |
"""
                                                            st.markdown(stats_table)
                                                        
                                                        # Quartile Analysis
                                                        if "quartile_analysis" in analysis:
                                                            q_analysis = analysis["quartile_analysis"]
                                                            st.markdown("#### 🎯 Quartile Performance Breakdown")
                                                            
                                                            quartile_table = f"""
| Quartile | Threshold | Count | Performance Level |
|----------|-----------|-------|------------------|
| **Q1 (25th percentile)** | {q_analysis['q1']:.4f} | {q_analysis['bottom_quartile_count']} performers | Bottom 25% |
| **Q3 (75th percentile)** | {q_analysis['q3']:.4f} | {q_analysis['top_quartile_count']} performers | Top 25% |
| **IQR (Interquartile Range)** | {q_analysis['iqr']:.4f} | - | Spread measure |
"""
                                                            st.markdown(quartile_table)
                                                            
                                                            # Top Quartile Values
                                                            if q_analysis['top_quartile_values']:
                                                                st.markdown("**🔝 Top Quartile Values (≥75th percentile):**")
                                                                top_values = ", ".join([f"{v:.3f}" for v in q_analysis['top_quartile_values'][:10]])
                                                                st.markdown(f"```\n{top_values}\n```")
                                                            
                                                            # Bottom Quartile Values  
                                                            if q_analysis['bottom_quartile_values']:
                                                                st.markdown("**🔻 Bottom Quartile Values (≤25th percentile):**")
                                                                bottom_values = ", ".join([f"{v:.3f}" for v in q_analysis['bottom_quartile_values'][:10]])
                                                                st.markdown(f"```\n{bottom_values}\n```")
                                                        
                                                        # Outlier Analysis
                                                        if "outlier_analysis" in analysis:
                                                            outlier_analysis = analysis["outlier_analysis"]
                                                            st.markdown("#### ⚡ Specific Outlier Details")
                                                            
                                                            outlier_table = f"""
| Outlier Type | Threshold | Count | Status |
|--------------|-----------|-------|--------|
| **High Outliers** | > {outlier_analysis['outlier_threshold_high']:.4f} | {outlier_analysis['outliers_high_count']} | {"⚠️ Detected" if outlier_analysis['outliers_high_count'] > 0 else "✅ None"} |
| **Low Outliers** | < {outlier_analysis['outlier_threshold_low']:.4f} | {outlier_analysis['outliers_low_count']} | {"⚠️ Detected" if outlier_analysis['outliers_low_count'] > 0 else "✅ None"} |
"""
                                                            st.markdown(outlier_table)
                                                            
                                                            # High Outliers
                                                            if outlier_analysis['outliers_high_count'] > 0:
                                                                st.markdown("**🔺 High Outlier Values:**")
                                                                high_outliers = ", ".join([f"**{v:.4f}**" for v in outlier_analysis['outliers_high_values']])
                                                                st.markdown(f"```\n{high_outliers}\n```")
                                                            
                                                            # Low Outliers
                                                            if outlier_analysis['outliers_low_count'] > 0:
                                                                st.markdown("**🔻 Low Outlier Values:**")
                                                                low_outliers = ", ".join([f"**{v:.4f}**" for v in outlier_analysis['outliers_low_values']])
                                                                st.markdown(f"```\n{low_outliers}\n```")
                                                        
                                                        # Performance Distribution
                                                        if "performance_distribution" in analysis:
                                                            perf_dist = analysis["performance_distribution"]
                                                            st.markdown("#### 📊 Performance Distribution")
                                                            
                                                            dist_table = f"""
| Distribution | Count | Percentage |
|--------------|-------|------------|
| **Above Median** | {perf_dist['above_median_count']} | {perf_dist['above_median_pct']}% |
| **Below Median** | {perf_dist['below_median_count']} | {perf_dist['below_median_pct']}% |
"""
                                                            st.markdown(dist_table)
                                                        
                                                        # Top and Bottom Performers
                                                        if "top_performers" in analysis and analysis["top_performers"]:
                                                            st.markdown("#### 🏆 Detailed Performer Analysis")
                                                            
                                                            # Top Performers Table
                                                            st.markdown("**🥇 Top 5 Performers:**")
                                                            top_table = "| Rank | Value | Details |\n|------|-------|----------|\n"
                                                            for i, performer in enumerate(analysis["top_performers"][:5], 1):
                                                                details = ""
                                                                if "PUBLISHER" in performer:
                                                                    details = f"Publisher: {performer['PUBLISHER']}"
                                                                elif len([k for k in performer.keys() if k not in ['value', 'index']]) > 0:
                                                                    other_info = [f"{k}: {v}" for k, v in performer.items() if k not in ['value', 'index']]
                                                                    details = ", ".join(other_info[:2])
                                                                top_table += f"| {i} | **{performer['value']:.4f}** | {details} |\n"
                                                            st.markdown(top_table)
                                                            
                                                            # Bottom Performers Table  
                                                            st.markdown("**🥉 Bottom 5 Performers:**")
                                                            bottom_table = "| Rank | Value | Details |\n|------|-------|----------|\n"
                                                            for i, performer in enumerate(analysis["bottom_performers"][:5], 1):
                                                                details = ""
                                                                if "PUBLISHER" in performer:
                                                                    details = f"Publisher: {performer['PUBLISHER']}"
                                                                elif len([k for k in performer.keys() if k not in ['value', 'index']]) > 0:
                                                                    other_info = [f"{k}: {v}" for k, v in performer.items() if k not in ['value', 'index']]
                                                                    details = ", ".join(other_info[:2])
                                                                bottom_table += f"| {i} | **{performer['value']:.4f}** | {details} |\n"
                                                            st.markdown(bottom_table)
                                                        
                                                        st.markdown("---")
                                                
                                                st.markdown("*Report generated using quartile-based analysis with IQR outlier detection*")
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'query': query,
                                'result': formatted_result,
                                'code': gpt_result['code'],
                                'explanation': gpt_result['explanation']
                            })
            
            # Show chat history
            if st.session_state.chat_history:
                st.header("📚 Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                    with st.expander(f"Query {len(st.session_state.chat_history)-i}: {chat['query']}"):
                        st.write(f"**Explanation:** {chat['explanation']}")
                        st.code(chat['code'])
                        if isinstance(chat['result'], pd.DataFrame):
                            st.dataframe(chat['result'])
                        else:
                            st.write(chat['result'])
                
                if st.button("🗑️ Clear History"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload an Excel file to get started")
        
        # Show demo information
        st.header("🌟 Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🤖 Powered by GPT-4o:**")
            st.write("- Advanced natural language understanding")
            st.write("- Intelligent pandas code generation")
            st.write("- Context-aware data analysis")
            
        with col2:
            st.write("**📊 Smart Analysis:**")
            st.write("- Automatic case-insensitive matching")
            st.write("- Proper grouping and aggregation")
            st.write("- Interactive visualizations")

if __name__ == "__main__":
    main() 