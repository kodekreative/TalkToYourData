#!/usr/bin/env python3
"""
Simple, working version of TalkToYourData
Bypasses all the caching and nesting issues
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
    """Initialize OpenAI client."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != "your_openai_api_key_here":
        return OpenAI(api_key=api_key)
    return None

def analyze_query_with_openai(query, df, client):
    """Use OpenAI to analyze the query and identify relevant columns."""
    try:
        columns_info = []
        for col in df.columns:
            sample_values = df[col].dropna().head(3).tolist()
            columns_info.append(f"- {col} ({df[col].dtype}): Sample values: {sample_values}")
        
        columns_text = "\n".join(columns_info)
        
        prompt = f"""
        You are analyzing a dataset query. Given this query: "{query}"
        
        Available columns:
        {columns_text}
        
        Please analyze and respond with ONLY a JSON object in this exact format:
        {{
            "target_columns": ["column1"],
            "group_by_columns": ["column2"],
            "filter_conditions": ["value1"],
            "intent": "sum|count|group_count|mean|max|min|distribution",
            "explanation": "Brief explanation of what to calculate"
        }}
        
        Rules:
        - target_columns: Columns that contain the data to analyze (e.g., SALE column)
        - group_by_columns: Columns to group by (e.g., if query says "by buyer", include BUYER column)
        - filter_conditions: Values to filter for (e.g., if query says 'yes', include "yes")
        - For queries with "by [something]", use intent "group_count" and include the grouping column
        - For simple counting without grouping, use intent "count"
        - Use exact column names as they appear in the dataset
        - Look for keywords like "by", "group by", "for each" to identify grouping
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content.strip()
        st.write(f"**🤖 OpenAI Raw Response:** {response_text}")  # Debug output
        
        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                # Validate the columns exist in the dataframe
                valid_target_cols = []
                for col in analysis.get('target_columns', []):
                    if col in df.columns:
                        valid_target_cols.append(col)
                    else:
                        # Try to find a close match (case insensitive)
                        close_matches = get_close_matches(col.lower(), [c.lower() for c in df.columns], n=1, cutoff=0.8)
                        if close_matches:
                            actual_col = [c for c in df.columns if c.lower() == close_matches[0]][0]
                            valid_target_cols.append(actual_col)
                
                valid_group_cols = []
                for col in analysis.get('group_by_columns', []):
                    if col in df.columns:
                        valid_group_cols.append(col)
                    else:
                        # Try to find a close match (case insensitive)
                        close_matches = get_close_matches(col.lower(), [c.lower() for c in df.columns], n=1, cutoff=0.8)
                        if close_matches:
                            actual_col = [c for c in df.columns if c.lower() == close_matches[0]][0]
                            valid_group_cols.append(actual_col)
                
                analysis['target_columns'] = valid_target_cols
                analysis['group_by_columns'] = valid_group_cols
                analysis['filter_conditions'] = analysis.get('filter_conditions', [])
                analysis['key_terms_used'] = [query]  # Add for compatibility
                
                return analysis
            
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing failed: {str(e)}")
            st.write(f"Raw response: {response_text}")
            
    except Exception as e:
        st.warning(f"OpenAI analysis failed: {str(e)}")
    
    # Fallback to simple analysis
    return None

def analyze_query_simple(query, df):
    """Improved rule-based query analysis with better column matching."""
    query_lower = query.lower()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Find relevant columns with more precise matching
    target_cols = []
    group_by_cols = []
    filter_conditions = []
    
    # Extract key terms from the query
    key_terms = []
    
    # Look for specific column-related terms
    if 'sales' in query_lower or 'sale' in query_lower:
        key_terms.extend(['sale', 'sales'])
    if 'buyer' in query_lower or 'customer' in query_lower:
        key_terms.extend(['buyer', 'customer', 'purchaser', 'client'])
    if 'revenue' in query_lower:
        key_terms.append('revenue')
    if 'amount' in query_lower:
        key_terms.append('amount')
    if 'total' in query_lower:
        key_terms.append('total')
    if 'price' in query_lower:
        key_terms.append('price')
    
    # Look for grouping patterns
    group_by_patterns = []
    if ' by ' in query_lower:
        # Extract what comes after "by"
        by_parts = query_lower.split(' by ')
        if len(by_parts) > 1:
            by_term = by_parts[1].split()[0]  # First word after "by"
            group_by_patterns.append(by_term)
            if by_term not in key_terms:
                key_terms.append(by_term)
    
    # Look for filter conditions (quoted values or specific terms)
    quoted_values = re.findall(r"'([^']*)'", query_lower)
    if quoted_values:
        filter_conditions.extend(quoted_values)
    
    # If no specific terms, extract words from query
    if not key_terms:
        # Remove common words and focus on meaningful terms
        common_words = {'the', 'of', 'and', 'or', 'how', 'many', 'what', 'is', 'are', 'total', 'sum', 'show', 'me', 'took', 'place', 'by'}
        query_words = [word for word in query_lower.split() if word not in common_words and len(word) > 2]
        key_terms = query_words[:3]  # Take first 3 meaningful words
    
    # Find columns that match key terms
    for term in key_terms:
        for col in df.columns:
            col_lower = col.lower()
            # Exact match or term is a significant part of the column name
            if (term in col_lower and 
                (term == col_lower or len(term) >= len(col_lower) * 0.4)):
                if col not in target_cols:
                    target_cols.append(col)
                    
                    # Check if this should be a group-by column
                    if term in group_by_patterns:
                        group_by_cols.append(col)
    
    # If still no matches, try fuzzy matching with get_close_matches
    if not target_cols and key_terms:
        for term in key_terms:
            close_matches = get_close_matches(term, [c.lower() for c in df.columns], n=2, cutoff=0.6)
            for match in close_matches:
                actual_col = [c for c in df.columns if c.lower() == match][0]
                if actual_col not in target_cols:
                    target_cols.append(actual_col)
                    
                    # Check if this should be a group-by column
                    if term in group_by_patterns:
                        group_by_cols.append(actual_col)
    
    # If still no matches and it's a numeric query, default to first numeric column
    if not target_cols and any(word in query_lower for word in ['total', 'sum', 'how many', 'count']):
        if numeric_cols:
            target_cols = [numeric_cols[0]]
    
    # Determine intent
    if any(word in query_lower for word in ['percentage', 'percent', '%', 'distribution', 'breakdown']):
        intent = 'distribution'
    elif any(word in query_lower for word in ['total', 'sum']):
        intent = 'sum'
    elif any(word in query_lower for word in ['how many', 'count']):
        # Check if it's a grouped count
        if group_by_cols or ' by ' in query_lower:
            intent = 'group_count'
        else:
            intent = 'count'
    elif any(word in query_lower for word in ['average', 'mean']):
        intent = 'mean' 
    elif any(word in query_lower for word in ['trend', 'over time']):
        intent = 'trend'
    elif any(word in query_lower for word in ['max', 'maximum', 'highest']):
        intent = 'max'
    elif any(word in query_lower for word in ['min', 'minimum', 'lowest']):
        intent = 'min'
    else:
        intent = 'sum'
    
    return {
        'target_columns': target_cols,
        'group_by_columns': group_by_cols,
        'filter_conditions': filter_conditions,
        'intent': intent,
        'query': query,
        'key_terms_used': key_terms
    }

def execute_query(analysis, df):
    """Execute the analysis and return results."""
    target_cols = analysis['target_columns']
    intent = analysis['intent']
    
    try:
        if not target_cols:
            return df.head(), "No relevant columns found"
        
        result_df = df[target_cols].copy()
        
        if intent == 'sum':
            if result_df.select_dtypes(include=['number']).columns.any():
                summary = result_df.select_dtypes(include=['number']).sum()
                total = summary.sum() if len(summary) > 1 else summary.iloc[0] if len(summary) == 1 else 0
                explanation = f"Total {', '.join(target_cols)}: {total:,.2f}"
            else:
                explanation = f"Count of records: {len(result_df)}"
        elif intent == 'count':
            if result_df.select_dtypes(include=['number']).columns.any():
                # Count non-zero values for numeric columns
                non_zero_count = (result_df.select_dtypes(include=['number']) > 0).sum().sum()
                explanation = f"Count of non-zero {', '.join(target_cols)}: {non_zero_count:,}"
            else:
                explanation = f"Count of records: {len(result_df):,}"
        elif intent == 'group_count':
            # Handle grouped counting with filtering
            group_cols = analysis.get('group_by_columns', [])
            filter_conditions = analysis.get('filter_conditions', [])
            
            if group_cols and len(group_cols) > 0:
                group_col = group_cols[0]  # Use first group column
                
                # Apply filters if specified
                filtered_df = df.copy()
                filter_applied = False
                
                for condition in filter_conditions:
                    for col in target_cols:
                        if condition.lower() in df[col].astype(str).str.lower().values:
                            # Filter for this condition
                            filtered_df = filtered_df[
                                filtered_df[col].astype(str).str.lower().str.contains(condition.lower(), na=False)
                            ]
                            filter_applied = True
                            break
                
                # Group by the specified column and count
                if group_col in filtered_df.columns:
                    grouped_result = filtered_df.groupby(group_col).size().reset_index(name='Count')
                    grouped_result = grouped_result.sort_values('Count', ascending=False)
                    
                    filter_text = f" where {', '.join(target_cols)} contains '{', '.join(filter_conditions)}'" if filter_applied else ""
                    explanation = f"Count of records{filter_text} grouped by {group_col}:\n"
                    
                    for _, row in grouped_result.head(10).iterrows():
                        explanation += f"{row[group_col]}: {row['Count']:,}\n"
                    
                    return grouped_result, explanation
                else:
                    explanation = f"Group column '{group_col}' not found in data"
            else:
                explanation = "No valid group-by column found for grouped count"
        elif intent == 'mean':
            if result_df.select_dtypes(include=['number']).columns.any():
                summary = result_df.select_dtypes(include=['number']).mean()
                explanation = f"Average {', '.join(target_cols)}: {summary.to_dict()}"
            else:
                explanation = f"Count of records: {len(result_df)}"
        elif intent == 'max':
            if result_df.select_dtypes(include=['number']).columns.any():
                summary = result_df.select_dtypes(include=['number']).max()
                explanation = f"Maximum {', '.join(target_cols)}: {summary.to_dict()}"
            else:
                explanation = f"Count of records: {len(result_df)}"
        elif intent == 'min':
            if result_df.select_dtypes(include=['number']).columns.any():
                summary = result_df.select_dtypes(include=['number']).min()
                explanation = f"Minimum {', '.join(target_cols)}: {summary.to_dict()}"
            else:
                explanation = f"Count of records: {len(result_df)}"
        elif intent == 'distribution':
            # Calculate value counts and percentages
            if len(target_cols) == 1:
                col = target_cols[0]
                value_counts = result_df[col].value_counts()
                percentages = result_df[col].value_counts(normalize=True) * 100
                
                # Create a summary DataFrame
                summary_df = pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': percentages.values
                })
                
                # Format explanation
                explanation_parts = []
                for idx, row in summary_df.iterrows():
                    explanation_parts.append(f"{row['Value']}: {row['Count']} ({row['Percentage']:.1f}%)")
                
                explanation = f"Distribution of {col}:\n" + "\n".join(explanation_parts)
                
                return summary_df, explanation
            else:
                explanation = f"Distribution analysis not supported for multiple columns: {target_cols}"
        else:
            explanation = f"Showing data for {', '.join(target_cols)}"
        
        return result_df, explanation
        
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return df.head(), f"Error: {str(e)}"

def create_visualization(result_df, analysis):
    """Create a simple visualization."""
    try:
        if result_df.empty:
            return None
        
        intent = analysis.get('intent', '')
        
        # Special handling for distribution/percentage queries
        if intent == 'distribution' and 'Percentage' in result_df.columns:
            # Create a pie chart for percentages
            fig = px.pie(result_df, values='Percentage', names='Value', 
                        title=f"Percentage Distribution")
            return fig
        
        numeric_cols = result_df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            # Show value counts for categorical data
            if len(result_df.columns) > 0:
                col = result_df.columns[0]
                if col == 'Value' and 'Count' in result_df.columns:
                    # This is likely a value counts result
                    fig = px.bar(result_df, x='Value', y='Count', 
                               title=f"Distribution")
                    return fig
                else:
                    value_counts = result_df[col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=f"Distribution of {col}")
                    return fig
            return None
        
        elif len(numeric_cols) == 1:
            # Single numeric column - show histogram
            col = numeric_cols[0]
            fig = px.histogram(result_df, x=col, title=f"Distribution of {col}")
            return fig
        
        else:
            # Multiple numeric columns - show bar chart of sums
            sums = result_df[numeric_cols].sum()
            fig = px.bar(x=sums.index, y=sums.values, 
                        title=f"Totals by Column")
            return fig
            
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Simple TalkToYourData", layout="wide")
    
    st.title("💬 Simple TalkToYourData")
    st.write("Upload Excel data and ask questions in natural language!")
    
    # Initialize session state for query history
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    
    # Initialize OpenAI
    client = init_openai()
    if client:
        st.success("✅ OpenAI connected successfully!")
    else:
        st.warning("⚠️ OpenAI not connected. Using simple rule-based analysis.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file:
        try:
            # Load data
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head())
            
            # Show column info
            st.subheader("📋 Columns")
            col_info = []
            for col in df.columns:
                col_info.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Non-null': df[col].count(),
                    'Sample': str(df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else 'N/A')
                })
            st.dataframe(pd.DataFrame(col_info))
            
            # Query interface
            st.subheader("💬 Ask a Question")
            
            # Query history section
            if st.session_state.query_history:
                st.write("**🔄 Recent Queries (Click to Rerun):**")
                cols = st.columns(min(len(st.session_state.query_history), 3))
                for i, historical_query in enumerate(st.session_state.query_history[-3:]):  # Show last 3
                    with cols[i]:
                        if st.button(f"🔄 {historical_query[:30]}...", key=f"rerun_{i}"):
                            st.session_state.query_input = historical_query
                            st.rerun()
            
            # Suggested questions
            st.write("**💡 Try these examples:**")
            suggestions = [
                f"What is the total of {df.select_dtypes(include=['number']).columns[0]}?" if len(df.select_dtypes(include=['number']).columns) > 0 else "How many records are there?",
                f"Show me the distribution of {df.columns[0]}",
                "What is the summary of this data?"
            ]
            
            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggest_{suggestion[:20]}"):
                    st.session_state.query_input = suggestion
                    st.rerun()
            
            # Text input with retest button
            col1, col2 = st.columns([4, 1])
            with col1:
                query = st.text_input("Your question:", key="query_input", placeholder="e.g., What is the total sales?")
            with col2:
                st.write("")  # Add some spacing
                if st.button("🔄 Retest Last", disabled=not st.session_state.last_query):
                    query = st.session_state.last_query
                    st.session_state.query_input = query
                    st.rerun()
            
            if query:
                # Add to history if it's a new query
                if query != st.session_state.last_query:
                    if query not in st.session_state.query_history:
                        st.session_state.query_history.append(query)
                        # Keep only last 10 queries
                        if len(st.session_state.query_history) > 10:
                            st.session_state.query_history.pop(0)
                    st.session_state.last_query = query
                
                st.write(f"**🔍 Processing:** {query}")
                
                # Analyze query - try OpenAI first, then fallback to simple
                if client:
                    analysis = analyze_query_with_openai(query, df, client)
                    if analysis:
                        st.success("🤖 Using OpenAI analysis")
                        # Fill in missing fields with rule-based analysis if needed
                        if not analysis.get('group_by_columns') or not analysis.get('filter_conditions'):
                            rule_analysis = analyze_query_simple(query, df)
                            if not analysis.get('group_by_columns') and rule_analysis.get('group_by_columns'):
                                analysis['group_by_columns'] = rule_analysis['group_by_columns']
                                st.write("**🔧 Enhanced with rule-based group detection**")
                            if not analysis.get('filter_conditions') and rule_analysis.get('filter_conditions'):
                                analysis['filter_conditions'] = rule_analysis['filter_conditions']
                                st.write("**🔧 Enhanced with rule-based filter detection**")
                            # Update intent if grouping was found
                            if analysis.get('group_by_columns') and analysis.get('intent') == 'count':
                                analysis['intent'] = 'group_count'
                                st.write("**🔧 Updated intent to group_count**")
                    else:
                        st.warning("⚠️ OpenAI failed, using rule-based analysis")
                        analysis = analyze_query_simple(query, df)
                else:
                    analysis = analyze_query_simple(query, df)
                    st.info("🔧 Using rule-based analysis")
                
                st.session_state.last_analysis = analysis
                
                st.write(f"**🎯 Detected columns:** {analysis['target_columns']}")
                st.write(f"**🔧 Intent:** {analysis['intent']}")
                if 'group_by_columns' in analysis and analysis['group_by_columns']:
                    st.write(f"**👥 Group by columns:** {analysis['group_by_columns']}")
                if 'filter_conditions' in analysis and analysis['filter_conditions']:
                    st.write(f"**🔍 Filter conditions:** {analysis['filter_conditions']}")
                if 'key_terms_used' in analysis:
                    st.write(f"**🔑 Key terms found:** {analysis['key_terms_used']}")
                if 'explanation' in analysis:
                    st.write(f"**💡 AI explanation:** {analysis['explanation']}")
                
                # Execute query
                result_df, explanation = execute_query(analysis, df)
                st.session_state.last_result = (result_df, explanation)
                
                # Add quick retest button for current query
                col_retest1, col_retest2, col_retest3 = st.columns([1, 1, 2])
                with col_retest1:
                    if st.button("🔄 Rerun This Query"):
                        st.rerun()
                with col_retest2:
                    if st.button("🧪 Test Similar"):
                        # Suggest similar queries
                        similar_queries = []
                        for col in analysis['target_columns']:
                            similar_queries.extend([
                                f"What is the average {col}?",
                                f"Show me the maximum {col}",
                                f"How many unique {col} values are there?"
                            ])
                        st.session_state.similar_suggestions = similar_queries[:3]
                        st.rerun()
                
                # Show similar suggestions if they exist
                if hasattr(st.session_state, 'similar_suggestions') and st.session_state.similar_suggestions:
                    st.write("**🧪 Try These Similar Queries:**")
                    for sim_query in st.session_state.similar_suggestions:
                        if st.button(sim_query, key=f"sim_{sim_query[:15]}"):
                            st.session_state.query_input = sim_query
                            st.rerun()
                
                # Show results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Results")
                    st.write(f"**📝 Explanation:** {explanation}")
                    st.dataframe(result_df.head(10))
                    
                    # Add download button for results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results as CSV",
                        data=csv,
                        file_name=f"query_results_{query[:20].replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.subheader("📈 Visualization")
                    fig = create_visualization(result_df, analysis)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No visualization available for this data")
                
                # Show debug info
                with st.expander("🔧 Debug Information"):
                    debug_info = {
                        'query': query,
                        'analysis': analysis,
                        'result_shape': result_df.shape,
                        'columns_found': list(result_df.columns),
                        'query_history': st.session_state.query_history
                    }
                    st.json(debug_info)
                    
                    # Add button to clear history
                    if st.button("🗑️ Clear Query History"):
                        st.session_state.query_history = []
                        st.session_state.last_query = ""
                        st.success("Query history cleared!")
                        st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.write("**Debug info:**")
            st.write(f"File name: {uploaded_file.name}")
            st.write(f"File size: {uploaded_file.size}")

if __name__ == "__main__":
    main() 