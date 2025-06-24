#!/usr/bin/env python3
"""
Performance Marketing Diagnostic Tool
Comprehensive analysis of Publisher-Buyer-Target combinations
Identifies lead quality vs sales execution issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from advanced_diagnostic_features import create_advanced_analysis_page, create_executive_summary_page

# Import talk to your data functions
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Performance Marketing Diagnostic Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Talk to Your Data Functions (imported from pandasai_app.py)
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

USER QUERY: "{query}"

Generate Python pandas code that:
1. Uses ONLY the variable name 'df' for the dataframe
2. MUST assign final result to variable named 'result'
3. Do NOT create intermediate variables - write everything in one statement when possible
4. For column detection, always use this exact pattern: [col for col in df.columns if 'KEYWORD' in col.upper()][0]
5. For groupby queries, always use reset_index() and assign column names
6. For calculations, return actual calculated values
7. Available functions: pd, np, len, sum, max, min, round, str, int, float
8. CRITICAL: Never use undefined variables like 'ad_misled_col', 'agent_col', 'ser', 'temp'
9. Always define any column reference inline within the same statement

EXAMPLES:

For "conversion rate by publisher":
{{
    "code": "result = df.groupby('PUBLISHER').apply(lambda x: (x['SALE'] == 'Yes').sum() / len(x) * 100).reset_index(); result.columns = ['PUBLISHER', 'Conversion_Rate']; result",
    "explanation": "Calculate conversion rate by publisher as percentage of total leads",
    "visualization_type": "bar"
}}

For "agent availability rates by buyer":
{{
    "code": "result = df.groupby('BUYER').apply(lambda x: (x[[col for col in df.columns if 'AGENT' in col.upper()][0]] == 'Yes').sum() / len(x) * 100).reset_index(); result.columns = ['BUYER', 'Agent_Availability_Rate']; result",
    "explanation": "Calculate agent availability rate by buyer as percentage",
    "visualization_type": "bar"
}}

For "ad misled calls by publisher":
{{
    "code": "result = df.groupby('PUBLISHER').apply(lambda x: (x[[col for col in df.columns if 'AD_MISLED' in col.upper()][0]] == 'Yes').sum()).reset_index(); result.columns = ['PUBLISHER', 'Ad_Misled_Count']; result['Percentage'] = (result['Ad_Misled_Count'] / df.groupby('PUBLISHER').size().values) * 100; result",
    "explanation": "Calculate ad misled calls count and percentage by publisher",
    "visualization_type": "bar"
}}

Return ONLY a JSON object with this EXACT format:
{{
    "code": "pandas code here",
    "explanation": "brief explanation of what the code does",
    "visualization_type": "bar"
}}
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
        # Create a safe execution environment with more pandas functionality
        safe_globals = {
            'pd': pd,
            'np': np,
            'df': df,
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'round': round,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            '__builtins__': {
                'len': len,
                'sum': sum,
                'max': max,
                'min': min,
                'round': round,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
            }
        }
        
        # Add some common variables that might be generated
        safe_locals = {}
        
        # Execute the code with both globals and locals
        exec(code, safe_globals, safe_locals)
        
        # Return the result - check both locals and globals
        if 'result' in safe_locals:
            return safe_locals['result'], None
        elif 'result' in safe_globals:
            return safe_globals['result'], None
        else:
            # Check if there's any dataframe or series in locals that could be the result
            for var_name, var_value in safe_locals.items():
                if isinstance(var_value, (pd.DataFrame, pd.Series)):
                    return var_value, None
            return None, "No 'result' variable found in executed code"
            
    except Exception as e:
        return None, f"Execution error: {str(e)}"

def create_visualization(result, viz_type, query):
    """Create visualization based on result and type."""
    try:
        if not isinstance(result, (pd.DataFrame, pd.Series)):
            return None
        
        if isinstance(result, pd.Series):
            result = result.to_frame()
        
        if len(result) == 0:
            return None
        
        # Determine x and y columns
        if len(result.columns) >= 2:
            x_col = result.columns[0]
            y_col = result.columns[1]
        else:
            x_col = result.index.name or 'Index'
            y_col = result.columns[0]
            result = result.reset_index()
        
        # Create appropriate visualization
        if viz_type == "bar":
            fig = px.bar(result, x=x_col, y=y_col, title=f"{query}")
        elif viz_type == "line":
            fig = px.line(result, x=x_col, y=y_col, title=f"{query}")
        elif viz_type == "scatter":
            fig = px.scatter(result, x=x_col, y=y_col, title=f"{query}")
        else:
            fig = px.bar(result, x=x_col, y=y_col, title=f"{query}")
        
        fig.update_layout(height=500)
        return fig
        
    except Exception as e:
        return None

def generate_intelligent_summary(result, query, client):
    """Generate comprehensive business intelligence summary."""
    if not client:
        return {"error": "OpenAI client not available"}
    
    try:
        # Prepare data summary for GPT-4o
        if isinstance(result, pd.DataFrame):
            data_summary = {
                "shape": result.shape,
                "columns": result.columns.tolist(),
                "sample_data": result.head(10).to_dict() if len(result) > 0 else {},
                "summary_stats": result.describe().to_dict() if len(result) > 0 else {}
            }
        else:
            data_summary = {"single_value": str(result)}
        
        prompt = f"""
You are a senior business intelligence analyst. Analyze the following query results and provide comprehensive insights.

QUERY: "{query}"
DATA SUMMARY: {json.dumps(data_summary, default=str)}

Provide a comprehensive business intelligence report with the following sections:

1. **Executive Summary** (2-3 paragraphs): High-level insights and key findings
2. **Performance Analysis** (2-3 paragraphs): Detailed analysis of the data patterns
3. **Strategic Recommendations** (3-4 actionable bullet points): Specific recommendations for improvement

Format your response as a JSON object with keys: "executive_summary", "performance_analysis", "recommendations"
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown if present
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse AI summary response"}
            
    except Exception as e:
        return {"error": f"Error generating summary: {str(e)}"}

def create_talk_to_data_page(diagnostic):
    """Create the Talk to Your Data page"""
    st.header("🤖 Talk to Your Data")
    st.subheader("AI-Powered Data Analysis with GPT-4o")
    
    if diagnostic.data is None:
        st.warning("Please load data first from the sidebar")
        return
    
    # Initialize OpenAI client
    client = init_openai()
    if not client:
        st.error("⚠️ OpenAI API key not found. Please add your API key to the .env file.")
        st.code("OPENAI_API_KEY=your_openai_api_key_here")
        return
    
    st.success("✅ Connected to OpenAI GPT-4o")
    
    # Show data preview
    with st.expander("📊 Data Preview", expanded=False):
        st.dataframe(diagnostic.data.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Columns:**")
            for col in diagnostic.data.columns:
                st.write(f"- {col} ({diagnostic.data[col].dtype})")
        
        with col2:
            st.write("**Summary:**")
            st.write(f"- Total rows: {diagnostic.data.shape[0]:,}")
            st.write(f"- Total columns: {diagnostic.data.shape[1]}")
            st.write(f"- Numeric columns: {len(diagnostic.data.select_dtypes(include=['number']).columns)}")
            st.write(f"- Text columns: {len(diagnostic.data.select_dtypes(include=['object']).columns)}")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    st.header("💬 Ask Questions About Your Data")
    
    # Query input
    query = st.text_input(
        "Ask me anything about your performance marketing data:",
        placeholder="e.g., What's the conversion rate by publisher?",
        help="Ask natural language questions about your data",
        key="chat_query_input"
    )
    
    # Add manual execution button and query status
    col1, col2 = st.columns([1, 4])
    with col1:
        execute_query = st.button("🚀 Execute Query", key="chat_execute_btn")
    with col2:
        if query:
            st.write(f"**Ready to analyze:** {query}")
    
    # Sample queries
    st.write("**💡 Try these sample queries:**")
    col1, col2, col3 = st.columns(3)
    
    sample_queries = [
        "What's the conversion rate by publisher?",
        "Show me the average duration by buyer",
        "Which target has the highest sales?"
    ]
    
    for i, sample in enumerate(sample_queries):
        with [col1, col2, col3][i]:
            if st.button(f"📊 {sample}", key=f"sample_chat_{i}"):
                st.session_state['auto_chat_query'] = sample
                st.rerun()
    
    # Check for auto-generated query
    if 'auto_chat_query' in st.session_state:
        query = st.session_state['auto_chat_query']
        del st.session_state['auto_chat_query']
        execute_query = True
    
    # Process query
    if query and execute_query:
        with st.spinner("🤖 GPT-4o is analyzing your query..."):
            # Get analysis from GPT-4o
            gpt_result, error = query_with_gpt4o(query, diagnostic.data, client)
            
            if error:
                st.error(f"❌ {error}")
            elif gpt_result:
                # Show the analysis
                st.write("**🎯 GPT-4o Analysis:**")
                st.write(f"**Explanation:** {gpt_result['explanation']}")
                st.write(f"**Generated Code:**")
                st.code(gpt_result['code'], language='python')
                
                # Execute the code
                result, exec_error = execute_code_safely(gpt_result['code'], diagnostic.data)
                
                if exec_error:
                    st.error(f"❌ Execution Error: {exec_error}")
                else:
                    # Show results
                    st.write("**📊 Results:**")
                    
                    if isinstance(result, (int, float)):
                        st.metric("Result", f"{result:,.2f}")
                    elif isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                        
                        # Download option
                        csv = result.to_csv(index=True)
                        st.download_button(
                            "📥 Download Results as CSV",
                            csv,
                            "query_results.csv",
                            "text/csv"
                        )
                    elif isinstance(result, pd.Series):
                        st.dataframe(result.to_frame())
                    else:
                        st.write(result)
                    
                    # Show visualization
                    viz = create_visualization(result, gpt_result.get('visualization_type', 'bar'), query)
                    if viz:
                        st.plotly_chart(viz, use_container_width=True)
                    
                    # AI Summary
                    if st.checkbox("🧠 Generate AI Summary", value=True, help="Get GPT-4o insights"):
                        with st.spinner("🤖 Generating business intelligence summary..."):
                            summary = generate_intelligent_summary(result, query, client)
                            
                            if summary:
                                if "error" in summary:
                                    st.error(f"❌ {summary['error']}")
                                else:
                                    st.markdown("---")
                                    st.markdown("## 📊 AI Business Intelligence Report")
                                    
                                    if "executive_summary" in summary:
                                        st.markdown("### 🎯 Executive Summary")
                                        st.markdown(summary['executive_summary'])
                                    
                                    if "performance_analysis" in summary:
                                        st.markdown("### 📈 Performance Analysis")
                                        st.markdown(summary['performance_analysis'])
                                    
                                    if "recommendations" in summary:
                                        st.markdown("### 🚀 Strategic Recommendations")
                                        st.markdown(summary['recommendations'])
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'query': query,
                        'result': result,
                        'code': gpt_result['code'],
                        'explanation': gpt_result['explanation']
                    })
    
    # Show chat history
    if st.session_state.chat_history:
        st.header("📚 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Query {len(st.session_state.chat_history)-i}: {chat['query']}"):
                st.write(f"**Explanation:** {chat['explanation']}")
                st.code(chat['code'], language='python')
                if isinstance(chat['result'], pd.DataFrame):
                    st.dataframe(chat['result'])
                else:
                    st.write(chat['result'])
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

def create_voice_test_page(diagnostic):
    """Create voice interaction test page"""
    st.header("🎤 Voice Interaction Test Page")
    st.subheader("AI-Powered Voice Analysis with OpenAI Whisper + ElevenLabs")
    
    if diagnostic.data is None:
        st.warning("⚠️ Please load data first from the sidebar to enable voice analysis")
        return
    
    # Check for API keys
    openai_client = init_openai()
    if not openai_client:
        st.error("⚠️ OpenAI API key not found. Please add your API key to the .env file.")
        st.code("OPENAI_API_KEY=your_openai_api_key_here", language="bash")
        return
    
    # ElevenLabs setup check
    elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
    if not elevenlabs_api_key or elevenlabs_api_key == "your_elevenlabs_api_key_here":
        st.warning("⚠️ ElevenLabs API key not found. Add to .env file for voice responses:")
        st.code("ELEVENLABS_API_KEY=your_elevenlabs_api_key_here", language="bash")
        st.info("Voice input will work without ElevenLabs, but responses will be text-only.")
        elevenlabs_available = False
    else:
        elevenlabs_available = True
        st.success("✅ Connected to OpenAI Whisper and ElevenLabs")
    
    # Initialize session state for voice conversation
    if 'voice_conversation' not in st.session_state:
        st.session_state.voice_conversation = []
    
    # Voice input section
    st.markdown("---")
    st.subheader("🗣️ Voice Input")
    st.write("Click the microphone and ask questions about your performance marketing data:")
    
    # Example queries
    with st.expander("💡 Try these voice queries", expanded=False):
        st.write("**Sample questions you can ask:**")
        st.write("• 'What are my top performing publishers?'")
        st.write("• 'Show me ad misled rates by buyer'")
        st.write("• 'Which campaigns have the highest conversion rates?'")
        st.write("• 'Are there any critical performance alerts?'")
        st.write("• 'What's my agent availability rate?'")
    
    # Voice recorder
    try:
        from streamlit_mic_recorder import mic_recorder
        
        audio_data = mic_recorder(
            start_prompt="🎤 Click to start recording your question",
            stop_prompt="⏹️ Click to stop recording",
            just_once=True,
            use_container_width=True,
            key="voice_test_recorder"
        )
        
        if audio_data:
            # Show listening animation
            listening_placeholder = st.empty()
            with listening_placeholder:
                create_listening_animation()
            
            # Process the audio
            with st.spinner("🤖 Processing your voice query..."):
                # Clear listening animation
                listening_placeholder.empty()
                
                # Convert audio to text using OpenAI Whisper
                try:
                    # Save audio to temporary file for Whisper API
                    import tempfile
                    import io
                    
                    # Create a file-like object from the audio bytes
                    audio_file = io.BytesIO(audio_data['bytes'])
                    audio_file.name = "audio.wav"  # Whisper API needs a filename
                    
                    # Transcribe using OpenAI Whisper
                    transcription = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                    
                    user_query = transcription.strip()
                    st.success(f"🗣️ **You said:** {user_query}")
                    
                    # Process the query with existing diagnostic engine
                    with st.spinner("📊 Analyzing your data..."):
                        gpt_result, error = query_with_gpt4o(user_query, diagnostic.data, openai_client)
                        
                        if error:
                            st.error(f"❌ Analysis Error: {error}")
                            response_text = f"I encountered an error analyzing your question: {error}"
                        elif gpt_result:
                            # Execute the generated code
                            result, exec_error = execute_code_safely(gpt_result['code'], diagnostic.data)
                            
                            if exec_error:
                                st.error(f"❌ Execution Error: {exec_error}")
                                response_text = f"I understand your question about {user_query}, but encountered a technical issue processing the data."
                            else:
                                # Generate human-readable response
                                response_text = generate_voice_friendly_response(user_query, gpt_result, result)
                                
                                # Show visual results
                                st.subheader("📊 Analysis Results")
                                
                                # Display data results
                                if isinstance(result, pd.DataFrame):
                                    st.dataframe(result, use_container_width=True)
                                    
                                    # Generate visualization
                                    viz = create_visualization(result, gpt_result.get('visualization_type', 'bar'), user_query)
                                    if viz:
                                        st.plotly_chart(viz, use_container_width=True)
                                elif isinstance(result, pd.Series):
                                    st.dataframe(result.to_frame(), use_container_width=True)
                                else:
                                    st.write(f"**Result:** {result}")
                        else:
                            response_text = "I'm having trouble understanding your question. Could you please rephrase it?"
                    
                                            # Generate voice response with animation
                        st.subheader("🔊 Voice Response")
                        
                        if elevenlabs_available:
                            try:
                                # Show speaking animation
                                animation_placeholder = st.empty()
                                with animation_placeholder:
                                    create_speaking_animation()
                                
                                # Generate audio response using ElevenLabs
                                audio_response = generate_elevenlabs_response(response_text, elevenlabs_api_key)
                                
                                # Standard Streamlit audio (requires manual play)
                                st.audio(audio_response, format='audio/mp3')
                                
                                # Auto-play version with faster speech
                                st.write("🔊 **Auto-Playing Response:**")
                                import base64
                                audio_base64 = base64.b64encode(audio_response).decode()
                                autoplay_html = f'''
                                <audio controls autoplay style="width: 100%; margin: 10px 0;">
                                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                                    Your browser does not support the audio element.
                                </audio>
                                <script>
                                    // Set up faster speech and auto-play
                                    setTimeout(function() {{
                                        var audio = document.querySelector('audio[autoplay]');
                                        if (audio) {{
                                            audio.playbackRate = 1.3; // 30% faster speech
                                            audio.play().catch(function(error) {{
                                                console.log('Auto-play blocked by browser:', error);
                                            }});
                                        }}
                                    }}, 100);
                                    
                                    // Clear animation when audio ends
                                    setTimeout(function() {{
                                        var animationElements = document.querySelectorAll('.voice-waveform-speaking');
                                        animationElements.forEach(function(el) {{
                                            el.style.display = 'none';
                                        }});
                                    }}, {len(response_text) * 50});  // Estimate duration
                                </script>
                                '''
                                st.markdown(autoplay_html, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.warning(f"Voice generation failed: {str(e)}")
                                st.write(f"**Text Response:** {response_text}")
                        else:
                            st.write(f"**Text Response:** {response_text}")
                    
                    # Add to conversation history
                    st.session_state.voice_conversation.append({
                        'query': user_query,
                        'response': response_text,
                        'timestamp': pd.Timestamp.now()
                    })
                    
                except Exception as e:
                    st.error(f"❌ Voice processing error: {str(e)}")
                    st.write("Please try recording again or check your microphone permissions.")
        
    except ImportError:
        st.error("❌ Voice recording not available. Please install streamlit-mic-recorder:")
        st.code("pip install streamlit-mic-recorder", language="bash")
    
    # Conversation history
    if st.session_state.voice_conversation:
        st.markdown("---")
        st.subheader("💬 Voice Conversation History")
        
        for i, conv in enumerate(reversed(st.session_state.voice_conversation[-5:])):
            with st.expander(f"🗣️ Query {len(st.session_state.voice_conversation)-i}: {conv['query']}", expanded=False):
                st.write(f"**Question:** {conv['query']}")
                st.write(f"**Response:** {conv['response']}")
                st.caption(f"Time: {conv['timestamp'].strftime('%H:%M:%S')}")
        
        # Clear history
        if st.button("🗑️ Clear Conversation History"):
            st.session_state.voice_conversation = []
            st.rerun()

def generate_voice_friendly_response(query, gpt_result, result):
    """Generate a structured conversational response suitable for voice output"""
    explanation = gpt_result.get('explanation', 'Analysis completed')
    
    # Start with confirmation of the issue/question
    confirmation = f"I understand you're asking about {query.lower()}. Let me analyze this for you."
    
    # Generate detailed findings
    if isinstance(result, pd.DataFrame):
        row_count = len(result)
        if row_count == 0:
            findings = "I've completed the analysis, but found no matching data in your dataset for this specific query. This could mean the criteria didn't match any records, or the data might be structured differently than expected."
            summary = "My recommendation is to try rephrasing your question or checking if the data contains the fields you're looking for."
        elif row_count == 1:
            # Get specific details for single result
            findings = f"I found exactly one result that matches your criteria. {explanation}."
            if 'percentage' in query.lower() or 'rate' in query.lower():
                if result.columns.tolist():
                    first_value = result.iloc[0, 1] if len(result.columns) > 1 else result.iloc[0, 0]
                    findings += f" The value is {first_value:.1f}%." if isinstance(first_value, (int, float)) else f" The result shows {first_value}."
            summary = "This gives you a clear, focused answer to your specific question."
        else:
            findings = f"I found {row_count} results for your analysis. {explanation}."
            
            # Add specific insights based on query type
            if 'publisher' in query.lower():
                findings += f" This covers {row_count} different publishers in your data."
            elif 'buyer' in query.lower():
                findings += f" This shows performance across {row_count} different buyers."
            elif 'ad misled' in query.lower():
                findings += f" I've identified {row_count} entries related to ad misled situations."
            
            # Try to get top performers
            if len(result.columns) >= 2 and result.iloc[:, 1].dtype in ['int64', 'float64']:
                top_performer = result.iloc[0, 0]
                top_value = result.iloc[0, 1]
                findings += f" The top performer is {top_performer} with {top_value:.1f}."
                
                if row_count > 1:
                    bottom_performer = result.iloc[-1, 0]
                    bottom_value = result.iloc[-1, 1]
                    findings += f" The lowest is {bottom_performer} with {bottom_value:.1f}."
            
            summary = f"You can see the complete breakdown in the table. This analysis helps identify patterns and performance gaps across your {row_count} results."
    
    elif isinstance(result, (int, float)):
        findings = f"The answer to your question is {result:,.2f}. {explanation}."
        if result > 50:
            summary = "This indicates a relatively high value that may require attention."
        elif result > 20:
            summary = "This shows a moderate level that's worth monitoring."
        else:
            summary = "This is a relatively low value, which could be either good or concerning depending on the metric."
    
    else:
        findings = f"I've processed your question about {query}. {explanation}."
        summary = "The detailed results are shown in the analysis below."
    
    # Combine all parts for a complete voice response
    full_response = f"{confirmation} {findings} {summary}"
    
    return full_response

def generate_elevenlabs_response(text, api_key, speed=1.2):
    """Generate voice response using ElevenLabs with professional business delivery"""
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
        
        client = ElevenLabs(api_key=api_key)
        
        # Generate audio with optimized settings for business analysis
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel - Professional Female (Free Tier)
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.7,        # Professional and clear
                similarity_boost=0.8, # Strong voice character
                style=0.4,           # Moderate expressiveness
                use_speaker_boost=True, # Enhanced clarity
                speed=speed          # Speech speed (0.7-1.2)
            ),
            output_format="mp3_22050_32"
        )
        
        # Convert generator to bytes
        audio_bytes = b''.join(audio_generator)
        return audio_bytes
        
    except ImportError:
        raise Exception("ElevenLabs library not installed. Please install: pip install elevenlabs")
    except Exception as e:
        raise Exception(f"ElevenLabs API error: {str(e)}")

class PerformanceDiagnostic:
    """Main diagnostic engine for performance marketing analysis"""
    
    def __init__(self):
        self.data = None
        self.alerts = []
        self.combinations_analysis = {}
        
    def load_data(self, uploaded_file):
        """Load and validate data from uploaded Excel file"""
        try:
            # Read Excel file
            self.data = pd.read_excel(uploaded_file)
            
            # Standardize column names (remove spaces, make uppercase)
            self.data.columns = [col.strip().upper().replace(' ', '_') for col in self.data.columns]
            
            # Validate required columns
            required_columns = [
                'PUBLISHER', 'BUYER', 'TARGET', 'SALE', 'CUSTOMER_INTENT',
                'REACHED_AGENT', 'AD_MISLED', 'BILLABLE', 'DURATION',
                'IVR', 'OBJECTION_WITH_NO_REBUTTAL'
            ]
            
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Data cleaning and standardization
            self._clean_data()
            
            st.success(f"✅ Data loaded successfully: {len(self.data):,} records")
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def _clean_data(self):
        """Clean and standardize the data"""
        # Standardize Yes/No fields
        yes_no_fields = ['SALE', 'REACHED_AGENT', 'AD_MISLED', 'BILLABLE', 'IVR', 'OBJECTION_WITH_NO_REBUTTAL']
        for field in yes_no_fields:
            if field in self.data.columns:
                self.data[field] = self.data[field].astype(str).str.title()
                self.data[field] = self.data[field].replace({'1': 'Yes', '0': 'No', 'True': 'Yes', 'False': 'No'})
        
        # Clean customer intent
        if 'CUSTOMER_INTENT' in self.data.columns:
            self.data['CUSTOMER_INTENT'] = self.data['CUSTOMER_INTENT'].astype(str).str.title()
        
        # Convert duration to numeric (handle various formats) - more robust cleaning
        if 'DURATION' in self.data.columns:
            # First try to clean string-based durations
            duration_series = self.data['DURATION'].astype(str)
            # Remove any dollar signs, commas, and other non-numeric characters except decimal points
            duration_series = duration_series.str.replace(r'[^\d\.]', '', regex=True)
            self.data['DURATION'] = pd.to_numeric(duration_series, errors='coerce').fillna(0)
        
        # Convert revenue to numeric if it exists
        if 'REVENUE' in self.data.columns:
            # Clean revenue data
            revenue_series = self.data['REVENUE'].astype(str)
            # Remove dollar signs, commas, and other non-numeric characters except decimal points
            revenue_series = revenue_series.str.replace(r'[^\d\.]', '', regex=True)
            self.data['REVENUE'] = pd.to_numeric(revenue_series, errors='coerce').fillna(0)
        
        # Add derived fields
        self.data['IS_STRONG_LEAD'] = self.data['CUSTOMER_INTENT'].isin(['Level 2', 'Level 3'])
        self.data['IS_SALE'] = self.data['SALE'] == 'Yes'
        self.data['HAS_QUOTE'] = self.data.get('QUOTE', 'No') == 'Yes'  # If quote column exists
    
    def generate_critical_alerts(self):
        """Generate critical alerts requiring immediate attention"""
        self.alerts = []
        
        if self.data is None:
            return
        
        # 1. Ad Misled Issues
        ad_misled = self.data[self.data['AD_MISLED'] == 'Yes']
        if len(ad_misled) > 0:
            ad_misled_by_publisher = ad_misled.groupby('PUBLISHER').size().sort_values(ascending=False)
            total_ad_misled = len(ad_misled)
            self.alerts.append({
                'type': 'CRITICAL',
                'category': 'Ad Misled',
                'count': total_ad_misled,
                'message': f"🚨 {total_ad_misled} Ad Misled incidents require immediate attention",
                'details': ad_misled_by_publisher.to_dict(),
                'data': ad_misled
            })
        
        # 2. Agent Availability Crisis
        no_agent = self.data[self.data['REACHED_AGENT'] == 'No']
        if len(no_agent) > 0:
            no_agent_by_buyer = no_agent.groupby('BUYER').size().sort_values(ascending=False)
            agent_availability_rate = self.data.groupby('BUYER')['REACHED_AGENT'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).sort_values()
            
            critical_buyers = agent_availability_rate[agent_availability_rate < 80]  # Less than 80% availability
            if len(critical_buyers) > 0:
                self.alerts.append({
                    'type': 'CRITICAL',
                    'category': 'Agent Availability',
                    'count': len(no_agent),
                    'message': f"🚨 {len(critical_buyers)} Buyers have critical agent availability issues (<80%)",
                    'details': critical_buyers.to_dict(),
                    'data': no_agent
                })
        
        # 3. High-Value Lead Waste
        high_value_leads = self.data[self.data['IS_STRONG_LEAD'] == True]
        wasted_high_value = high_value_leads[high_value_leads['IS_SALE'] == False]
        
        if len(wasted_high_value) > 0:
            waste_rate = len(wasted_high_value) / len(high_value_leads) * 100
            level_3_waste = wasted_high_value[wasted_high_value['CUSTOMER_INTENT'] == 'Level 3']
            
            self.alerts.append({
                'type': 'HIGH',
                'category': 'High-Value Lead Waste',
                'count': len(wasted_high_value),
                'message': f"⚠️ {len(wasted_high_value)} high-value leads wasted ({waste_rate:.1f}% waste rate)",
                'details': {
                    'Level 3 wasted': len(level_3_waste),
                    'Total high-value wasted': len(wasted_high_value)
                },
                'data': wasted_high_value
            })
        
        # 4. Poor Agent Performance
        poor_rebuttals = self.data[self.data['OBJECTION_WITH_NO_REBUTTAL'] == 'Yes']
        if len(poor_rebuttals) > 0:
            rebuttal_by_buyer = poor_rebuttals.groupby('BUYER').size().sort_values(ascending=False)
            self.alerts.append({
                'type': 'MEDIUM',
                'category': 'Poor Agent Performance', 
                'count': len(poor_rebuttals),
                'message': f"⚠️ {len(poor_rebuttals)} instances of poor objection handling",
                'details': rebuttal_by_buyer.to_dict(),
                'data': poor_rebuttals
            })
    
    def analyze_combinations(self):
        """Analyze Publisher-Buyer-Target combinations"""
        if self.data is None:
            return
        
        # Ensure numeric columns are properly converted
        duration_clean = pd.to_numeric(self.data['DURATION'], errors='coerce').fillna(0)
        
        # Publisher-Buyer combinations
        pub_buyer = self.data.groupby(['PUBLISHER', 'BUYER']).agg({
            'IS_SALE': ['count', 'sum', 'mean'],
            'IS_STRONG_LEAD': 'mean',
            'AD_MISLED': lambda x: (x == 'Yes').sum(),
            'REACHED_AGENT': lambda x: (x == 'Yes').mean(),
            'BILLABLE': lambda x: (x == 'Yes').mean()
        }).round(3)
        
        # Add duration separately with safe conversion
        duration_by_combo = self.data.groupby(['PUBLISHER', 'BUYER']).apply(
            lambda x: pd.to_numeric(x['DURATION'], errors='coerce').mean()
        ).fillna(0)
        
        # Flatten column names
        pub_buyer.columns = ['Total_Leads', 'Total_Sales', 'Conversion_Rate', 
                           'Strong_Lead_Rate', 'Ad_Misled_Count',
                           'Agent_Availability_Rate', 'Billable_Rate']
        
        # Add duration column
        pub_buyer['Avg_Duration'] = duration_by_combo
        
        # Publisher-Buyer-Target combinations
        pub_buyer_target = self.data.groupby(['PUBLISHER', 'BUYER', 'TARGET']).agg({
            'IS_SALE': ['count', 'sum', 'mean'],
            'IS_STRONG_LEAD': 'mean',
            'CUSTOMER_INTENT': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(3)
        
        # Add duration separately for target combinations
        duration_by_target_combo = self.data.groupby(['PUBLISHER', 'BUYER', 'TARGET']).apply(
            lambda x: pd.to_numeric(x['DURATION'], errors='coerce').mean()
        ).fillna(0)
        
        pub_buyer_target.columns = ['Total_Leads', 'Total_Sales', 'Conversion_Rate',
                                  'Strong_Lead_Rate', 'Most_Common_Intent']
        
        # Add duration column
        pub_buyer_target['Avg_Duration'] = duration_by_target_combo
        
        self.combinations_analysis = {
            'publisher_buyer': pub_buyer.reset_index(),
            'publisher_buyer_target': pub_buyer_target.reset_index()
        }
    
    def analyze_lead_quality(self):
        """Analyze lead quality by publisher"""
        publisher_quality = self.data.groupby('PUBLISHER').agg({
            'IS_SALE': ['count', 'sum', 'mean'],
            'IS_STRONG_LEAD': 'mean',
            'CUSTOMER_INTENT': lambda x: x.value_counts().to_dict(),
            'BILLABLE': lambda x: (x == 'Yes').mean(),
            'AD_MISLED': lambda x: (x == 'Yes').sum(),
            'IVR': lambda x: (x == 'Yes').mean()
        }).round(3)
        
        # Add duration separately with safe conversion
        duration_by_publisher = self.data.groupby('PUBLISHER').apply(
            lambda x: pd.to_numeric(x['DURATION'], errors='coerce').mean()
        ).fillna(0)
        
        publisher_quality.columns = ['Total_Leads', 'Total_Sales', 'Conversion_Rate',
                                   'Strong_Lead_Rate', 'Intent_Distribution', 'Billable_Rate',
                                   'Ad_Misled_Count', 'IVR_Rate']
        
        # Add duration column
        publisher_quality['Avg_Duration'] = duration_by_publisher
        
        return publisher_quality.reset_index()
    
    def analyze_sales_execution(self):
        """Analyze sales execution by buyer"""
        buyer_execution = self.data.groupby('BUYER').agg({
            'IS_SALE': ['count', 'sum', 'mean'],
            'REACHED_AGENT': lambda x: (x == 'Yes').mean(),
            'OBJECTION_WITH_NO_REBUTTAL': lambda x: (x == 'Yes').sum()
        }).round(3)
        
        # Add duration separately with safe conversion
        duration_by_buyer = self.data.groupby('BUYER').apply(
            lambda x: pd.to_numeric(x['DURATION'], errors='coerce').mean()
        ).fillna(0)
        
        buyer_execution.columns = ['Total_Leads', 'Total_Sales', 'Conversion_Rate',
                                 'Agent_Availability_Rate', 'Poor_Rebuttal_Count']
        
        # Add duration column
        buyer_execution['Avg_Duration'] = duration_by_buyer
        
        # Add Level 3 conversion analysis
        level_3_data = self.data[self.data['CUSTOMER_INTENT'] == 'Level 3']
        if len(level_3_data) > 0:
            level_3_conversion = level_3_data.groupby('BUYER')['IS_SALE'].mean()
            buyer_execution['Level_3_Conversion_Rate'] = buyer_execution.index.map(level_3_conversion).fillna(0)
        
        return buyer_execution.reset_index()

def create_alert_dashboard(diagnostic):
    """Create the alert dashboard"""
    st.header("🚨 Critical Alert Dashboard")
    
    if not diagnostic.alerts:
        st.success("✅ No critical alerts at this time")
        return
    
    # Alert summary
    critical_count = len([a for a in diagnostic.alerts if a['type'] == 'CRITICAL'])
    high_count = len([a for a in diagnostic.alerts if a['type'] == 'HIGH'])
    medium_count = len([a for a in diagnostic.alerts if a['type'] == 'MEDIUM'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Critical Alerts", critical_count, delta=None, delta_color="inverse")
    with col2:
        st.metric("High Priority", high_count)
    with col3:
        st.metric("Medium Priority", medium_count)
    
    # Display alerts
    for alert in diagnostic.alerts:
        if alert['type'] == 'CRITICAL':
            st.error(alert['message'])
        elif alert['type'] == 'HIGH':
            st.warning(alert['message'])
        else:
            st.info(alert['message'])
        
        # Show details in expander
        with st.expander(f"Details - {alert['category']}"):
            if alert['category'] == 'Ad Misled':
                st.write("**Ad Misled by Publisher:**")
                for publisher, count in alert['details'].items():
                    pct = count / alert['count'] * 100
                    st.write(f"- {publisher}: {count} incidents ({pct:.1f}%)")
            
            elif alert['category'] == 'Agent Availability':
                st.write("**Agent Availability Rates by Buyer:**")
                for buyer, rate in alert['details'].items():
                    st.write(f"- {buyer}: {rate:.1f}% availability")
            
            else:
                st.write("**Details:**")
                for key, value in alert['details'].items():
                    st.write(f"- {key}: {value}")

def create_combination_analysis(diagnostic):
    """Create combination analysis section"""
    st.header("📊 Combination Performance Analysis")
    
    if 'publisher_buyer' not in diagnostic.combinations_analysis:
        st.warning("No combination analysis available. Please ensure data is loaded.")
        return
    
    tab1, tab2 = st.tabs(["Publisher-Buyer Combinations", "Publisher-Buyer-Target Combinations"])
    
    with tab1:
        st.subheader("Publisher × Buyer Performance Matrix")
        
        pb_data = diagnostic.combinations_analysis['publisher_buyer']
        
        # Performance heatmap
        if len(pb_data) > 0:
            pivot_data = pb_data.pivot_table(
                index='PUBLISHER', 
                columns='BUYER', 
                values='Conversion_Rate', 
                fill_value=0
            )
            
            fig = px.imshow(
                pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                color_continuous_scale='RdYlGn',
                aspect='auto',
                title='Conversion Rate Heatmap: Publisher × Buyer'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top and bottom performers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Performing Combinations")
                top_combinations = pb_data.nlargest(5, 'Conversion_Rate')[
                    ['PUBLISHER', 'BUYER', 'Conversion_Rate', 'Total_Leads', 'Total_Sales']
                ]
                st.dataframe(top_combinations)
            
            with col2:
                st.subheader("⚠️ Underperforming Combinations")
                bottom_combinations = pb_data.nsmallest(5, 'Conversion_Rate')[
                    ['PUBLISHER', 'BUYER', 'Conversion_Rate', 'Total_Leads', 'Total_Sales']
                ]
                st.dataframe(bottom_combinations)
    
    with tab2:
        st.subheader("Publisher × Buyer × Target Analysis")
        
        pbt_data = diagnostic.combinations_analysis['publisher_buyer_target']
        
        if len(pbt_data) > 0:
            # Filter for combinations with meaningful volume
            significant_combos = pbt_data[pbt_data['Total_Leads'] >= 10]
            
            if len(significant_combos) > 0:
                # Scatter plot: Leads vs Conversion Rate
                fig = px.scatter(
                    significant_combos,
                    x='Total_Leads',
                    y='Conversion_Rate',
                    size='Total_Sales',
                    color='Strong_Lead_Rate',
                    hover_data=['PUBLISHER', 'BUYER', 'TARGET'],
                    title='Lead Volume vs Conversion Rate (bubble size = total sales)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.subheader("Detailed Combination Metrics")
                st.dataframe(significant_combos)

def create_lead_quality_analysis(diagnostic):
    """Create lead quality analysis section"""
    st.header("🎯 Lead Quality Analysis")
    
    if diagnostic.data is None:
        return
    
    lead_quality = diagnostic.analyze_lead_quality()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Publisher Lead Quality Rankings")
        
        # Sort by conversion rate
        quality_ranking = lead_quality.sort_values('Conversion_Rate', ascending=False)
        
        # Create ranking chart
        fig = px.bar(
            quality_ranking.head(10),
            x='Conversion_Rate',
            y='PUBLISHER',
            orientation='h',
            title='Top 10 Publishers by Conversion Rate',
            color='Strong_Lead_Rate',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Lead Quality Metrics")
        
        # Customer intent distribution
        intent_dist = diagnostic.data['CUSTOMER_INTENT'].value_counts()
        
        fig = px.pie(
            values=intent_dist.values,
            names=intent_dist.index,
            title='Customer Intent Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed quality metrics table
    st.subheader("Publisher Quality Scorecard")
    
    # Add quality scores
    lead_quality['Quality_Score'] = (
        lead_quality['Conversion_Rate'] * 0.4 +
        lead_quality['Strong_Lead_Rate'] * 0.3 +
        lead_quality['Billable_Rate'] * 0.2 +
        (1 - lead_quality['IVR_Rate']) * 0.1
    ).round(3)
    
    # Sort by quality score
    quality_scorecard = lead_quality.sort_values('Quality_Score', ascending=False)
    
    st.dataframe(quality_scorecard)

def create_sales_execution_analysis(diagnostic):
    """Create sales execution analysis section"""
    st.header("💼 Sales Execution Analysis")
    
    if diagnostic.data is None:
        return
    
    sales_execution = diagnostic.analyze_sales_execution()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Buyer Performance Rankings")
        
        execution_ranking = sales_execution.sort_values('Conversion_Rate', ascending=False)
        
        fig = px.bar(
            execution_ranking,
            x='Conversion_Rate',
            y='BUYER',
            orientation='h',
            title='Buyer Conversion Rate Performance',
            color='Agent_Availability_Rate',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Agent Availability Analysis")
        
        # Agent availability by buyer
        fig = px.scatter(
            sales_execution,
            x='Agent_Availability_Rate',
            y='Conversion_Rate',
            size='Total_Leads',
            hover_name='BUYER',
            title='Agent Availability vs Conversion Rate'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Level 3 conversion analysis
    if 'Level_3_Conversion_Rate' in sales_execution.columns:
        st.subheader("Level 3 Lead Conversion Analysis")
        
        level_3_performance = sales_execution[sales_execution['Level_3_Conversion_Rate'] > 0].sort_values(
            'Level_3_Conversion_Rate', ascending=False
        )
        
        fig = px.bar(
            level_3_performance,
            x='Level_3_Conversion_Rate',
            y='BUYER',
            orientation='h',
            title='Level 3 Lead Conversion Rate by Buyer',
            color='Level_3_Conversion_Rate',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Training needs identification
    st.subheader("Training Needs Assessment")
    
    training_needs = sales_execution[
        (sales_execution['Poor_Rebuttal_Count'] > 0) | 
        (sales_execution['Agent_Availability_Rate'] < 0.8)
    ].sort_values('Poor_Rebuttal_Count', ascending=False)
    
    if len(training_needs) > 0:
        st.warning(f"⚠️ {len(training_needs)} buyers require training intervention")
        st.dataframe(training_needs[['BUYER', 'Poor_Rebuttal_Count', 'Agent_Availability_Rate', 'Conversion_Rate']])
    else:
        st.success("✅ No immediate training needs identified")

def get_section_suggestions(section: str) -> list:
    """Get suggested questions for a specific section."""
    section_lower = section.lower()
    
    if 'ad misled' in section_lower or 'crisis' in section_lower:
        return [
            "List publishers with ad misled calls and percentages",
            "Show worst performing publishers for ad misled",
            "Calculate total ad misled crisis impact",
            "Break down ad misled by publisher and buyer",
            "Show ad misled trend analysis"
        ]
    elif 'agent' in section_lower or 'availability' in section_lower:
        return [
            "Show agent availability rates by buyer",
            "Calculate average agent availability percentage",
            "Identify buyers with low agent availability",
            "Show agent performance vs conversion rates",
            "Analyze peak hours for agent availability"
        ]
    elif 'lead quality' in section_lower:
        return [
            "Analyze lead quality scores by publisher",
            "Show conversion rates by lead quality",
            "Identify best performing lead sources",
            "Calculate quality score distribution",
            "Show lead qualification breakdown"
        ]
    elif 'sales execution' in section_lower:
        return [
            "Show sales execution performance by buyer",
            "Calculate conversion rates by sales team",
            "Identify training needs for buyers",
            "Show rebuttal effectiveness analysis",
            "Analyze sales performance trends"
        ]
    elif 'combination' in section_lower:
        return [
            "Show top publisher-buyer combinations",
            "Analyze worst performing combinations",
            "Calculate ROI by combination",
            "Show combination performance matrix",
            "Identify optimization opportunities"
        ]
    elif 'customer intent' in section_lower:
        return [
            "Show customer intent distribution",
            "Analyze conversion by intent level",
            "Calculate intent detection accuracy",
            "Show intent vs sales correlation",
            "Identify intent improvement areas"
        ]
    elif 'billable' in section_lower:
        return [
            "Show billable leads percentage by publisher",
            "Calculate billable vs non-billable conversion",
            "Analyze billable lead quality",
            "Show billable leads revenue impact",
            "Identify billing optimization opportunities"
        ]
    else:
        return [
            "Show me a detailed breakdown",
            "What are the key metrics?",
            "Create a performance visualization",
            "Show top and bottom performers",
            "Identify improvement opportunities"
        ]

def main():
    """Main application"""
    st.title("📊 Performance Marketing Diagnostic Tool")
    st.markdown("**Comprehensive analysis of Publisher-Buyer-Target combinations**")
    
    # Initialize diagnostic engine
    if 'diagnostic' not in st.session_state:
        st.session_state.diagnostic = PerformanceDiagnostic()
    
    diagnostic = st.session_state.diagnostic
    
    # Sidebar for data upload and navigation
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Excel file",
            type=['xlsx', 'xls'],
            help="Upload your performance marketing data"
        )
        
        if uploaded_file is not None:
            if st.button("Load Data"):
                if diagnostic.load_data(uploaded_file):
                    diagnostic.generate_critical_alerts()
                    diagnostic.analyze_combinations()
                    st.success("Data analysis complete!")
        
        st.markdown("---")
        
        # Navigation
        st.header("🧭 Navigation")
        nav_options = [
            "Dashboard Overview",
            "Alert Center", 
            "Combination Analysis",
            "Lead Quality Analysis",
            "Sales Execution Analysis",
            "Advanced Analysis",
            "Executive Summary",
            "Talk to Your Data",
            "🎤 Voice Test Page"
        ]
        
        selected_page = st.selectbox("Go to:", nav_options)
        
        # Section Analysis Feature
        if diagnostic.data is not None:
            st.markdown("---")
            st.header("🎯 Section Analysis")
            
            # Detect sections from the navigation options and data
            section_options = [
                "AD MISLED CRISIS", 
                "AGENT AVAILABILITY", 
                "LEAD QUALITY",
                "SALES EXECUTION",
                "COMBINATION PERFORMANCE",
                "CUSTOMER INTENT ANALYSIS",
                "BILLABLE LEADS ANALYSIS"
            ]
            
            selected_section = st.selectbox(
                "Select section to analyze:",
                ["Select a section..."] + section_options,
                key="section_selector"
            )
            
            if selected_section != "Select a section...":
                st.write(f"**Analyzing: {selected_section}**")
                
                # Section-specific query input
                section_query = st.text_input(
                    "Ask about this section:",
                    placeholder=f"e.g., Show me detailed breakdown of {selected_section.lower()}",
                    key="section_query_input"
                )
                
                if st.button("🔍 Analyze Section", type="primary"):
                    if section_query:
                        # Execute analysis immediately
                        st.session_state.section_analysis_query = f"Focusing on {selected_section}: {section_query}"
                        st.session_state.section_analysis_active = True
                        st.rerun()
                    else:
                        st.warning("Please enter a question about the selected section.")
                
                # Quick section suggestions
                st.write("**Quick suggestions:**")
                section_suggestions = get_section_suggestions(selected_section)
                for i, suggestion in enumerate(section_suggestions):
                    if st.button(suggestion, key=f"section_suggestion_{i}", use_container_width=True):
                        # Execute analysis immediately
                        st.session_state.section_analysis_query = f"Focusing on {selected_section}: {suggestion}"
                        st.session_state.section_analysis_active = True
                        st.rerun()
    
    # Main content area
    if diagnostic.data is None:
        st.info("👆 Please upload your performance marketing data to begin analysis")
        
        # Show sample data format
        st.subheader("Required Data Format")
        st.markdown("""
        Your Excel file should contain the following columns:
        - **PUBLISHER**: Lead source publisher
        - **BUYER**: Lead buyer/company
        - **TARGET**: Target division/vertical
        - **SALE**: Yes/No - Was sale made
        - **CUSTOMER_INTENT**: Level 1/2/3, Negative Intent, Not Detected
        - **REACHED_AGENT**: Yes/No - Did lead reach an agent
        - **AD_MISLED**: Yes/No - Was customer misled by ad
        - **BILLABLE**: Yes/No - Is lead billable
        - **DURATION**: Call duration in seconds
        - **IVR**: Yes/No - Did call go to IVR
        - **OBJECTION_WITH_NO_REBUTTAL**: Yes/No - Poor objection handling
        """)
        
    else:
        # Check for active section analysis
        if st.session_state.get('section_analysis_active', False):
            st.header("🎯 Section Analysis Results")
            
            query = st.session_state.get('section_analysis_query', '')
            if query:
                st.subheader(f"Query: {query}")
                
                # Initialize OpenAI client for analysis
                client = init_openai()
                if client:
                    with st.spinner("🤖 Analyzing your section query..."):
                        # Get analysis from GPT-4o
                        gpt_result, error = query_with_gpt4o(query, diagnostic.data, client)
                        
                        if error:
                            st.error(f"❌ {error}")
                        elif gpt_result:
                            # Show the analysis
                            st.write("**🎯 Analysis Results:**")
                            st.write(f"**Explanation:** {gpt_result['explanation']}")
                            
                            # Execute the code
                            result, exec_error = execute_code_safely(gpt_result['code'], diagnostic.data)
                            
                            if exec_error:
                                st.error(f"❌ Execution Error: {exec_error}")
                            else:
                                # Show results
                                if isinstance(result, (int, float)):
                                    st.metric("Result", f"{result:,.2f}")
                                elif isinstance(result, pd.DataFrame):
                                    st.subheader("📊 Data Results")
                                    st.dataframe(result, use_container_width=True)
                                    
                                    # Download option
                                    csv = result.to_csv(index=True)
                                    st.download_button(
                                        "📥 Download Results as CSV",
                                        csv,
                                        f"section_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        "text/csv"
                                    )
                                elif isinstance(result, pd.Series):
                                    st.dataframe(result.to_frame(), use_container_width=True)
                                else:
                                    st.write(result)
                                
                                # Show visualization
                                viz = create_visualization(result, gpt_result.get('visualization_type', 'bar'), query)
                                if viz:
                                    st.subheader("📈 Visualization")
                                    st.plotly_chart(viz, use_container_width=True)
                                
                                # Show generated code
                                with st.expander("🔍 View Generated Code", expanded=False):
                                    st.code(gpt_result['code'], language='python')
                else:
                    st.warning("⚠️ OpenAI API key not configured. Please add your API key to .env file.")
                
                # Clear analysis and return to normal view
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Run Another Section Analysis", type="primary"):
                        st.session_state.section_analysis_active = False
                        st.session_state.section_analysis_query = ""
                        st.rerun()
                
                with col2:
                    if st.button("📊 Return to Dashboard", type="secondary"):
                        st.session_state.section_analysis_active = False
                        st.session_state.section_analysis_query = ""
                        st.rerun()
                
            return  # Don't show other pages when section analysis is active
        
        # Show selected page
        if selected_page == "Dashboard Overview":
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_leads = len(diagnostic.data)
            total_sales = diagnostic.data['IS_SALE'].sum()
            conversion_rate = (total_sales / total_leads * 100) if total_leads > 0 else 0
            strong_lead_pct = (diagnostic.data['IS_STRONG_LEAD'].mean() * 100) if total_leads > 0 else 0
            
            with col1:
                st.metric("Total Leads", f"{total_leads:,}")
            with col2:
                st.metric("Total Sales", f"{total_sales:,}")
            with col3:
                st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
            with col4:
                st.metric("Strong Leads", f"{strong_lead_pct:.1f}%")
            
            # Quick alerts summary
            if diagnostic.alerts:
                st.subheader("🚨 Quick Alert Summary")
                critical_alerts = [a for a in diagnostic.alerts if a['type'] == 'CRITICAL']
                if critical_alerts:
                    for alert in critical_alerts[:3]:  # Show top 3 critical
                        st.error(alert['message'])
        
        elif selected_page == "Alert Center":
            create_alert_dashboard(diagnostic)
            
        elif selected_page == "Combination Analysis":
            create_combination_analysis(diagnostic)
            
        elif selected_page == "Lead Quality Analysis":
            create_lead_quality_analysis(diagnostic)
            
        elif selected_page == "Sales Execution Analysis":
            create_sales_execution_analysis(diagnostic)
            
        elif selected_page == "Advanced Analysis":
            create_advanced_analysis_page(diagnostic)
            
        elif selected_page == "Executive Summary":
            create_executive_summary_page(diagnostic)
            
        elif selected_page == "Talk to Your Data":
            create_talk_to_data_page(diagnostic)
            
        elif selected_page == "🎤 Voice Test Page":
            create_voice_test_page(diagnostic)

def create_listening_animation():
    """Create subtle listening waveform animation"""
    
    listening_css = """
    <style>
    .voice-waveform-container {
        position: relative;
        width: 100%;
        height: 80px;
        margin: 10px 0;
        overflow: hidden;
        border-radius: 10px;
        background: linear-gradient(90deg, rgba(0, 200, 81, 0.1) 0%, rgba(0, 126, 51, 0.2) 50%, rgba(0, 200, 81, 0.1) 100%);
        border: 1px solid rgba(0, 200, 81, 0.3);
    }
    
    .listening-wave {
        position: absolute;
        top: 50%;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #00c851 25%, #00ff88 50%, #00c851 75%, transparent 100%);
        transform: translateY(-50%);
        animation: listening-flow 2s ease-in-out infinite;
    }
    
    .listening-wave::before {
        content: '';
        position: absolute;
        top: -10px;
        left: 0;
        width: 100%;
        height: 20px;
        background: linear-gradient(90deg, transparent 0%, rgba(0, 200, 81, 0.3) 25%, rgba(0, 255, 136, 0.5) 50%, rgba(0, 200, 81, 0.3) 75%, transparent 100%);
        animation: listening-flow 2s ease-in-out infinite;
        filter: blur(5px);
    }
    
    @keyframes listening-flow {
        0%, 100% { 
            transform: translateY(-50%) scaleY(1) scaleX(1);
            opacity: 0.8;
        }
        25% { 
            transform: translateY(-60%) scaleY(2) scaleX(1.1);
            opacity: 1;
        }
        75% { 
            transform: translateY(-40%) scaleY(2) scaleX(0.9);
            opacity: 1;
        }
    }
    
    .listening-status {
        position: absolute;
        top: 50%;
        left: 15px;
        transform: translateY(-50%);
        font-size: 14px;
        color: #00c851;
        font-weight: 500;
        z-index: 10;
    }
    
    .listening-icon {
        position: absolute;
        top: 50%;
        right: 15px;
        transform: translateY(-50%);
        font-size: 20px;
        animation: pulse 2s ease-in-out infinite;
        z-index: 10;
    }
    
    @keyframes pulse {
        0%, 100% { transform: translateY(-50%) scale(1); opacity: 0.8; }
        50% { transform: translateY(-50%) scale(1.1); opacity: 1; }
    }
    </style>
    
    <div class="voice-waveform-container">
        <div class="listening-wave"></div>
        <div class="listening-status">🎤 Listening...</div>
        <div class="listening-icon">🟢</div>
    </div>
    """
    
    return st.markdown(listening_css, unsafe_allow_html=True)

def create_speaking_animation():
    """Create subtle speaking waveform animation"""
    
    speaking_css = """
    <style>
    .voice-waveform-speaking {
        position: relative;
        width: 100%;
        height: 80px;
        margin: 10px 0;
        overflow: hidden;
        border-radius: 10px;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.2) 50%, rgba(102, 126, 234, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .speaking-waveform {
        position: absolute;
        top: 50%;
        left: 0;
        width: 100%;
        height: 100%;
        transform: translateY(-50%);
    }
    
    .speaking-wave-line {
        position: absolute;
        top: 50%;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, #667eea 20%, #764ba2 50%, #667eea 80%, transparent 100%);
        transform: translateY(-50%);
        animation: speaking-flow 1.5s ease-in-out infinite;
    }
    
    .speaking-wave-line::before {
        content: '';
        position: absolute;
        top: -15px;
        left: 0;
        width: 100%;
        height: 30px;
        background: linear-gradient(90deg, transparent 0%, rgba(102, 126, 234, 0.2) 20%, rgba(118, 75, 162, 0.4) 50%, rgba(102, 126, 234, 0.2) 80%, transparent 100%);
        animation: speaking-flow 1.5s ease-in-out infinite;
        filter: blur(8px);
    }
    
    @keyframes speaking-flow {
        0%, 100% { 
            transform: translateY(-50%) scaleY(1) scaleX(1);
            opacity: 0.9;
        }
        20% { 
            transform: translateY(-45%) scaleY(1.5) scaleX(1.2);
            opacity: 1;
        }
        40% { 
            transform: translateY(-55%) scaleY(2.2) scaleX(0.8);
            opacity: 1;
        }
        60% { 
            transform: translateY(-48%) scaleY(1.8) scaleX(1.1);
            opacity: 1;
        }
        80% { 
            transform: translateY(-52%) scaleY(2.5) scaleX(0.9);
            opacity: 1;
        }
    }
    
    .speaking-status {
        position: absolute;
        top: 50%;
        left: 15px;
        transform: translateY(-50%);
        font-size: 14px;
        color: #667eea;
        font-weight: 500;
        z-index: 10;
    }
    
    .speaking-icon {
        position: absolute;
        top: 50%;
        right: 15px;
        transform: translateY(-50%);
        font-size: 20px;
        animation: speaking-pulse 1.5s ease-in-out infinite;
        z-index: 10;
    }
    
    @keyframes speaking-pulse {
        0%, 100% { transform: translateY(-50%) scale(1); opacity: 0.8; }
        33% { transform: translateY(-50%) scale(1.1); opacity: 1; }
        66% { transform: translateY(-50%) scale(0.9); opacity: 1; }
    }
    </style>
    
    <div class="voice-waveform-speaking">
        <div class="speaking-waveform">
            <div class="speaking-wave-line"></div>
        </div>
        <div class="speaking-status">🤖 AI Responding...</div>
        <div class="speaking-icon">🔵</div>
    </div>
    """
    
    return st.markdown(speaking_css, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 