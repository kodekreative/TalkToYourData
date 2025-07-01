#!/usr/bin/env python3
"""
Performance Marketing Diagnostic Tool
Comprehensive analysis of Publisher-Buyer-Target combinations
Identifies lead quality vs sales execution issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import importlib
import sys

# Force reload of marketing agents modules to ensure latest changes
if 'marketing_agents.agents.lead_analyst' in sys.modules:
    importlib.reload(sys.modules['marketing_agents.agents.lead_analyst'])
if 'marketing_agents.agents.writer_agent' in sys.modules:
    importlib.reload(sys.modules['marketing_agents.agents.writer_agent'])
if 'marketing_agents.agents.pds_lead_agent' in sys.modules:
    importlib.reload(sys.modules['marketing_agents.agents.pds_lead_agent'])
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import re
from advanced_diagnostic_features import create_advanced_analysis_page, create_executive_summary_page

# Import talk to your data functions
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, RunContextWrapper
from typing import TypedDict

class QueryContext(TypedDict):
    query: str
    df_info: dict
# Import marketing agents system

try:
    from marketing_agents.orchestrator import AgentOrchestrator
    from marketing_agents.sample_data import generate_sample_marketing_data
    AGENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Marketing Agents system not available: {e}")
    AGENTS_AVAILABLE = False

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

def dynamic_instructions(
    context: RunContextWrapper[QueryContext], 
    agent: Agent[QueryContext]
) -> str:
    """Generate dynamic instructions for the agent based on context"""
    try:
        # Access the context data
        if not context.context:
            raise ValueError("Missing context values")

        ctx = context.context
        df_info = ctx["df_info"]
        query = ctx["query"]
        
        return f"""
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
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error generating dynamic instructions: {str(e)}\n{tb}")
        if 'st' in globals():
            st.error(f"Error generating dynamic instructions: {str(e)}")
            st.code(tb)
        
        # Return a fallback instruction instead of None
        return """
You are a data analyst. Generate Python pandas code to answer user queries.
Return a JSON object with 'code', 'explanation', and 'visualization_type' keys.
"""

def init_dynamic_agent():
    """Initialize the dynamic agent with proper error handling"""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None

        # Create agent with dynamic instructions function
        agent = Agent[QueryContext](
            name="Pandas Analyst Agent",
            instructions=dynamic_instructions,  # Pass the function directly
            model="gpt-4"
        )
        
        return agent
        
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
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

async def query_with_gpt4o(query: str, df, agent):
    """Query the agent with dynamic context"""
    try:
        if not agent:
            return None, "Agent not initialized"

        # Get dataframe information
        df_info = get_dataframe_info(df)

        # Prepare context for dynamic instructions
        context: QueryContext = {
            "query": query,
            "df_info": df_info
        }

        # Run the agent with the query and context
        # The dynamic_instructions function will receive this context
        result = await Runner.run(
            agent,
            query,          # The user input/query
            context=context # This gets passed to dynamic_instructions
        )

        if not result or not hasattr(result, 'final_output') or not result.final_output:
            return None, "No response from agent"

        result_text = result.final_output.strip()

        # Parse JSON response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```", 1)[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            parsed = json.loads(result_text)
            if all(key in parsed for key in ['code', 'explanation', 'visualization_type']):
                return parsed, None
            else:
                return None, "Missing required keys in response"
        except json.JSONDecodeError as je:
            return None, f"JSON parsing error: {str(je)}"

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Agent query error: {str(e)}\n{tb}")
        if 'st' in globals():
            st.error(f"Agent query error: {str(e)}")
        return None, f"Agent query error: {str(e)}"

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

def create_listening_animation():
    """Create a visual animation while listening for voice input"""
    st.markdown("""
    <div class="voice-waveform-listening">
        <style>
        .voice-waveform-listening {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 60px;
            margin: 20px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 15px;
        }
        .wave {
            width: 4px;
            height: 30px;
            background: white;
            margin: 0 2px;
            border-radius: 2px;
            animation: wave 1.5s ease-in-out infinite;
        }
        .wave:nth-child(2) { animation-delay: 0.1s; }
        .wave:nth-child(3) { animation-delay: 0.2s; }
        .wave:nth-child(4) { animation-delay: 0.3s; }
        .wave:nth-child(5) { animation-delay: 0.4s; }
        .wave:nth-child(6) { animation-delay: 0.5s; }
        .wave:nth-child(7) { animation-delay: 0.6s; }
        
        @keyframes wave {
            0%, 100% { transform: scaleY(0.3); opacity: 0.6; }
            50% { transform: scaleY(1); opacity: 1; }
        }
        </style>
        <div style="text-align: center; color: white;">
            <h3 style="margin: 0; color: white;">🎤 Listening... Speak now!</h3>
            <div style="margin-top: 10px;">
                <div class="wave"></div>
                <div class="wave"></div>
                <div class="wave"></div>
                <div class="wave"></div>
                <div class="wave"></div>
                <div class="wave"></div>
                <div class="wave"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_speaking_animation():
    """Create a visual animation while AI is speaking"""
    st.markdown("""
    <div class="voice-waveform-speaking">
        <style>
        .voice-waveform-speaking {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 50px;
            margin: 15px 0;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            border-radius: 10px;
            padding: 10px;
        }
        .speak-wave {
            width: 3px;
            height: 25px;
            background: white;
            margin: 0 1.5px;
            border-radius: 2px;
            animation: speak 0.8s ease-in-out infinite;
        }
        .speak-wave:nth-child(2) { animation-delay: 0.05s; }
        .speak-wave:nth-child(3) { animation-delay: 0.1s; }
        .speak-wave:nth-child(4) { animation-delay: 0.15s; }
        .speak-wave:nth-child(5) { animation-delay: 0.2s; }
        .speak-wave:nth-child(6) { animation-delay: 0.25s; }
        .speak-wave:nth-child(7) { animation-delay: 0.3s; }
        .speak-wave:nth-child(8) { animation-delay: 0.35s; }
        .speak-wave:nth-child(9) { animation-delay: 0.4s; }
        
        @keyframes speak {
            0%, 100% { transform: scaleY(0.2); opacity: 0.5; }
            25% { transform: scaleY(0.8); opacity: 0.8; }
            50% { transform: scaleY(1.2); opacity: 1; }
            75% { transform: scaleY(0.6); opacity: 0.7; }
        }
        </style>
        <div style="text-align: center; color: white;">
            <h4 style="margin: 0; color: white;">🔊 AI Speaking...</h4>
            <div style="margin-top: 8px;">
                <div class="speak-wave"></div>
                <div class="speak-wave"></div>
                <div class="speak-wave"></div>
                <div class="speak-wave"></div>
                <div class="speak-wave"></div>
                <div class="speak-wave"></div>
                <div class="speak-wave"></div>
                <div class="speak-wave"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

def create_marketing_agents_page(diagnostic):
    """Create the Marketing Agents System page"""
    st.header("🤖 Marketing Agents System")
    st.subheader("AI-Powered Two-Agent Marketing Analytics")
    
    if not AGENTS_AVAILABLE:
        st.error("⚠️ Marketing Agents system is not available. Please check the installation.")
        return
    
    # Check for data availability - use persistent data as fallback
    data_source = None
    if diagnostic.data is not None:
        data_source = diagnostic.data
        st.success("✅ Using data from main diagnostic system")
    elif st.session_state.get('data_loaded', False) and st.session_state.get('persistent_data') is not None:
        data_source = st.session_state.persistent_data
        st.success("✅ Using persistent data from memory")
        # Try to load into diagnostic system for compatibility
        try:
            import io
            buffer = io.BytesIO()
            data_source.to_excel(buffer, index=False)
            buffer.seek(0)
            buffer.name = "persistent_data.xlsx"
            diagnostic.load_data(buffer)
            diagnostic.generate_critical_alerts()
            diagnostic.analyze_combinations()
            st.info("🔄 Also loaded into diagnostic system for full compatibility")
        except Exception as e:
            st.warning(f"⚠️ Could not load into diagnostic system: {str(e)}")
    else:
        st.warning("⚠️ No data available for agent analysis")
        st.info("Upload a CSV/Excel file with marketing data containing columns like: PUBLISHER, TARGET, COST PER SALE, CALL COUNT, SALES RATE, etc.")
        
        # Quick data upload option
        st.markdown("---")
        st.markdown("### 🚀 Quick Data Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload data directly for agents:",
                type=['xlsx', 'xls', 'csv'],
                help="Upload your marketing data directly here",
                key="agents_direct_upload"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.persistent_data = df.copy()
                    st.session_state.data_loaded = True
                    
                    # Load into diagnostic
                    diagnostic.load_data(uploaded_file)
                    diagnostic.generate_critical_alerts()
                    diagnostic.analyze_combinations()
                    
                    st.success(f"✅ Data loaded! Shape: {df.shape}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
        
        with col2:
            st.markdown("**Or use sample data:**")
            if st.button("📊 Load Sample Marketing Data", type="secondary", use_container_width=True):
                try:
                    # Generate sample data using the marketing agents system
                    from marketing_agents.sample_data import generate_sample_marketing_data
                    sample_df = generate_sample_marketing_data()
                    
                    st.session_state.persistent_data = sample_df.copy()
                    st.session_state.data_loaded = True
                    
                    # Create a temporary file for diagnostic loading
                    import io
                    buffer = io.BytesIO()
                    sample_df.to_excel(buffer, index=False)
                    buffer.seek(0)
                    buffer.name = "sample_data.xlsx"
                    
                    diagnostic.load_data(buffer)
                    diagnostic.generate_critical_alerts()
                    diagnostic.analyze_combinations()
                    
                    st.success(f"✅ Sample data loaded! Shape: {sample_df.shape}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading sample data: {str(e)}")
        
        return
    
    # Initialize the agent orchestrator with loaded data
    if 'agent_orchestrator' not in st.session_state:
        st.session_state.agent_orchestrator = AgentOrchestrator()
    
    # Load data into orchestrator using the available data source
    try:
        load_result = st.session_state.agent_orchestrator.load_data(data_source)
        if load_result['status'] == 'success':
            st.success(f"✅ {load_result['message']}")
        else:
            st.error(f"❌ Failed to load data: {load_result}")
            return
    except Exception as e:
        st.error(f"❌ Error loading data into agents: {str(e)}")
        return
    
    # Show system status
    with st.expander("🔧 System Status", expanded=False):
        status = st.session_state.agent_orchestrator.get_system_status()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Agent Status:**")
            for agent, status_val in status['agents'].items():
                st.write(f"- {agent.replace('_', ' ').title()}: {status_val}")
            
            st.write("**Data Info:**")
            if 'data_info' in status:
                data_info = status['data_info']
                st.write(f"- Records: {data_info['records']:,}")
                st.write(f"- Publishers: {data_info['publishers']}")
                st.write(f"- Total Calls: {data_info['total_calls']:,}")
        
        with col2:
            st.write("**Capabilities:**")
            caps = status['capabilities']
            nlq_status = '✅' if caps['natural_language_queries'] else '❌'
            report_status = '✅' if caps['business_report_generation'] else '❌'
            st.write(f"- Natural Language Queries: {nlq_status}")
            st.write(f"- Business Reports: {report_status}")
            st.write(f"- Predefined Analyses: {caps['predefined_analyses']}")
            st.write(f"- Report Formats: {caps['report_formats']}")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Natural Language Analysis", "📊 Predefined Analyses", "📝 Business Reports", "📚 History"])
    
    with tab1:
        st.subheader("Ask Questions About Your Marketing Data")
        
        # Query input
        query = st.text_input(
            "Ask the Lead Management Analyst:",
            placeholder="e.g., What's the percentage of Level 2 and Level 3 calls?",
            help="Ask natural language questions about your marketing performance data",
            key="agents_query_input"
        )
        
        # Sample queries
        st.write("**💡 Try these sample queries:**")
        sample_queries = st.session_state.agent_orchestrator.get_sample_queries()
        
        cols = st.columns(3)
        for i, sample in enumerate(sample_queries[:6]):
            with cols[i % 3]:
                if st.button(f"📊 {sample[:30]}...", key=f"sample_agents_{i}", help=sample):
                    st.session_state['auto_agents_query'] = sample
                    st.rerun()
        
        # Check for auto-generated query and execute automatically
        auto_execute = False
        if 'auto_agents_query' in st.session_state:
            query = st.session_state['auto_agents_query']
            del st.session_state['auto_agents_query']
            auto_execute = True
            st.info(f"🎯 **Selected Query:** {query}")
            st.write("*Analyzing automatically...*")
        
        # Execute query button or auto-execute
        if (st.button("🚀 Analyze Query", key="execute_agents_query") and query) or (auto_execute and query):
            with st.spinner("🤖 Lead Management Analyst is analyzing your query..."):
                # Validate query first
                validation = st.session_state.agent_orchestrator.validate_query(query)
                
                if not validation['valid']:
                    st.error(f"❌ {validation['error']}")
                    if 'suggestions' in validation:
                        st.write("**Suggestions:**")
                        for suggestion in validation['suggestions']:
                            st.write(f"- {suggestion}")
                else:
                    # Analyze query
                    analysis_result = st.session_state.agent_orchestrator.analyze_query(query)
                    
                    if "error" in analysis_result:
                        st.error(f"❌ Analysis Error: {analysis_result['error']}")
                    else:
                        # Display analysis results
                        st.success("✅ Analysis Complete!")
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            query_type = analysis_result.get('query_type', analysis_result.get('analysis_type', 'general'))
                            st.write(f"**Query Type:** {query_type.replace('_', ' ').title()}")
                            st.write(f"**Confidence:** {analysis_result.get('confidence', 'medium').title()}")
                        with col2:
                            st.write(f"**Timestamp:** {analysis_result.get('timestamp', 'N/A')}")
                        
                        # Show insights
                        if 'insights' in analysis_result and analysis_result['insights']:
                            st.subheader("🎯 Key Insights")
                            for insight in analysis_result['insights']:
                                st.write(f"• {insight}")
                        
                        # Show data if available
                        if 'data' in analysis_result and analysis_result['data']:
                            st.subheader("📊 Analysis Data")
                            data = analysis_result['data']
                            
                            if isinstance(data, list) and len(data) > 0:
                                # Convert list of dicts to DataFrame for display
                                df_display = pd.DataFrame(data)
                                st.dataframe(df_display)
                            elif isinstance(data, dict):
                                # Display dict data in a formatted way
                                for key, value in data.items():
                                    if key not in ['error']:
                                        # Handle different value types for st.metric
                                        if isinstance(value, (int, float)):
                                            if 'rate' in key.lower() or 'percentage' in key.lower():
                                                st.metric(key.replace('_', ' ').title(), f"{value:.1f}%")
                                            elif 'cost' in key.lower() or 'revenue' in key.lower():
                                                st.metric(key.replace('_', ' ').title(), f"${value:,.2f}")
                                            else:
                                                st.metric(key.replace('_', ' ').title(), f"{value:,}")
                                        elif isinstance(value, str):
                                            st.metric(key.replace('_', ' ').title(), value)
                                        elif isinstance(value, dict):
                                            # For nested dictionaries, show as expandable
                                            st.write(f"**{key.replace('_', ' ').title()}:**")
                                            for sub_key, sub_value in value.items():
                                                if isinstance(sub_value, (int, float)):
                                                    st.write(f"  - {sub_key}: {sub_value:.2f}")
                                                else:
                                                    st.write(f"  - {sub_key}: {sub_value}")
                                        else:
                                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        
                        # Show recommendations
                        if 'recommendations' in analysis_result and analysis_result['recommendations']:
                            st.subheader("🚀 Recommendations")
                            for i, rec in enumerate(analysis_result['recommendations'], 1):
                                st.write(f"{i}. {rec}")
                        
                        # Store result for report generation
                        st.session_state['last_analysis'] = analysis_result
    
    with tab2:
        st.subheader("Predefined Marketing Analyses")
        
        predefined_analyses = st.session_state.agent_orchestrator.get_predefined_analyses()
        
        # Create cards for each analysis
        for analysis in predefined_analyses:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{analysis['name']}**")
                    st.write(analysis['description'])
                    st.caption(f"Recommended format: {analysis['recommended_format'].replace('_', ' ').title()}")
                
                with col2:
                    if st.button("Run Analysis", key=f"run_{analysis['name']}"):
                        with st.spinner(f"Running {analysis['name']}..."):
                            result = st.session_state.agent_orchestrator.run_predefined_analysis(analysis['name'])
                            
                            if "error" in result:
                                st.error(f"❌ {result['error']}")
                            else:
                                st.success("✅ Analysis Complete!")
                                
                                # Show analysis
                                if 'analysis' in result:
                                    analysis_data = result['analysis']
                                    if 'insights' in analysis_data:
                                        st.write("**Insights:**")
                                        for insight in analysis_data['insights']:
                                            st.write(f"• {insight}")
                                
                                # Show report
                                if 'report' in result:
                                    report_data = result['report']
                                    st.subheader(f"📝 {report_data.get('title', 'Report')}")
                                    st.text(report_data.get('content', 'No content available'))
                
                st.divider()
    
    with tab3:
        st.subheader("Generate Business Reports")
        
        if 'last_analysis' not in st.session_state:
            st.info("👆 Run an analysis first to generate business reports")
        else:
            st.write("**Generate reports from your last analysis:**")
            
            # Report format selection
            format_options = {
                'daily_summary': '📅 Daily Summary - What happened yesterday',
                'recommendations': '🎯 Recommendations - What actions to take',
                'detailed_analysis': '📊 Detailed Analysis - Comprehensive insights',
                'executive_summary': '👔 Executive Summary - Strategic overview'
            }
            
            selected_format = st.selectbox(
                "Choose report format:",
                options=list(format_options.keys()),
                format_func=lambda x: format_options[x],
                key="report_format_select"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📝 Generate Single Report", key="generate_single_report"):
                    with st.spinner("✍️ Writer Agent is creating your report..."):
                        report = st.session_state.agent_orchestrator.generate_report(
                            st.session_state['last_analysis'], 
                            selected_format
                        )
                        
                        if "error" in report:
                            st.error(f"❌ Report Generation Error: {report['error']}")
                        else:
                            st.success("✅ Report Generated!")
                            st.subheader(f"📄 {report.get('title', 'Report')}")
                            st.markdown(report.get('content', 'No content available'))
                            
                            # Show metadata
                            if 'word_count' in report:
                                st.caption(f"Word count: {report['word_count']} | Confidence: {report.get('confidence', 'N/A')}")
            
            with col2:
                if st.button("📚 Generate All Report Formats", key="generate_all_reports"):
                    with st.spinner("✍️ Writer Agent is creating all report formats..."):
                        all_reports = st.session_state.agent_orchestrator.generate_all_reports(
                            st.session_state['last_analysis']
                        )
                        
                        if "error" in all_reports:
                            st.error(f"❌ Report Generation Error: {all_reports['error']}")
                        else:
                            st.success("✅ All Reports Generated!")
                            
                            # Display all reports in expandable sections
                            formats = all_reports.get('formats', {})
                            for format_type, report_data in formats.items():
                                if "error" not in report_data:
                                    with st.expander(f"📄 {format_options.get(format_type, format_type)}", expanded=False):
                                        st.subheader(report_data.get('title', 'Report'))
                                        st.markdown(report_data.get('content', 'No content available'))
                                        
                                        if 'word_count' in report_data:
                                            st.caption(f"Word count: {report_data['word_count']} | Confidence: {report_data.get('confidence', 'N/A')}")
    
    with tab4:
        st.subheader("Conversation History")
        
        # History controls
        col1, col2 = st.columns([1, 1])
        with col1:
            history_limit = st.selectbox("Show last:", [5, 10, 20, 50], index=1, key="history_limit")
        with col2:
            if st.button("🗑️ Clear History", key="clear_agents_history"):
                result = st.session_state.agent_orchestrator.clear_conversation_history()
                st.success(result['message'])
                st.rerun()
        
        # Display history
        history = st.session_state.agent_orchestrator.get_conversation_history(history_limit)
        
        if not history:
            st.info("No conversation history yet. Start by asking a question!")
        else:
            for i, entry in enumerate(reversed(history), 1):
                with st.expander(f"Entry {len(history) - i + 1}: {entry['type'].title()} - {entry['timestamp'][:19]}", expanded=False):
                    if entry['type'] == 'analysis':
                        st.write(f"**Query:** {entry.get('query', 'N/A')}")
                        result = entry.get('result', {})
                        if 'insights' in result:
                            st.write("**Insights:**")
                            for insight in result['insights']:
                                st.write(f"• {insight}")
                    
                    elif entry['type'] == 'report':
                        st.write(f"**Format:** {entry.get('format_type', 'N/A')}")
                        result = entry.get('result', {})
                        if 'content' in result:
                            content = result.get('content', '')
                            # Show truncated content with markdown formatting
                            if len(content) > 500:
                                st.markdown(content[:500] + "...")
                                with st.expander("📄 Show Full Content"):
                                    st.markdown(content)
                            else:
                                st.markdown(content)
                    
                    st.caption(f"Timestamp: {entry['timestamp']}")
    
    # Footer info
    st.markdown("---")
    st.caption("🤖 **Two-Agent System:** Lead Management Analyst analyzes data → Writer Agent formats business communications")
    st.caption("💡 **Tip:** For guided step-by-step analysis, use the dedicated '🎯 Guided Analysis' page from the navigation menu.")

def create_guided_analysis_page(diagnostic):
    """Create the Guided Analysis page"""
    st.header("🎯 Guided Analysis Workflow")
    st.subheader("AI-Powered Step-by-Step Marketing Analysis")
    st.markdown("**Select what you want to analyze, how you want it formatted, and for which time period**")
    
    # Check for data availability - use persistent data as fallback
    data_source = None
    if diagnostic.data is not None:
        data_source = diagnostic.data
        st.success("✅ Using data from main diagnostic system")
    elif st.session_state.get('data_loaded', False) and st.session_state.get('persistent_data') is not None:
        data_source = st.session_state.persistent_data
        st.success("✅ Using persistent data from memory")
        # Try to load into diagnostic system for compatibility
        try:
            import io
            buffer = io.BytesIO()
            data_source.to_excel(buffer, index=False)
            buffer.seek(0)
            buffer.name = "persistent_data.xlsx"
            diagnostic.load_data(buffer)
            diagnostic.generate_critical_alerts()
            diagnostic.analyze_combinations()
            st.info("🔄 Also loaded into diagnostic system for full compatibility")
        except Exception as e:
            st.warning(f"⚠️ Could not load into diagnostic system: {str(e)}")
    else:
        st.warning("⚠️ No data available for guided analysis")
        st.info("Upload a CSV/Excel file with marketing data containing columns like: PUBLISHER, TARGET, COST PER SALE, CALL COUNT, SALES RATE, etc.")
        
        # Quick data upload option
        st.markdown("---")
        st.markdown("### 🚀 Quick Data Upload")
        uploaded_file = st.file_uploader(
            "Upload data directly for guided analysis:",
            type=['xlsx', 'xls', 'csv'],
            help="Upload your marketing data directly here",
            key="guided_direct_upload"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.persistent_data = df.copy()
                st.session_state.data_loaded = True
                
                # Load into diagnostic
                diagnostic.load_data(uploaded_file)
                diagnostic.generate_critical_alerts()
                diagnostic.analyze_combinations()
                
                st.success(f"✅ Data loaded! Shape: {df.shape}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
        return

    # Initialize the agent orchestrator with loaded data
    if 'agent_orchestrator' not in st.session_state:
        from marketing_agents.orchestrator import AgentOrchestrator
        st.session_state.agent_orchestrator = AgentOrchestrator()
    
    # Load data into orchestrator using the available data source
    try:
        load_result = st.session_state.agent_orchestrator.load_data(data_source)
        if load_result['status'] == 'success':
            st.success(f"✅ {load_result['message']}")
        else:
            st.error(f"❌ Failed to load data: {load_result}")
            return
    except Exception as e:
        st.error(f"❌ Error loading data into guided analysis: {str(e)}")
        return

    # Prompt Management Section
    with st.expander("⚙️ Customize Agent Prompts", expanded=False):
        st.markdown("### 🔧 Agent Prompt Management")
        st.markdown("Customize the prompts used by each agent to improve analysis quality. Changes are saved to `prompts_config.md`.")
        
        # Load current prompts
        prompts = load_prompts_from_markdown()
        
        # Create tabs for each agent
        prompt_tab1, prompt_tab2, prompt_tab3, prompt_tab4, prompt_tab5 = st.tabs([
            "🎯 Lead Quality Analyst", 
            "📈 Publisher Performance", 
            "💰 Cost Efficiency", 
            "🔄 Conversion Funnel",
            "✍️ Writer Agent"
        ])
        
        with prompt_tab1:
            st.markdown("**Lead Quality Analyst Prompt:**")
            new_lead_prompt = st.text_area(
                "Edit the Lead Quality Analyst prompt:",
                value=prompts.get('lead_quality_analyst', ''),
                height=200,
                key="lead_quality_prompt"
            )
            if st.button("💾 Save Lead Quality Prompt", key="save_lead_prompt"):
                save_prompt_to_markdown('lead_quality_analyst', new_lead_prompt)
                st.success("✅ Lead Quality Analyst prompt saved!")
                st.rerun()
        
        with prompt_tab2:
            st.markdown("**Publisher Performance Analyst Prompt:**")
            new_pub_prompt = st.text_area(
                "Edit the Publisher Performance Analyst prompt:",
                value=prompts.get('publisher_performance_analyst', ''),
                height=200,
                key="publisher_prompt"
            )
            if st.button("💾 Save Publisher Performance Prompt", key="save_pub_prompt"):
                save_prompt_to_markdown('publisher_performance_analyst', new_pub_prompt)
                st.success("✅ Publisher Performance Analyst prompt saved!")
                st.rerun()
        
        with prompt_tab3:
            st.markdown("**Cost Efficiency Analyst Prompt:**")
            new_cost_prompt = st.text_area(
                "Edit the Cost Efficiency Analyst prompt:",
                value=prompts.get('cost_efficiency_analyst', ''),
                height=200,
                key="cost_prompt"
            )
            if st.button("💾 Save Cost Efficiency Prompt", key="save_cost_prompt"):
                save_prompt_to_markdown('cost_efficiency_analyst', new_cost_prompt)
                st.success("✅ Cost Efficiency Analyst prompt saved!")
                st.rerun()
        
        with prompt_tab4:
            st.markdown("**Conversion Funnel Analyst Prompt:**")
            new_funnel_prompt = st.text_area(
                "Edit the Conversion Funnel Analyst prompt:",
                value=prompts.get('conversion_funnel_analyst', ''),
                height=200,
                key="funnel_prompt"
            )
            if st.button("💾 Save Conversion Funnel Prompt", key="save_funnel_prompt"):
                save_prompt_to_markdown('conversion_funnel_analyst', new_funnel_prompt)
                st.success("✅ Conversion Funnel Analyst prompt saved!")
                st.rerun()
        
        with prompt_tab5:
            st.markdown("**Writer Agent Prompt:**")
            new_writer_prompt = st.text_area(
                "Edit the Writer Agent prompt:",
                value=prompts.get('writer_agent', ''),
                height=200,
                key="writer_prompt"
            )
            if st.button("💾 Save Writer Agent Prompt", key="save_writer_prompt"):
                save_prompt_to_markdown('writer_agent', new_writer_prompt)
                st.success("✅ Writer Agent prompt saved!")
                st.rerun()
        
        # Show current prompt file location
        st.markdown("---")
        st.info(f"📄 Prompts are stored in: `prompts_config.md`")
        if st.button("🔄 Reset All Prompts to Default", key="reset_prompts"):
            if st.button("⚠️ Confirm Reset", key="confirm_reset"):
                reset_prompts_to_default()
                st.success("✅ All prompts reset to default!")
                st.rerun()

    st.markdown("---")

    # Simple guided analysis interface
    st.markdown("### 🎯 Quick Analysis")
    
    # Pre-defined analysis options - Multiple analyst types available
    analysis_options = {
        # "intent_analysis": "🎯 Lead Management Analysis - AI-powered analysis with Claude 3.5 Sonnet (Uses Custom Prompt)",
        "intent_analysis": "🎯 Lead Management Analysis - AI-powered analysis with OpenAI GPT-4 (Uses Custom Prompt)",
        "pds_analysis": "📊 PDS Lead Analysis - Detailed pandas-based statistical analysis with performance rankings and ROI calculations",
    }
    
    selected_analysis = st.selectbox(
        "Choose your analysis type:",
        options=list(analysis_options.keys()),
        format_func=lambda x: analysis_options[x],
        key="guided_analysis_type"
    )
    
    # Format selection
    report_format = st.radio(
        "Report format:",
        options=['detailed_analysis', 'executive_summary'],
        format_func=lambda x: '📊 Detailed Analysis' if x == 'detailed_analysis' else '👔 Executive Summary',
        key="guided_report_format"
    )
    
    # Generate analysis button
    if st.button("🚀 Generate Analysis", type="primary", key="generate_analysis"):
        with st.spinner("🤖 Generating your analysis..."):
            
            # FIRST: Run the working quantitative analysis to get real data
            st.info("📊 Step 1: Generating quantitative analysis...")
            
            # Generate real lead quality data using the working function
            lead_quality_data = diagnostic.analyze_lead_quality()
            
            # Calculate real intent distribution
            intent_distribution = data_source['CUSTOMER_INTENT'].value_counts(normalize=True) * 100
            
            # Calculate real conversion metrics
            overall_conversion_rate = (data_source['SALE'] == 'Yes').mean() * 100
            
            # Get top and bottom performers
            top_publishers = lead_quality_data.nlargest(3, 'Conversion_Rate')[['PUBLISHER', 'Conversion_Rate', 'Total_Leads']].to_dict('records')
            bottom_publishers = lead_quality_data.nsmallest(3, 'Conversion_Rate')[['PUBLISHER', 'Conversion_Rate', 'Total_Leads']].to_dict('records')
            
            # Create real data summary for GPT-4o to interpret
            real_data_summary = f"""
REAL DATA ANALYSIS RESULTS:

INTENT LEVEL DISTRIBUTION:
{intent_distribution.to_dict()}

OVERALL METRICS:
- Total Records: {len(data_source):,}
- Overall Conversion Rate: {overall_conversion_rate:.2f}%
- Total Publishers: {data_source['PUBLISHER'].nunique()}

TOP PERFORMING PUBLISHERS:
{top_publishers}

BOTTOM PERFORMING PUBLISHERS:
{bottom_publishers}

DETAILED PUBLISHER DATA:
{lead_quality_data.to_string()}
"""
            
            st.info("🤖 Step 2: Generating AI insights from real data...")
            
            # Create analysis query based on selection - Make it extensive to trigger GPT-4o
            query_map = {
                "intent_analysis": f"""As a Lead Management Analyst, perform an extensive analysis focusing on: Comprehensive publisher performance evaluation, lead quality assessment, and cost optimization analysis

REAL DATA TO ANALYZE:
{real_data_summary}

ANALYSIS FOCUS:
- Publisher performance evaluation across all three layers (Publisher, Publisher+Buyer, Publisher+Buyer+Target)
- Lead quality assessment with intent level distribution analysis
- Cost analysis including cost per sale, cost per quote, and ROI calculations
- Performance variance analysis and statistical outlier identification
- Trending analysis and anomaly detection

REPORT FORMAT: Comprehensive detailed analysis with minimum 1000 words
Analysis Depth Required: Three-layer publisher analysis with volume thresholds and statistical significance

Key Metrics to Analyze:
- Publisher performance indices and relative comparisons
- Call count, sales rate, quote rate, and conversion metrics by publisher
- Intent level distribution (Level 3, Level 2, Level 1) with conversion rates
- Ad misled rates and impact on performance
- Cost per sale and cost per quote analysis

Business Questions:
1. Which publishers deliver the best ROI and should receive increased budget allocation?
2. What is the performance variance within each publisher (avg vs high vs low)?
3. Which publishers are statistical outliers requiring immediate attention?
4. What are the cost implications and optimization opportunities by publisher?
5. What trending patterns and anomalies require investigation?

Please provide specific publisher names, statistical analysis, and actionable recommendations with implementation timelines.""",

                "publisher_performance": """As a 📈 Publisher Performance Analyst, perform an extensive analysis focusing on: Evaluate publisher effectiveness, conversion rates, and ROI by source

ANALYSIS FOCUS:
- Publisher-by-publisher performance analysis with conversion rates, cost efficiency, and ROI metrics
- Traffic quality assessment and lead generation effectiveness by source
- Revenue attribution and budget allocation optimization recommendations
- Competitive benchmarking and market positioning analysis

REPORT FORMAT: Comprehensive detailed analysis with minimum 1000 words
Analysis Depth Required: Comprehensive publisher performance evaluation, ROI analysis, strategic optimization planning

Key Metrics to Analyze:
- Conversion rates by publisher with statistical significance testing
- Cost-per-acquisition and return-on-ad-spend (ROAS) by source
- Lead quality distribution across publishers and revenue impact
- Market share analysis and competitive positioning assessment

Business Questions:
1. Which publishers deliver the highest ROI and should receive increased budget allocation?
2. What is the revenue impact of reallocating budget from underperforming sources?
3. How do our publisher partnerships compare to industry benchmarks?
4. Which traffic sources provide the highest-quality leads with best conversion potential?
5. What strategic partnerships should we develop or terminate based on performance data?

Please provide executive-level strategic recommendations with implementation timelines and expected ROI calculations.""",

                "cost_efficiency": """As a 💰 Cost Efficiency Analyst, perform an extensive analysis focusing on: Analyze cost per sale, ROI, and revenue optimization opportunities

ANALYSIS FOCUS:
- Comprehensive cost-per-acquisition analysis across all marketing channels and publishers
- Revenue optimization opportunities and profit margin enhancement strategies
- Budget allocation efficiency and resource utilization assessment
- Financial impact modeling and ROI maximization recommendations

REPORT FORMAT: Comprehensive detailed analysis with minimum 1000 words
Analysis Depth Required: Comprehensive cost analysis, ROI optimization, financial impact modeling

Key Metrics to Analyze:
- Cost-per-sale by channel, publisher, and customer intent level
- Return-on-investment (ROI) calculations with confidence intervals
- Revenue per lead and lifetime value projections
- Budget efficiency scores and allocation optimization opportunities

Business Questions:
1. What is our current cost-per-acquisition by channel and how does it compare to industry benchmarks?
2. Which marketing investments provide the highest ROI and should receive increased funding?
3. What cost reduction opportunities exist without sacrificing lead quality or conversion rates?
4. How can we optimize our budget allocation to maximize revenue generation?
5. What financial impact would result from implementing recommended cost optimization strategies?

Please provide executive-level strategic recommendations with implementation timelines and expected ROI calculations.""",

                "conversion_funnel": """As a 🔄 Conversion Funnel Analyst, perform an extensive analysis focusing on: Track leads through sales stages and identify bottlenecks

ANALYSIS FOCUS:
- End-to-end conversion funnel analysis from lead generation to sale completion
- Bottleneck identification and conversion rate optimization opportunities
- Sales stage performance assessment and pipeline efficiency evaluation
- Revenue leakage analysis and retention improvement strategies

REPORT FORMAT: Comprehensive detailed analysis with minimum 1000 words
Analysis Depth Required: Comprehensive funnel analysis, bottleneck identification, conversion optimization planning

Key Metrics to Analyze:
- Conversion rates at each stage of the sales funnel
- Drop-off points and revenue leakage quantification
- Time-to-conversion analysis and sales velocity optimization
- Lead nurturing effectiveness and follow-up strategy performance

Business Questions:
1. Where in our sales funnel are we losing the most potential revenue?
2. What specific bottlenecks are preventing higher conversion rates?
3. How can we optimize our lead nurturing and follow-up processes?
4. What is the financial impact of improving conversion rates at each funnel stage?
5. Which sales process improvements would deliver the highest ROI?

Please provide executive-level strategic recommendations with implementation timelines and expected ROI calculations."""
            }
            
            # Handle PDS analysis separately - ENHANCED CLAUDE + PDS STATISTICAL ANALYSIS
            if selected_analysis == "pds_analysis":
                try:
                    from marketing_agents.agents.pds_lead_agent import PDSLeadAnalyst
                    
                    # Initialize the actual PDS Lead Analyst with statistical capabilities
                    pds_analyst = PDSLeadAnalyst(data_source)
                    
                    # Get Claude's excellent analysis first
                    claude_analysis = st.session_state.agent_orchestrator.analyze_query(
                        f"As a Lead Management Analyst, perform extensive analysis focusing on: Comprehensive publisher performance evaluation, lead quality assessment, and statistical analysis with pandas/numpy. Remember that Level 1 = POOR intent (low-quality leads), Level 2 = MEDIUM TO HIGH intent (moderate-quality leads), Level 3 = BEST intent (highest-quality leads)."
                    )
                    
                    # Get PDS statistical report
                    complete_report = pds_analyst.run_complete_analysis()
                    
                    # Get structured results for display
                    pds_results = pds_analyst.analyze_query(
                        "Comprehensive publisher performance analysis", 
                        data_source, 
                        "pds_publisher_analysis"
                    )
                    
                    if "error" in pds_results:
                        st.error(f"❌ PDS Analysis Error: {pds_results['error']}")
                    else:
                        # Display Enhanced PDS Analysis Results
                        st.success("✅ Enhanced PDS Lead Management Analysis Complete!")
                        
                        st.markdown("# 🎯 ENHANCED PDS LEAD MANAGEMENT ANALYST")
                        st.markdown("### AI-Powered Insights + Statistical Analysis with Publisher-by-Publisher Breakdown")
                        
                        # Display summary metrics with comparisons
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Publishers Analyzed", pds_results.get('publisher_count', 0))
                        with col2:
                            st.metric("Total Calls", f"{pds_results.get('total_calls', 0):,}")
                        with col3:
                            st.metric("Total Sales", pds_results.get('total_sales', 0))
                        with col4:
                            st.metric("Portfolio Conversion", f"{pds_results.get('avg_conversion', 0):.1f}%")
                        
                        # Claude's AI Insights Section
                        st.markdown("## 🧠 AI-Enhanced Performance Insights")
                        # st.markdown("*Powered by Claude 3.5 Sonnet with statistical benchmarking*")
                        st.markdown("*Powered by OpenAI GPT-4 with statistical benchmarking*")
                        
                        insights = claude_analysis.get('insights', [])
                        if insights:
                            for i, insight in enumerate(insights, 1):
                                insight_text = insight if isinstance(insight, str) else str(insight)
                                st.markdown(f"**{i}.** {insight_text}")
                        
                        # Strategic Recommendations from Claude
                        st.markdown("## 🎯 Strategic Recommendations")
                        recommendations = claude_analysis.get('recommendations', [])
                        if recommendations:
                            for i, rec in enumerate(recommendations, 1):
                                if isinstance(rec, dict):
                                    rec_text = rec.get('recommendation', str(rec))
                                    category = rec.get('category', 'Strategic')
                                    implementation = rec.get('implementation', [])
                                    st.markdown(f"**{i}. {rec_text}**")
                                    st.markdown(f"   *Category: {category}*")
                                    if implementation:
                                        with st.expander(f"Implementation Steps for Recommendation {i}"):
                                            for step in implementation:
                                                st.markdown(f"- {step}")
                                else:
                                    st.markdown(f"**{i}.** {rec}")
                        
                        # Statistical Analysis & Publisher Breakdown
                        with st.expander("📊 Statistical Analysis & Publisher Breakdown", expanded=True):
                            st.markdown("### Detailed Statistical Analysis with Anomaly Detection")
                            st.markdown("*Portfolio averages, percentile rankings, and buyer-level performance*")
                            
                            # Display the complete formatted report with statistical comparisons
                            st.text(complete_report)
                            
                            # Additional structured insights from PDS analysis
                            pds_insights = pds_results.get('insights', [])
                            if pds_insights:
                                st.markdown("### Statistical Insights")
                                for i, insight in enumerate(pds_insights, 1):
                                    st.markdown(f"**{i}.** {insight}")
                        
                        # Business Context from Claude
                        business_context = claude_analysis.get('business_context', '')
                        if business_context:
                            with st.expander("💼 Business Context & Strategic Implications", expanded=False):
                                st.markdown(business_context)
                        
                        # Confidence Assessment
                        confidence = claude_analysis.get('confidence', {})
                        if confidence:
                            st.markdown("## ✅ Analysis Confidence")
                            if isinstance(confidence, dict):
                                level = confidence.get('level', 'Unknown')
                                factors = confidence.get('factors', [])
                                st.markdown(f"**Confidence Level:** {level}")
                                if factors:
                                    st.markdown("**Supporting Factors:**")
                                    for factor in factors:
                                        st.markdown(f"- {factor}")
                        
                        # Enhanced Download with Combined Report
                        enhanced_report = f"""ENHANCED PDS LEAD MANAGEMENT ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== AI-ENHANCED INSIGHTS (Claude 3.5 Sonnet) ===
{chr(10).join([f"{i+1}. {insight}" for i, insight in enumerate(insights)])}

=== STRATEGIC RECOMMENDATIONS ===
{chr(10).join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])}

=== STATISTICAL ANALYSIS & PUBLISHER BREAKDOWN ===
{complete_report}

=== BUSINESS CONTEXT ===
{business_context}
"""
                        
                        st.download_button(
                            "📥 Download Enhanced PDS Analysis Report", 
                            enhanced_report,
                            file_name=f"enhanced_pds_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            key="download_enhanced_pds_report"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Enhanced PDS Analysis Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    
            else:
                # Handle other analysis types (Claude/GPT)
                query = query_map[selected_analysis]
                
                try:
                    # Run analysis
                    analysis_result = st.session_state.agent_orchestrator.analyze_query(query)
                    
                    if "error" in analysis_result:
                        st.error(f"❌ Analysis Error: {analysis_result['error']}")
                    else:
                        # Generate report
                        enhanced_analysis_result = analysis_result.copy()
                        enhanced_analysis_result['extensive_mode'] = True
                        
                        # SKIP WRITER AGENT - Display Claude's results Directly
                        # st.success("✅ Claude 3.5 Sonnet Analysis Complete!")
                        st.success("✅ OpenAI GPT-4 Analysis Complete!")
                        
                        # Display Claude's raw analysis directly
                        insights = enhanced_analysis_result.get('insights', [])
                        recommendations = enhanced_analysis_result.get('recommendations', [])
                        business_context = enhanced_analysis_result.get('business_context', '')
                        strategic_implications = enhanced_analysis_result.get('strategic_implications', '')
                        confidence = enhanced_analysis_result.get('confidence', {})
                        
                        st.markdown("# 🎯 Lead Quality Analysis Report")
                        # st.markdown("### Powered by Claude 3.5 Sonnet")
                        st.markdown("### Powered by OpenAI GPT-4")
                        
                        # Executive Summary with first insight
                        if insights:
                            first_insight = insights[0] if isinstance(insights[0], str) else str(insights[0])
                            st.markdown(f"**🔑 Key Finding:** {first_insight}")
                        
                        # Key Insights Section
                        if insights:
                            st.markdown("## 📊 Strategic Insights")
                            for i, insight in enumerate(insights, 1):
                                insight_text = insight if isinstance(insight, str) else str(insight)
                                st.markdown(f"**{i}.** {insight_text}")
                                st.markdown("")  # Add spacing
                        
                        # Strategic Recommendations
                        if recommendations:
                            st.markdown("## 🎯 Strategic Recommendations")
                            for i, rec in enumerate(recommendations, 1):
                                if isinstance(rec, dict):
                                    rec_text = rec.get('recommendation', str(rec))
                                    category = rec.get('category', 'General')
                                    implementation = rec.get('implementation', [])
                                    expected_roi = rec.get('expected_roi', 'Not specified')
                                    
                                    st.markdown(f"### {i}. {rec_text}")
                                    st.markdown(f"**Category:** {category}")
                                    if expected_roi != 'Not specified':
                                        st.markdown(f"**Expected ROI:** {expected_roi}")
                                    if implementation:
                                        st.markdown("**Implementation Steps:**")
                                        for step in implementation:
                                            st.markdown(f"- {step}")
                                else:
                                    st.markdown(f"**{i}.** {rec}")
                                st.markdown("")  # Add spacing
                        
                        # Business Context
                        if business_context:
                            st.markdown("## 💼 Business Context")
                            st.markdown(business_context)
                        
                        # Strategic Implications
                        if strategic_implications:
                            st.markdown("## 🚀 Strategic Implications")
                            st.markdown(strategic_implications)
                        
                        # Confidence Level
                        if confidence:
                            st.markdown("## ✅ Analysis Confidence")
                            if isinstance(confidence, dict):
                                level = confidence.get('level', 'Unknown')
                                factors = confidence.get('factors', [])
                                st.markdown(f"**Confidence Level:** {level}")
                                if factors:
                                    st.markdown("**Supporting Factors:**")
                                    for factor in factors:
                                        st.markdown(f"- {factor}")
                            else:
                                st.markdown(f"**Confidence Level:** {confidence}")
                                
                        # Add data summary
                        st.markdown("## 📋 Analysis Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Records Analyzed", f"{len(data_source):,}")
                        with col2:
                            st.metric("Insights Generated", len(insights))
                        with col3:
                            st.metric("Recommendations", len(recommendations))
                            
                            st.markdown("---")
                            
                            # WORKING QUANTITATIVE ANALYSIS (THE GOOD STUFF)
                            st.subheader("📊 WORKING ANALYSIS RESULTS")
                            st.info("This section shows the quantitative analysis that actually works and provides real insights")
                            
                            # Show the working lead quality analysis
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**🏆 Publisher Performance Rankings**")
                                
                                # Sort by conversion rate
                                quality_ranking = lead_quality_data.sort_values('Conversion_Rate', ascending=False)
                                
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
                                
                                # Show top performers with real insights
                                st.markdown("**🎯 Key Insights from Real Data:**")
                                top_3 = quality_ranking.head(3)
                                for idx, row in top_3.iterrows():
                                    st.write(f"• **{row['PUBLISHER']}**: {row['Conversion_Rate']:.1f}% conversion rate")
                            
                            with col2:
                                st.markdown("**📈 Customer Intent Distribution**")
                                
                                # Customer intent distribution with correct interpretation
                                intent_dist = data_source['CUSTOMER_INTENT'].value_counts()
                                
                                fig = px.pie(
                                    values=intent_dist.values,
                                    names=intent_dist.index,
                                    title='Customer Intent Distribution'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add real business insights [CORRECTED MEMORY: Level 1 = POOR intent, Level 3 = BEST intent]
                                st.markdown("**💡 Intent Level Analysis:**")
                                level_3_pct = (intent_dist.get('Level 3', 0) / len(data_source) * 100)
                                level_1_pct = (intent_dist.get('Level 1', 0) / len(data_source) * 100)
                                st.write(f"• **Level 3 (Best Intent)**: {level_3_pct:.1f}% - Premium leads")
                                st.write(f"• **Level 1 (Poor Intent)**: {level_1_pct:.1f}% - Low-quality leads")
                            
                            # Detailed quality metrics table
                            st.markdown("**📋 Publisher Quality Scorecard**")
                            
                            # Add quality scores
                            lead_quality_data['Quality_Score'] = (
                                lead_quality_data['Conversion_Rate'] * 0.4 +
                                lead_quality_data['Strong_Lead_Rate'] * 0.3 +
                                lead_quality_data['Billable_Rate'] * 0.2 +
                                (1 - lead_quality_data['IVR_Rate']) * 0.1
                            ).round(3)
                            
                            # Sort by quality score
                            quality_scorecard = lead_quality_data.sort_values('Quality_Score', ascending=False)
                            
                            st.dataframe(quality_scorecard, use_container_width=True)
                            
                            st.markdown("---")
                            
                            # DEBUG SECTION - Show Raw Claude Results
                            # with st.expander("🔍 DEBUG: Raw Claude 3.5 Sonnet Results", expanded=False):
                            #     st.markdown("**Raw Analysis Result from Claude:**")
                            #     st.json(enhanced_analysis_result)
                            with st.expander("🔍 DEBUG: Raw OpenAI GPT-4 Results", expanded=False):
                                st.markdown("**Raw Analysis Result from GPT-4")
                                st.json(enhanced_analysis_result)
                            # Download option
                            download_content = f"""# Lead Quality Analysis Report
Powered by Claude 3.5 Sonnet

## Key Insights
{chr(10).join([f"{i+1}. {insight}" for i, insight in enumerate(insights)])}

## Strategic Recommendations  
{chr(10).join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])}

## Business Context
{business_context}

## Strategic Implications
{strategic_implications}
"""
                            
                            st.download_button(
                                # "📥 Download Claude Analysis Report",
                                "📥 Download OpenAI GPT-4 Analysis Report",
                                download_content,
                                file_name=f"claude_analysis_{selected_analysis}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                key="download_claude_report"
                            )
                            
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

def create_debug_data_page(diagnostic):
    """Create a debug page that duplicates data upload and stores in pandas dataframe"""
    st.header("🔧 Debug Data Page")
    st.subheader("Step-by-step data analysis debugging")
    
    # Initialize session state for debug data
    if 'debug_data' not in st.session_state:
        st.session_state.debug_data = None
    
    # Step 1: Data Upload
    st.markdown("### Step 1: Upload Data")
    uploaded_file = st.file_uploader(
        "Upload Excel file for debugging",
        type=['xlsx', 'xls'],
        help="Upload your performance marketing data for step-by-step analysis",
        key="debug_upload"
    )
    
    if uploaded_file is not None:
        try:
            # Load the data
            df = pd.read_excel(uploaded_file)
            st.session_state.debug_data = df
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Show basic info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
                st.metric("Total Columns", df.shape[1])
            with col2:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                st.metric("Numeric Columns", len(df.select_dtypes(include=['number']).columns))
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.session_state.debug_data = None
    
    # Step 2: Data Inspection
    if st.session_state.debug_data is not None:
        df = st.session_state.debug_data
        
        st.markdown("---")
        st.markdown("### Step 2: Data Inspection")
        
        # Data preview tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Preview", "📋 Info", "🔍 Sample", "📈 Stats"])
        
        with tab1:
            st.markdown("**Data Preview (First 10 rows):**")
            st.dataframe(df.head(10), use_container_width=True)
            
        with tab2:
            st.markdown("**Column Information:**")
            info_data = []
            for col in df.columns:
                info_data.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Non-Null Count': df[col].notna().sum(),
                    'Null Count': df[col].isna().sum(),
                    'Unique Values': df[col].nunique(),
                    'Sample Value': str(df[col].iloc[0]) if len(df) > 0 else 'N/A'
                })
            
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True)
            
        with tab3:
            st.markdown("**Random Sample (5 rows):**")
            if len(df) >= 5:
                sample_df = df.sample(n=5)
                st.dataframe(sample_df, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
                
        with tab4:
            st.markdown("**Statistical Summary:**")
            # Numeric columns stats
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.markdown("**Numeric Columns:**")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Step 3: Export Options
        st.markdown("---")
        st.markdown("### Step 3: Export & Next Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download as CSV",
                csv,
                f"debug_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        
        with col2:
            if st.button("🔄 Load into Main App"):
                st.session_state.persistent_data = df.copy()
                st.session_state.data_loaded = True
                st.success("✅ Data loaded into main app!")
        
        with col3:
            if st.button("🤖 Go to Agents"):
                st.session_state.direct_to_agents = True
                st.rerun()
    
    else:
        st.info("👆 Upload a file to begin debugging")

def load_prompts_from_markdown():
    """Load prompts from the markdown configuration file"""
    try:
        with open('prompts_config.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        prompts = {}
        
        # Extract Lead Management Analyst prompt
        lead_match = re.search(r'## Lead Management Analyst Prompt\s*```(.*?)```', content, re.DOTALL)
        if lead_match:
            prompts['lead_quality_analyst'] = lead_match.group(1).strip()
        
        # Extract Publisher Performance Analyst prompt
        pub_match = re.search(r'## Publisher Performance Analyst Prompt\s*```(.*?)```', content, re.DOTALL)
        if pub_match:
            prompts['publisher_performance_analyst'] = pub_match.group(1).strip()
        
        # Extract Cost Efficiency Analyst prompt
        cost_match = re.search(r'## Cost Efficiency Analyst Prompt\s*```(.*?)```', content, re.DOTALL)
        if cost_match:
            prompts['cost_efficiency_analyst'] = cost_match.group(1).strip()
        
        # Extract Conversion Funnel Analyst prompt
        funnel_match = re.search(r'## Conversion Funnel Analyst Prompt\s*```(.*?)```', content, re.DOTALL)
        if funnel_match:
            prompts['conversion_funnel_analyst'] = funnel_match.group(1).strip()
        
        # Extract Writer Agent prompt
        writer_match = re.search(r'## Writer Agent Prompt\s*```(.*?)```', content, re.DOTALL)
        if writer_match:
            prompts['writer_agent'] = writer_match.group(1).strip()
        
        return prompts
        
    except FileNotFoundError:
        # Return default prompts if file doesn't exist
        return {
            'lead_quality_analyst': 'Default Lead Management Analyst prompt not found.',
            'publisher_performance_analyst': 'Default Publisher Performance Analyst prompt not found.',
            'cost_efficiency_analyst': 'Default Cost Efficiency Analyst prompt not found.',
            'conversion_funnel_analyst': 'Default Conversion Funnel Analyst prompt not found.',
            'writer_agent': 'Default Writer Agent prompt not found.'
        }
    except Exception as e:
        st.error(f"Error loading prompts: {str(e)}")
        return {}

def save_prompt_to_markdown(agent_type, new_prompt):
    """Save updated prompt to the markdown configuration file"""
    try:
        # Load current content
        try:
            with open('prompts_config.md', 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            content = "# Agent Prompts Configuration\n\n"
        
        # Update timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Map agent types to section headers
        section_map = {
            'lead_quality_analyst': 'Lead Management Analyst Prompt',
            'publisher_performance_analyst': 'Publisher Performance Analyst Prompt', 
            'cost_efficiency_analyst': 'Cost Efficiency Analyst Prompt',
            'conversion_funnel_analyst': 'Conversion Funnel Analyst Prompt',
            'writer_agent': 'Writer Agent Prompt'
        }
        
        section_header = section_map.get(agent_type, 'Unknown Agent Prompt')
        
        # Create new section content
        new_section = f"## {section_header}\n\n```\n{new_prompt}\n```\n\n"
        
        # Replace existing section or add new one
        pattern = f"## {re.escape(section_header)}.*?```.*?```"
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, new_section.strip(), content, flags=re.DOTALL)
        else:
            # Add new section before the footer
            if "---" in content:
                content = content.replace("---", f"{new_section}---")
            else:
                content += f"\n{new_section}"
        
        # Update timestamp
        content = re.sub(r'\*Last Updated:.*?\*', f'*Last Updated: {timestamp}*', content)
        if '*Last Updated:' not in content:
            content += f"\n---\n\n*Last Updated: {timestamp}*\n*Modified by: User*"
        
        # Save updated content
        with open('prompts_config.md', 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        st.error(f"Error saving prompt: {str(e)}")
        return False

def reset_prompts_to_default():
    """Reset all prompts to their default values"""
    try:
        # Read the current prompts_config.md to get the default content
        # This will restore the original prompts
        default_content = """# Agent Prompts Configuration

This file stores customizable prompts for the marketing analysis agents. Users can modify these prompts through the Guided Analysis interface to improve analysis quality.

## Lead Management Analyst Prompt

```
You are a Lead Management Analyst AI agent that analyzes marketing performance data with expertise in publisher evaluation, lead quality assessment, and performance optimization.

ANALYSIS REQUEST: {query}
ANALYSIS TYPE: {analysis_type}

CURRENT PERFORMANCE DATA:
{data_summary}

[Rest of default prompt content...]
```

---

*Last Updated: {timestamp}*
*Modified by: System*
""".replace('{timestamp}', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        with open('prompts_config.md', 'w', encoding='utf-8') as f:
            f.write(default_content)
        
        return True
        
    except Exception as e:
        st.error(f"Error resetting prompts: {str(e)}")
        return False

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
        st.header("📁 Data Source")
        
        # Initialize persistent data storage
        if 'persistent_data' not in st.session_state:
            st.session_state.persistent_data = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            
        # Auto-load sample data if no data is loaded
        if not st.session_state.data_loaded:
            try:
                sample_file = "sample_data.xlsx"
                df = pd.read_excel(sample_file)
                st.session_state.persistent_data = df.copy()
                st.session_state.data_loaded = True
                
                # Load into diagnostic
                with open(sample_file, 'rb') as file:
                    if diagnostic.load_data(file):
                        diagnostic.generate_critical_alerts()
                        diagnostic.analyze_combinations()
                        st.success("✅ Sample data loaded automatically!")
                    else:
                        st.error("Failed to load sample data into diagnostic system")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
                st.info("Falling back to manual upload...")
        
        # Keep file upload as fallback
        st.markdown("### Upload Custom Data")
        uploaded_file = st.file_uploader(
            "Upload Excel file (optional)",
            type=['xlsx', 'xls'],
            help="Upload your own performance marketing data to replace sample data"
        )
        
        if uploaded_file is not None:
            if st.button("Load Custom Data"):
                try:
                    # Load into persistent storage
                    df = pd.read_excel(uploaded_file)
                    st.session_state.persistent_data = df.copy()
                    st.session_state.data_loaded = True
                    
                    # Load into diagnostic
                    if diagnostic.load_data(uploaded_file):
                        diagnostic.generate_critical_alerts()
                        diagnostic.analyze_combinations()
                        st.success("Custom data loaded successfully!")
                    else:
                        st.error("Failed to load custom data into diagnostic system")
                except Exception as e:
                    st.error(f"Error loading custom data: {str(e)}")
        
        # Data status
        if st.session_state.get('data_loaded', False):
            st.success(f"✅ Data loaded: {len(st.session_state.persistent_data):,} records")
        else:
            st.info("📤 No data loaded")
        
        st.markdown("---")
        
        # DIRECT AGENTIC LINK - Most prominent
        st.header("🚀 Quick Access")
        if st.button("🎯 Guided Analysis", type="primary", use_container_width=True):
            st.session_state.direct_to_guided = True
            st.rerun()
            
        if st.button("🤖 AI Marketing Agents", type="secondary", use_container_width=True):
            st.session_state.direct_to_agents = True
            st.rerun()
        
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
            "🎤 Voice Test Page",
            "🤖 Marketing Agents",
            "🎯 Guided Analysis",
            "🔧 Debug Data Page"
        ]
        
        # Set default index based on current page
        default_index = 0
        if 'current_page' in st.session_state and st.session_state.current_page in nav_options:
            default_index = nav_options.index(st.session_state.current_page)
        
        selected_page = st.selectbox("Choose page:", nav_options, index=default_index)
    
    # Check for direct navigation to guided analysis
    if st.session_state.get('direct_to_guided', False):
        st.session_state.direct_to_guided = False  # Reset flag
        st.session_state.current_page = "🎯 Guided Analysis"
        selected_page = "🎯 Guided Analysis"
    
    # Check for direct navigation to agents
    if st.session_state.get('direct_to_agents', False):
        st.session_state.direct_to_agents = False  # Reset flag
        st.session_state.current_page = "🤖 Marketing Agents"
        selected_page = "🤖 Marketing Agents"
    
    # Maintain current page state if set
    if 'current_page' in st.session_state and st.session_state.current_page in nav_options:
        if selected_page == nav_options[0]:  # Only override if user didn't explicitly select a different page
            selected_page = st.session_state.current_page
    
    # Update current page when user makes a selection
    if selected_page != st.session_state.get('current_page', ''):
        st.session_state.current_page = selected_page
    
    # Main content area
    if diagnostic.data is None and not st.session_state.get('data_loaded', False):
        st.info("👆 Please upload your performance marketing data to begin analysis")
    else:
        # Route to selected page
        if selected_page == "Dashboard Overview":
            if diagnostic.data is not None:
                create_alert_dashboard(diagnostic)
            else:
                st.info("Please load data first")
        
        elif selected_page == "Alert Center":
            if diagnostic.data is not None:
                create_alert_dashboard(diagnostic)
            else:
                st.info("Please load data first")
        
        elif selected_page == "Combination Analysis":
            if diagnostic.data is not None:
                create_combination_analysis(diagnostic)
            else:
                st.info("Please load data first")
        
        elif selected_page == "Lead Quality Analysis":
            if diagnostic.data is not None:
                create_lead_quality_analysis(diagnostic)
            else:
                st.info("Please load data first")
        
        elif selected_page == "Sales Execution Analysis":
            if diagnostic.data is not None:
                create_sales_execution_analysis(diagnostic)
            else:
                st.info("Please load data first")
        
        elif selected_page == "Advanced Analysis":
            st.header("🔬 Advanced Analysis")
            st.info("Advanced analysis features coming soon...")
        
        elif selected_page == "Executive Summary":
            st.header("📊 Executive Summary")
            st.info("Executive summary features coming soon...")
        
        elif selected_page == "Talk to Your Data":
            create_talk_to_data_page(diagnostic)
        
        elif selected_page == "🎤 Voice Test Page":
            create_voice_test_page(diagnostic)
        
        elif selected_page == "🤖 Marketing Agents":
            create_marketing_agents_page(diagnostic)
            
        elif selected_page == "🎯 Guided Analysis":
            create_guided_analysis_page(diagnostic)
            
        elif selected_page == "🔧 Debug Data Page":
            create_debug_data_page(diagnostic)

if __name__ == "__main__":
    main() 