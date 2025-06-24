import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import logging
from typing import List
from multi_agent_core import run_selected_analysis
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import tempfile
import base64
import openai
from streamlit_mic_recorder import mic_recorder

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Clear Streamlit cache at startup
st.cache_data.clear()
st.cache_resource.clear()

# Page config
st.set_page_config(
    page_title="Multi-Agent Performance Marketing Analysis",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

def transcribe_audio(audio_data):
    """Transcribe audio using OpenAI Whisper"""
    # Create a file-like object from audio bytes
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_data['bytes'])
        tmp_file.flush()
        
        # Use the modern OpenAI API (v1.0+)
        client = openai.OpenAI()
        with open(tmp_file.name, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        os.unlink(tmp_file.name)  # Clean up temp file
    
    return transcript.text

def generate_elevenlabs_audio(text):
    """Generate audio using ElevenLabs"""
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        st.error("ElevenLabs API key not found")
        return None
    
    client = ElevenLabs(api_key=api_key)
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    
    voice_settings = VoiceSettings(
        stability=0.7,
        similarity_boost=0.8,
        style=0.4,
        use_speaker_boost=True,
        speed=1.0
    )
    
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        voice_settings=voice_settings,
        output_format="mp3_22050_32"
    )
    
    return b"".join(audio)

def show_speaking_animation():
    """Show speaking animation"""
    st.markdown(
        "<div style='font-size:48px; text-align:center;'>🗣️<br><span style='font-size:24px;'>Speaking...</span></div>",
        unsafe_allow_html=True
    )

# Main app
st.title("🤖 Multi-Agent Performance Marketing Analysis")
st.markdown("Upload your data and select agent analyses to run. Use voice input for questions!")

# Create tabs for main app and reference
tab1, tab2 = st.tabs(["📊 Analysis", "📖 Reference"])

with tab1:
    # File upload
    st.subheader("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        st.session_state.data = data
        st.success("File uploaded successfully!")
        
        with st.expander("📊 Data Preview", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", len(data))
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            with col4:
                missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                st.metric("Data Completeness", f"{100 - missing_pct:.1f}%")
            
            st.dataframe(data.head(10), use_container_width=True)

    # Only use uploaded data - no demo fallback
    data_to_use = st.session_state.data

    # Agent selection
    st.subheader("🤖 Select Agent Analyses")
    agent_options = [
        "Buyer Performance",
        "Publisher Quality", 
        "Comparative Analysis",
        "Executive Summary"
    ]

    selected_agents = st.multiselect(
        "Choose which analyses to run:",
        agent_options,
        default=agent_options
    )

    # Simple Voice Interface
    st.subheader("🎤 Voice Input")
    
    # Initialize session state
    if 'processing_audio' not in st.session_state:
        st.session_state.processing_audio = False
    
    # Simple voice recorder
    audio_data = mic_recorder(
        start_prompt="🎤 Record Your Question",
        stop_prompt="⏹️ Stop & Process",
        just_once=False,
        use_container_width=True,
        key="simple_voice"
    )
    
    user_query = ""
    # Process audio immediately when available
    if audio_data and not st.session_state.processing_audio:
        st.session_state.processing_audio = True
        
        if isinstance(audio_data, dict) and 'bytes' in audio_data:
            st.audio(audio_data['bytes'], format="audio/wav")
            
            with st.spinner("🔄 Transcribing your voice..."):
                try:
                    user_query = transcribe_audio(audio_data)
                    if user_query and len(user_query.strip()) > 2:
                        st.success(f"🗣️ You said: **{user_query}**")
                        
                        # Auto-run analysis for voice input
                        if data_to_use is not None:
                            with st.spinner("🤖 Running analysis..."):
                                results, summary = run_selected_analysis(data_to_use, selected_agents)
                                
                                # Display results
                                st.subheader("📊 Analysis Results")
                                st.write(summary)
                                
                                # Auto voice response
                                if summary:
                                    with st.spinner("🗣️ Generating response..."):
                                        try:
                                            audio_bytes = generate_elevenlabs_audio(summary)
                                            if audio_bytes:
                                                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                                                st.success("🔊 Response complete!")
                                        except Exception as e:
                                            st.error(f"Voice generation failed: {e}")
                        else:
                            st.error("⚠️ Please upload data first!")
                    else:
                        st.warning("🔇 No speech detected. Try again!")
                        
                except Exception as e:
                    st.error(f"❌ Transcription failed: {e}")
        
        # Reset processing flag
        st.session_state.processing_audio = False
    
    # Text input as alternative
    st.subheader("💬 Text Input")
    text_query = st.text_input("Type your question:", placeholder="e.g., How many sales did we have today?")

    # Manual text analysis
    if st.button("🚀 Run Analysis", type="primary") and text_query:
        if data_to_use is None:
            st.error("⚠️ Please upload a data file first before running analysis.")
        else:
            with st.spinner("🤖 Running multi-agent analysis..."):
                results, summary = run_selected_analysis(data_to_use, selected_agents)
                
                # Display results
                st.subheader("📊 Agent Findings")
                for agent in selected_agents:
                    key = agent.lower().replace(" ", "_")
                    if key in results and results[key]:
                        with st.expander(f"**{agent}**", expanded=True):
                            st.json(results[key])
                
                # Executive summary
                st.subheader("📋 Executive Summary")
                st.write(summary)
                
                # Generate voice response for text input too
                if summary:
                    with st.spinner("🗣️ Generating voice response..."):
                        try:
                            audio_bytes = generate_elevenlabs_audio(summary)
                            if audio_bytes:
                                show_speaking_animation()
                                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                                st.success("🔊 Response delivered!")
                        except Exception as e:
                            st.error(f"Voice generation failed: {e}")

    # Info
    if st.session_state.data is None:
        st.info("👆 Please upload a CSV or Excel file to begin analysis. The app will analyze your data using the Performance Marketing Analysis Dictionary business logic.")

with tab2:
    # Performance Marketing Analysis Dictionary Reference
    st.header("📖 Performance Marketing Analysis Dictionary")
    st.markdown("### *Business Logic Reference for Multi-Agent Analysis*")
    
    st.markdown("---")
    st.subheader("🎯 Core Entity Fields")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **PUBLISHER** - The affiliate or traffic source generating leads
        - *Data Type*: Text
        - *Business Impact*: Primary determinant of lead quality
        - *Analysis Role*: Source of traffic quality issues vs execution problems
        
        **BUYER** - The client company purchasing the leads
        - *Data Type*: Text  
        - *Business Impact*: Controls sales execution and operational capacity
        - *Analysis Role*: Responsible for conversion optimization and agent performance
        """)
    
    with col2:
        st.markdown("""
        **TARGET** - The specific call center division within the buyer organization
        - *Data Type*: Text
        - *Business Impact*: Operational unit handling the leads
        - *Analysis Role*: Division-level performance analysis within buyers
        
        **CAMPAIGN** - The marketing initiative driving the traffic
        - *Data Type*: Text
        - *Business Impact*: Campaign-level optimization insights
        """)
    
    st.markdown("---")
    st.subheader("📈 Primary Success Metrics")
    
    st.markdown("""
    ### **Conversion Rate** (Primary KPI)
    **Formula**: `(Number of Sales ÷ Number of Leads) × 100`
    - *Measurement*: Percentage of calls that result in sales
    - *Success Threshold*: Varies by industry (typically 5-15%)
    - *Analysis Use*: Primary measure of Publisher-Buyer combination success
    
    ### **Successful Combinations Analysis**
    **Definition**: Publisher-Buyer pairs or Publisher-Buyer-Target triplets with optimal conversion rates
    - *Measurement*: Statistical comparison of conversion rates across combinations
    - *Success Indicators*:
      - Above-average conversion rates
      - High lead quality scores
      - Efficient sales execution
      - Low issue rates
    """)
    
    st.markdown("---")
    st.subheader("🔍 Lead Quality Indicators")
    
    st.markdown("""
    ### **CUSTOMER INTENT** (Primary Quality Metric)
    **Values**: "Level 1", "Level 2", "Level 3", "Negative Intent", "Not Detected"
    - **Level 2 & 3**: High-quality leads requiring immediate attention
    - **Level 1, Negative Intent, Not Detected**: Poor quality leads
    - *Quality Threshold*: Only Level 2 & 3 considered strong leads
    - *Critical Rule*: **Level 3 non-conversions indicate sales execution problems**
    - *Action Required*: All Level 2 & 3 leads need immediate follow-up
    
    ### **BILLABLE** 
    **Values**: "Yes", "No"
    - *Business Logic*: "No" = Publisher sending very poor quality leads
    - *Quality Indicator*: Billable rate should be >70% for healthy publishers
    - *Action Trigger*: Low billable rates require publisher review
    
    ### **AD MISLED**
    **Values**: "Yes", "No"
    - *Business Logic*: "Yes" = Critical compliance violation
    - *Action Required*: **Immediate publisher review and correction**
    - *Reporting*: Must quantify total count and highlight offending publishers
    - *Escalation*: **Zero tolerance policy** for misleading advertising
    """)
    
    st.markdown("---")
    st.subheader("💼 Sales Execution Indicators")
    
    st.markdown("""
    ### **Quote-to-Call Ratio**
    **Formula**: `(QUOTE = "Yes" count ÷ Total Calls) × 100`
    - *Business Logic*: Measures agent engagement and lead qualification
    - *Benchmark*: Should be 40-60% for healthy operations
    
    ### **Quote-to-Sale Ratio**
    **Formula**: `(SALE = "Yes" count ÷ QUOTE = "Yes" count) × 100`
    - *Business Logic*: Measures closing ability
    - *Critical Analysis*: If Quote-to-Call ≠ Quote-to-Sale, indicates closing problems
    - *Action Trigger*: <30% suggests sales training needed
    
    ### **REACHED AGENT**
    **Values**: "Yes", "No"
    - *Business Logic*: "No" = Buyer capacity/availability problem
    - *Critical Threshold*: **>90% reach rate required**
    - *Escalation Trigger*: Immediate management attention for low rates
    - *Impact*: Direct revenue loss from missed opportunities
    """)
    
    st.markdown("---")
    st.subheader("⚠️ Critical Business Rules & Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### **Immediate Action Triggers**
        1. 🚨 **Agent Availability Crisis**: REACHED AGENT <90%
        2. 🚨 **Ad Compliance Violation**: AD MISLED = "Yes" (any count)
        3. 🚨 **High-Value Lead Waste**: Level 3 intent with SALE = "No"
        4. 🚨 **Sales Training Need**: OBJECTION WITH NO REBUTTAL = "Yes"
        5. 🚨 **Capacity Issue**: IVR rate >20%
        
        ### **Quality Thresholds**
        - **Strong Publisher**: Billable Rate >70%, Level 2+3 Intent >30%
        - **Strong Buyer**: Agent Reach >90%, Quote-to-Sale >30%
        - **Successful Combination**: Conversion Rate >Industry Average + Lead Quality Score >70
        """)
    
    with col2:
        st.markdown("""
        ### **Performance Benchmarks**
        - **Conversion Rate**: 5-15% (industry dependent)
        - **Lead Quality Rate**: >50% Level 2+3 intent
        - **Sales Execution**: >30% quote-to-sale conversion
        - **Operational Efficiency**: >90% agent reach rate
        - **Billable Rate**: >70% for healthy publishers
        - **Ad Compliance**: 0% tolerance for violations
        
        ### **Stage Progression Quality**
        - **Stage 5 (Enrollment)**: Highest quality outcome
        - **Stage 4 (Plan Detail)**: Strong engagement
        - **Stage 3 (Needs Analysis)**: Good qualification
        - **Stage 2 (Eligibility)**: Basic qualification
        - **Stage 1 (Introduction)**: Initial contact only
        """)
    
    st.markdown("---")
    st.subheader("🤖 Analysis Framework: Publisher vs Buyer Issues")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### **Publisher Quality Problems** (Lead Generation Issues)
        **Indicators**:
        - Low billable rates (<50%)
        - High ad misled rates (>5%)
        - Low customer intent levels (<30% Level 2+3)
        - Short call durations (<3 minutes average)
        - Poor stage 1-2 progression
        
        **Root Cause**: Traffic quality, targeting, or compliance issues
        """)
    
    with col2:
        st.markdown("""
        ### **Buyer Execution Problems** (Sales Performance Issues)
        **Indicators**:
        - Good billable rates (>70%) but low conversion
        - High Level 2+3 intent but low sales
        - Poor quote-to-sale ratios (<30%)
        - High objection-no-rebuttal rates
        - Poor stage 3-5 progression
        - Low agent reach rates
        
        **Root Cause**: Sales training, capacity, or process issues
        """)
    
    st.markdown("---")
    st.subheader("💡 Voice Assistant Query Examples")
    
    st.markdown("""
    ### **Successful Combination Analysis**
    - *"What makes Publisher X and Buyer Y successful together?"*
    - *"Which Publisher-Buyer-Target combinations have the highest conversion rates?"*
    - *"Show me the performance factors for our best combinations"*
    
    ### **Issue Identification**
    - *"Is this a lead quality or sales execution problem?"*
    - *"Which publishers are sending bad leads vs which buyers can't close?"*
    - *"Show me all ad misled violations by publisher"*
    
    ### **Immediate Actions**
    - *"Which Level 3 leads didn't convert today?"*
    - *"Show me agent availability issues requiring immediate attention"*
    - *"List all objection handling problems for training"*
    """)
    
    st.markdown("---")
    st.info("💡 **This framework enables sophisticated analysis to distinguish between lead quality issues (publisher responsibility) and sales execution issues (buyer responsibility) while identifying the optimal combinations for maximum conversion success.**") 