#!/usr/bin/env python3
"""
Simple ElevenLabs Voice Test
Troubleshoot voice generation and playback issues
"""

import streamlit as st
import os
import io
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="ElevenLabs Voice Test",
    page_icon="🔊",
    layout="wide"
)

st.title("🔊 ElevenLabs Voice Test")
st.write("Simple test to troubleshoot ElevenLabs voice generation")

# Check API key
elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
if not elevenlabs_api_key or elevenlabs_api_key == "your_elevenlabs_api_key_here":
    st.error("❌ ElevenLabs API key not found!")
    st.write("Please check your .env file contains:")
    st.code("ELEVENLABS_API_KEY=your_actual_api_key_here", language="bash")
    st.stop()
else:
    st.success(f"✅ ElevenLabs API key found: {elevenlabs_api_key[:10]}...")

# Voice discovery and selection
st.subheader("🎭 Voice Selection")

# Add button to fetch available voices
if st.button("🔍 Fetch My Available Voices from ElevenLabs"):
    try:
        with st.spinner("Fetching your available voices..."):
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs(api_key=elevenlabs_api_key)
            
            # Get available voices
            voices = client.voices.get_all()
            
            st.success(f"✅ Found {len(voices.voices)} voices in your account!")
            
            # Display voices in a nice format
            st.subheader("📋 Your Available Voices:")
            for voice in voices.voices:
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**{voice.name}**")
                with col2:
                    st.write(f"`{voice.voice_id}`")
                with col3:
                    if hasattr(voice, 'labels') and voice.labels:
                        labels = voice.labels
                        gender = labels.get('gender', 'Unknown')
                        accent = labels.get('accent', 'Unknown')
                        st.write(f"{gender}, {accent}")
                    else:
                        st.write("No labels")
                
                # Add description if available
                if hasattr(voice, 'description') and voice.description:
                    st.caption(voice.description)
                st.markdown("---")
                
    except Exception as e:
        st.error(f"❌ Error fetching voices: {str(e)}")

# Manual voice ID input
st.subheader("🎯 Custom Voice Selection")
use_custom_voice = st.checkbox("Use custom voice ID", help="Enter a specific voice ID manually")

if use_custom_voice:
    custom_voice_id = st.text_input(
        "Enter Voice ID:",
        placeholder="e.g., pNInz6obpgDQGcFmaJgB",
        help="Copy the voice ID from the list above"
    )
    custom_voice_name = st.text_input(
        "Voice Name (for display):",
        placeholder="e.g., Shannon",
        help="What do you want to call this voice?"
    )
    
    if custom_voice_id and custom_voice_name:
        selected_voice_id = custom_voice_id
        selected_voice_name = custom_voice_name
        st.info(f"✅ Using custom voice: **{custom_voice_name}** (`{custom_voice_id}`)")
    else:
        st.warning("⚠️ Please enter both Voice ID and Name")
        selected_voice_id = None
        selected_voice_name = None
else:
    # Predefined voice options (fallback)
    st.subheader("📚 Predefined Voice Options")
    
    st.info("💡 **Free Tier Note**: Some voices require a paid ElevenLabs subscription. Free voices are marked with 🆓")
    
    voice_options = {
        "Custom Voice (Test)": "F7hCTbeEDbm7osolS21j",
        "Rachel (Professional Female) 🆓": "21m00Tcm4TlvDq8ikWAM",
        "Drew (Warm Male) 🆓": "29vD33N1CtxCmqQRPOHJ", 
        "Clyde (Middle-aged Male) 🆓": "2EiwWnXFnfNSVaq5eqzxF",
        "Bella (Professional Female) 🆓": "EXAVITQu4vr4xnSDxMaL",
        "Antoni (Young Male) 🆓": "ErXwobaYiN019PkySvjV",
        "Elli (Young Female) 🆓": "MF3mGyEYCl7XYWbV9V6O",
        "Josh (Deep Male) 🆓": "TxGEqnHWrfWFTfGW9XjX",
        "Arnold (Narrator Male) 🆓": "VR6AewLTigWG4xSOukaG",
        "Adam (Deep Male) 🆓": "pNInz6obpgDQGcFmaJgB",
        "Sam (Raspy Male) 🆓": "yoZ06aMxZJJ28mfd3POQ",
        "Shannon (Expressive Female) 💰": "0GoLoBHogFMTLhDROxLD",
        "Charlotte (British Female) 💰": "XB0fDUnXU5powFXDhCwa", 
    }
    
    selected_voice_name = st.selectbox(
        "Choose from predefined voices:",
        options=list(voice_options.keys()),
        index=0
    )
    selected_voice_id = voice_options[selected_voice_name]

# Voice speed and style settings
st.subheader("⚙️ Voice Settings")
st.info("💡 **How Settings Work**: All sliders affect ElevenLabs voice generation! Speed controls actual speech rate (0.7=slower, 1.0=normal, 1.2=fastest). API enforces 0.7-1.2 range.")

col1, col2 = st.columns(2)

with col1:
    speaking_speed = st.slider("Speaking Speed", 0.7, 1.2, 1.2, 0.1, help="ElevenLabs speech speed: 0.7=slower, 1.0=normal, 1.2=fastest (API limits: 0.7-1.2)")
    stability = st.slider("Stability", 0.0, 1.0, 0.7, 0.1, help="Voice consistency - higher = more stable/predictable")

with col2:
    similarity_boost = st.slider("Similarity Boost", 0.0, 1.0, 0.8, 0.1, help="How closely to match original voice - higher = more similar")
    style_exaggeration = st.slider("Style", 0.0, 1.0, 0.4, 0.1, help="Voice expressiveness - higher = more dramatic/emotional")

# Test text input
test_text = st.text_area(
    "Enter text to convert to speech:",
    value="Good afternoon. I'm delighted to present today's comprehensive analysis of your performance marketing metrics. The data reveals several fascinating insights that warrant immediate attention.",
    height=100
)

def generate_elevenlabs_audio(text, api_key, voice_id, voice_settings_dict):
    """Generate audio using ElevenLabs with detailed error handling"""
    try:
        st.write("🔄 Importing ElevenLabs...")
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
        st.success("✅ ElevenLabs library imported successfully")
        
        st.write("🔄 Creating ElevenLabs client...")
        client = ElevenLabs(api_key=api_key)
        st.success("✅ ElevenLabs client created")
        
        st.write(f"🔄 Generating audio with voice: {voice_id}")
        st.write(f"⚙️ Voice Settings - Stability: {voice_settings_dict['stability']:.1f}, Similarity: {voice_settings_dict['similarity_boost']:.1f}, Style: {voice_settings_dict['style']:.1f}")
        st.write(f"🏃‍♀️ Speech Speed: {voice_settings_dict['speed']:.1f}x (ElevenLabs VoiceSettings speed parameter)")
        
        # Create voice settings from the dictionary
        voice_settings = VoiceSettings(
            stability=voice_settings_dict['stability'],
            similarity_boost=voice_settings_dict['similarity_boost'],
            style=voice_settings_dict['style'],
            use_speaker_boost=True,
            speed=voice_settings_dict['speed']  # Speed parameter in VoiceSettings
        )
        
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            voice_settings=voice_settings,
            output_format="mp3_22050_32"
        )
        
        st.success("✅ Audio generated successfully")
        
        # Show audio details
        audio_bytes = b"".join(audio)
        st.write(f"📊 Audio size: {len(audio_bytes):,} bytes")
        
        # Apply speed adjustment by modifying playback rate if needed
        if voice_settings_dict['speed'] != 1.0:
            st.info(f"🏃‍♀️ Audio will play at {voice_settings_dict['speed']:.1f}x speed")
        
        return audio_bytes
        
    except ImportError as e:
        st.error(f"❌ Import Error: {str(e)}")
        st.write("Try installing: pip install elevenlabs")
        return None
    except Exception as e:
        st.error(f"❌ ElevenLabs Error: {str(e)}")
        return None

if st.button("🎵 Generate Voice", type="primary"):
    if test_text.strip():
        # Prepare voice settings
        voice_settings_dict = {
            'speed': speaking_speed,
            'stability': stability,
            'similarity_boost': similarity_boost,
            'style': style_exaggeration
        }
        
        with st.spinner(f"Generating audio with {selected_voice_name}..."):
            audio_data = generate_elevenlabs_audio(test_text, elevenlabs_api_key, selected_voice_id, voice_settings_dict)
            
            if audio_data:
                st.success("🎉 Audio generated successfully!")
                
                # Method 1: Standard Streamlit audio
                st.subheader("🔊 Method 1: Standard Streamlit Audio")
                st.audio(audio_data, format='audio/mp3')
                
                # Method 2: HTML audio with controls
                st.subheader("🔊 Method 2: HTML Audio with Controls")
                audio_base64 = base64.b64encode(audio_data).decode()
                audio_html = f'''
                <audio controls style="width: 100%;">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                '''
                st.markdown(audio_html, unsafe_allow_html=True)
                
                # Method 2b: Auto-play version with speed control
                st.subheader("🔊 Method 2b: Auto-Play Audio (Speed Controlled)")
                st.write("⚠️ Note: Auto-play may be blocked by browser security policies")
                st.write("🏃‍♀️ Playing at 1.6x speed (hard coded)")
                autoplay_html = f'''
                <audio controls autoplay style="width: 100%;" onloadstart="this.playbackRate = 1.6;">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                <script>
                    // Set playback speed when audio loads
                    document.addEventListener('DOMContentLoaded', function() {{
                        const audios = document.querySelectorAll('audio[autoplay]');
                        audios.forEach(function(audio) {{
                            audio.playbackRate = 1.6;
                        }});
                    }});
                </script>
                '''
                st.markdown(autoplay_html, unsafe_allow_html=True)
                
                # Method 2c: JavaScript Auto-play with speed
                st.subheader("🔊 Method 2c: JavaScript Auto-Play (Enhanced)")
                st.write("🎯 This attempts to play immediately at 1.6x speed")
                js_autoplay_html = f'''
                <audio id="autoPlayAudio" controls style="width: 100%;">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                <script>
                    const audio = document.getElementById('autoPlayAudio');
                    audio.playbackRate = 1.6;
                    audio.play().catch(function(error) {{
                        console.log('Auto-play blocked by browser:', error);
                    }});
                </script>
                '''
                st.markdown(js_autoplay_html, unsafe_allow_html=True)
                
                # Method 3: Download link
                st.subheader("💾 Method 3: Download Audio File")
                st.download_button(
                    label="📥 Download MP3",
                    data=audio_data,
                    file_name="test_voice.mp3",
                    mime="audio/mp3"
                )
                
                # Debug info
                st.subheader("🔍 Debug Information")
                st.write(f"**Audio format:** MP3")
                st.write(f"**Audio size:** {len(audio_data):,} bytes")
                st.write(f"**Text length:** {len(test_text)} characters")
                
            else:
                st.error("❌ Failed to generate audio")
    else:
        st.warning("⚠️ Please enter some text to convert")

# Troubleshooting tips
st.markdown("---")
st.subheader("🛠️ Troubleshooting Tips")

with st.expander("💰 ElevenLabs Subscription Tiers", expanded=False):
    st.write("**Free Tier (Current)**")
    st.write("• ✅ Access to 10 pre-made voices (marked with 🆓)")
    st.write("• ✅ 10,000 characters per month")
    st.write("• ❌ No access to premium voices like Shannon")
    st.write("")
    st.write("**Creator Tier ($5/month)**")
    st.write("• ✅ All pre-made voices including Shannon")
    st.write("• ✅ 30,000 characters per month")
    st.write("• ✅ Voice cloning features")
    st.write("")
    st.write("**Pro Tier ($22/month)**")
    st.write("• ✅ Everything in Creator")
    st.write("• ✅ 100,000 characters per month")
    st.write("• ✅ Professional voice cloning")

with st.expander("🎧 Audio Not Playing?", expanded=False):
    st.write("**Common issues and solutions:**")
    st.write("1. **Browser Audio Settings**")
    st.write("   • Check if your browser allows audio playback")
    st.write("   • Look for a speaker icon in the browser tab - click to unmute")
    st.write("   • Try refreshing the page")
    
    st.write("2. **Auto-Play Blocked**")
    st.write("   • Most browsers block auto-play until user interaction")
    st.write("   • Click anywhere on the page first, then try again")
    st.write("   • Look for auto-play blocking notifications in your browser")
    st.write("   • In Chrome: Click the 🔒 icon → Site Settings → Sound → Allow")
    
    st.write("3. **System Volume**")
    st.write("   • Check your system volume is not muted")
    st.write("   • Check your browser volume settings")
    
    st.write("4. **Browser Compatibility**")
    st.write("   • Try a different browser (Chrome, Firefox, Safari)")
    st.write("   • Clear browser cache and cookies")
    
    st.write("5. **Network Issues**")
    st.write("   • Check your internet connection")
    st.write("   • Try downloading the MP3 file instead")

with st.expander("🔧 Technical Details", expanded=False):
    st.write("**Current Voice Settings:**")
    st.write(f"**Selected Voice:** {selected_voice_name}")
    st.write(f"**Voice ID:** {selected_voice_id}")
    st.code(f"""
Model: eleven_multilingual_v2
Speaking Speed: {speaking_speed:.1f}x
Stability: {stability:.1f} (consistency vs naturalness)
Similarity Boost: {similarity_boost:.1f} (voice character strength)
Style: {style_exaggeration:.1f} (expressiveness level)
Speaker Boost: True (enhanced clarity)
Format: MP3 22050Hz 32kbps
    """)
    
    st.write("**Available Voice Personalities:**")
    st.write("• **Shannon**: Expressive, dynamic female voice with natural flow")
    st.write("• **Charlotte**: Educated, sophisticated British accent")
    st.write("• **Alice**: Young, clear British female voice")  
    st.write("• **Lily**: Sophisticated British accent with professional tone")
    st.write("• **Bella**: Professional, clear American female voice")

# Test browser audio permissions
st.markdown("---")
st.subheader("🎤 Browser Audio Test")
st.write("Test if your browser can play audio:")

test_audio_html = '''
<audio controls>
    <source src="https://www.soundjay.com/misc/sounds/bell-ringing-05.wav" type="audio/wav">
    Your browser does not support the audio element.
</audio>
'''
st.markdown(test_audio_html, unsafe_allow_html=True)
st.caption("If you can hear the bell sound above, your browser audio is working correctly.") 