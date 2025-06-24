import pyttsx3
import time
import os
import tempfile
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import pygame

# --- Agent Definitions ---

class Agent:
    def analyze(self, data):
        raise NotImplementedError

class BuyerPerformanceAgent(Agent):
    def analyze(self, data):
        buyers = data.get("buyers", [])
        findings = []
        for buyer in buyers:
            if buyer["availability"] < 0.9:
                findings.append(
                    f"{buyer['name']} shows {int((1-buyer['availability'])*100)}% agent availability issues "
                    f"with quote-to-sale rate {buyer['quote_to_sale']*100:.1f}%."
                )
        return {"buyer_findings": findings}

class PublisherQualityAgent(Agent):
    def analyze(self, data):
        publishers = data.get("publishers", [])
        findings = []
        for pub in publishers:
            if pub["ad_misled_rate"] > 0.1:
                findings.append(
                    f"{pub['name']} has {pub['ad_misled_rate']*100:.1f}% ad misled rate but "
                    f"{pub['billable_rate']*100:.1f}% billable rate."
                )
        return {"publisher_findings": findings}

class ComparativeAnalysisAgent(Agent):
    def analyze(self, data):
        leads = data.get("leads", [])
        findings = []
        for lead in leads:
            if lead["intent_level_2_3"] > 0.4 and lead["conversion"] < 0.02:
                findings.append(
                    f"{lead['name']} generates {lead['intent_level_2_3']*100:.1f}% Level 2/3 intent leads "
                    f"but converts at only {lead['conversion']*100:.1f}%."
                )
        return {"comparative_findings": findings}

class ExecutiveSummaryAgent(Agent):
    def analyze(self, data):
        # No-op for compatibility with orchestrator
        return {}
    def synthesize(self, results):
        summary = []
        if results.get("buyer_performance", {}).get("buyer_findings"):
            summary.append("Buyer Agent: " + "; ".join(results["buyer_performance"]["buyer_findings"]))
        if results.get("publisher_quality", {}).get("publisher_findings"):
            summary.append("Publisher Agent: " + "; ".join(results["publisher_quality"]["publisher_findings"]))
        if results.get("comparative_analysis", {}).get("comparative_findings"):
            summary.append("Comparative Agent: " + "; ".join(results["comparative_analysis"]["comparative_findings"]))
        if not summary:
            return "No critical issues detected."
        summary.append("Executive Summary: High-quality leads are being wasted due to sales execution gaps. Immediate training focus needed.")
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

# --- Voice & Animation Functions ---

def transcribe_voice_input():
    # In real use, replace with actual STT
    print("User (voice): Why is my conversion rate dropping?")
    return "Why is my conversion rate dropping?"

def generate_voice_response(text):
    print("\n[Voice Output]:")
    print(text)
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        print("[Error] ELEVENLABS_API_KEY not found in environment variables.")
        return None
    client = ElevenLabs(api_key=api_key)
    # Use Rachel (free, professional female) as default
    voice_id = "21m00Tcm4TlvDq8ikWAM"
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
    audio_bytes = b"".join(audio)
    # Save to a temporary file and play with pygame
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"[Error] Could not play audio: {e}")
    finally:
        os.remove(tmp_path)
    return None

def speak_with_animation(text, tts_audio):
    print("\n[Animation]: Speaking animation triggered.")
    for _ in range(3):
        print("[Animation]: ...speaking...")
        time.sleep(0.5)
    print("[Animation]: Animation ended.")

# --- Example Data ---

performance_data = {
    "buyers": [
        {"name": "HPOne", "availability": 0.86, "quote_to_sale": 0.23},
        {"name": "OtherBuyer", "availability": 0.95, "quote_to_sale": 0.18},
    ],
    "publishers": [
        {"name": "MMX-MED-IB-BID-LD-1", "ad_misled_rate": 0.161, "billable_rate": 0.776},
        {"name": "GoodPublisher", "ad_misled_rate": 0.05, "billable_rate": 0.85},
    ],
    "leads": [
        {"name": "WEG-MED-IB-CPL-0-1", "intent_level_2_3": 0.5, "conversion": 0.016},
        {"name": "OtherLead", "intent_level_2_3": 0.2, "conversion": 0.03},
    ]
}

# --- Main Flow ---

agents = {
    "buyer_performance": BuyerPerformanceAgent(),
    "publisher_quality": PublisherQualityAgent(),
    "comparative_analysis": ComparativeAnalysisAgent(),
    "executive_summary": ExecutiveSummaryAgent(),
}
orchestrator = Orchestrator(agents)

def handle_user_voice_query(data, orchestrator):
    user_query = transcribe_voice_input()
    focus_areas = None
    results = orchestrator.run_comprehensive_analysis(data, focus_areas)
    voice_summary = orchestrator.get_summary_for_voice(results)
    tts_audio = generate_voice_response(voice_summary)
    speak_with_animation(voice_summary, tts_audio)

# --- Run the Example ---

if __name__ == "__main__":
    handle_user_voice_query(performance_data, orchestrator) 