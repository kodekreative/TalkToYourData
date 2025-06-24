"""
Lead Management Analyst
Specialized agent for analyzing marketing lead data and performance
Enhanced with GPT-4o for intelligent analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from openai import OpenAI

from ..data.analyzer import MarketingDataAnalyzer

class LeadManagementAnalyst:
    """
    Lead Management Analyst - Analyzes marketing performance data with AI intelligence
    """
    
    def __init__(self, data: pd.DataFrame = None):
        self.data = data
        self.analyzer = MarketingDataAnalyzer() if data is not None else None
        
        # Initialize AI clients (OpenAI and Claude)
        self.ai_clients = self._init_openai()
        
        # Legacy compatibility
        self.openai_client = self.ai_clients.get('openai') if self.ai_clients else None
        
    def _init_openai(self):
        """Initialize AI clients (OpenAI and Claude)"""
        clients = {}
        
        # Initialize OpenAI client
        try:
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key and openai_key != "your_openai_api_key_here":
                clients['openai'] = OpenAI(api_key=openai_key)
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")
        
        # Initialize Claude client
        try:
            import anthropic
            claude_key = os.getenv('ANTHROPIC_API_KEY')
            if claude_key and claude_key != "your_anthropic_api_key_here":
                clients['claude'] = anthropic.Anthropic(api_key=claude_key)
                print("✅ Claude 3.5 Sonnet client initialized successfully")
        except ImportError:
            print("Warning: anthropic package not installed. Install with: pip install anthropic")
        except Exception as e:
            print(f"Warning: Could not initialize Claude client: {e}")
        
        return clients
    
    def _load_custom_prompt(self):
        """Load custom prompt from prompts_config.md file"""
        try:
            import re
            with open('prompts_config.md', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract Lead Management Analyst prompt
            match = re.search(r'## Lead Management Analyst Prompt\s*```(.*?)```', content, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                print("Warning: Custom prompt not found, using fallback")
                return self._get_fallback_prompt()
                
        except FileNotFoundError:
            print("Warning: prompts_config.md not found, using fallback prompt")
            return self._get_fallback_prompt()
        except Exception as e:
            print(f"Warning: Error loading custom prompt: {e}")
            return self._get_fallback_prompt()
    
    def _get_fallback_prompt(self):
        """Fallback prompt if custom prompt can't be loaded"""
        return """
You are a Lead Management Analyst AI agent that analyzes marketing performance data with expertise in publisher evaluation, lead quality assessment, and performance optimization.

ANALYSIS REQUEST: {query}
ANALYSIS TYPE: {analysis_type}

CURRENT PERFORMANCE DATA:
{data_summary}

CRITICAL BUSINESS CONTEXT - CUSTOMER INTENT LEVELS:
- Level 3: HIGH INTENT (premium leads, maximum conversion potential, highest ROI, priority follow-up required)
- Level 2: MEDIUM INTENT (moderate-quality leads, good conversion potential, acceptable ROI)  
- Level 1: LOW INTENT (low-quality leads, minimal conversion potential, high cost-per-acquisition)

PERFORMANCE HIERARCHY: Level 3 > Level 2 > Level 1

Provide a comprehensive business analysis in JSON format with insights, recommendations, performance analysis, cost analysis, trending analysis, and confidence assessment.

Required JSON structure:
{{
    "insights": ["Provide 10-15 specific, quantified business insights"],
    "recommendations": [
        {{
            "category": "Immediate",
            "recommendation": "Specific action",
            "expected_impact": "Quantified improvement",
            "implementation_steps": "Detailed action plan"
        }}
    ],
    "confidence": "Confidence level with explanation"
}}
"""
    
    def load_data(self, data: pd.DataFrame):
        """Load new data into the analyst"""
        self.data = data
        self.analyzer = MarketingDataAnalyzer()
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a natural language query about marketing performance
        Enhanced with GPT-4o for intelligent insights
        """
        if self.data is None:
            return {
                'error': 'No data loaded',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Detect if this is an extensive analysis request
            extensive_mode = self._detect_extensive_mode(query)
            
            # Determine analysis type
            analysis_type = self._determine_analysis_type(query)
            
            # Get base data analysis
            base_analysis = self._get_base_analysis(analysis_type)
            
            if "error" in base_analysis:
                return base_analysis
            
            # Enhanced analysis with AI (Claude 3.5 Sonnet preferred, GPT fallback)
            if (self.ai_clients and extensive_mode):
                print(f"DEBUG: Using AI for extensive analysis. Query length: {len(query)}")
                
                # Try Claude 3.5 Sonnet first (better at following complex prompts)
                enhanced_analysis = None
                if 'claude' in self.ai_clients:
                    print("DEBUG: Trying Claude 3.5 Sonnet...")
                    enhanced_analysis = self._get_claude_analysis(query, base_analysis, analysis_type)
                    
                # Fallback to GPT if Claude fails
                if not enhanced_analysis and 'openai' in self.ai_clients:
                    print("DEBUG: Claude failed, trying GPT models...")
                    enhanced_analysis = self._get_gpt4o_analysis(query, base_analysis, analysis_type)
                
                if enhanced_analysis:
                    print("DEBUG: AI analysis successful")
                    # Merge base data with AI insights
                    return {
                        **base_analysis,
                        'insights': enhanced_analysis.get('insights', base_analysis.get('insights', [])),
                        'recommendations': enhanced_analysis.get('recommendations', base_analysis.get('recommendations', [])),
                        'business_context': enhanced_analysis.get('business_context', ''),
                        'strategic_implications': enhanced_analysis.get('strategic_implications', ''),
                        'confidence': enhanced_analysis.get('confidence', 'high'),
                        'extensive_mode': True,
                        'ai_enhanced': True
                    }
                else:
                    print("DEBUG: GPT-4o analysis failed, returning incomplete analysis indicator")
                    # Return incomplete analysis indicator instead of fictitious data
                    return {
                        **base_analysis,
                        'insights': [
                            "⚠️ ANALYSIS INCOMPLETE: GPT-4o analysis failed due to technical issues.",
                            "❌ Unable to provide comprehensive business insights at this time.",
                            "🔄 Please try again or contact support if the issue persists.",
                            f"📊 Basic data shows: {len(self.data) if self.data is not None else 0:,} records processed"
                        ],
                        'recommendations': [
                            "Retry the analysis to get complete GPT-4o powered insights.",
                            "Check system status and ensure proper API connectivity.",
                            "Use the basic analysis dropdown options for immediate insights."
                        ],
                        'business_context': "Analysis could not be completed due to technical limitations. GPT-4o failed to process the request.",
                        'strategic_implications': "Complete analysis required for strategic decision-making. Please retry.",
                        'confidence': "incomplete_analysis",
                        'extensive_mode': True,
                        'ai_enhanced': False,
                        'analysis_status': 'failed'
                    }
            else:
                print(f"DEBUG: Not using GPT-4o. OpenAI client: {self.openai_client is not None}, Extensive mode: {extensive_mode}")
                # For non-extensive mode, provide basic analysis
                return self._enhance_base_analysis(base_analysis, query, extensive_mode)
            
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_claude_analysis(self, query: str, base_analysis: Dict, analysis_type: str) -> Optional[Dict]:
        """
        Use Claude 3.5 Sonnet to generate intelligent business insights using custom prompts
        Claude is generally better at following complex business analysis requirements
        """
        try:
            claude_client = self.ai_clients.get('claude')
            if not claude_client:
                return None
                
            # Load custom prompt from markdown file
            prompt_template = self._load_custom_prompt()
            print(f"DEBUG: Custom prompt loaded for Claude, length: {len(prompt_template)} characters")
            print(f"DEBUG: Prompt contains JSON requirements: {'RETURN ONLY VALID JSON' in prompt_template}")
            
            # Prepare data summary for Claude with correct intent level context
            data_summary = self._prepare_data_summary(base_analysis)
            
            # Format the prompt with actual values
            user_prompt = prompt_template.format(
                query=query,
                analysis_type=analysis_type,
                data_summary=data_summary
            )
            print(f"DEBUG: Final prompt length for Claude: {len(user_prompt)} characters")
            
            # Claude system message - more explicit about JSON requirements
            system_message = """You are a Lead Management Analyst AI. You MUST follow the detailed custom prompt requirements exactly.

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON format - no markdown, no explanations, no code blocks
2. Follow the exact JSON structure specified in the prompt
3. Include all required sections: performance_analysis, cost_analysis, trending_analysis
4. Provide detailed business insights with specific metrics and ROI calculations
5. Start response with { and end with }

Your response must be pure JSON that can be parsed directly."""

            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.1,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response_content = response.content[0].text
            print(f"DEBUG: Claude response (first 500 chars): {response_content[:500]}")
            
            # Clean the response - Claude is usually better at following JSON instructions
            cleaned_content = response_content.strip()
            
            # Remove any markdown code blocks if present (though Claude usually doesn't add them)
            if '```json' in cleaned_content:
                start = cleaned_content.find('```json') + 7
                end = cleaned_content.find('```', start)
                if end != -1:
                    cleaned_content = cleaned_content[start:end].strip()
                else:
                    cleaned_content = cleaned_content[start:].strip()
            
            # Find JSON object boundaries
            first_brace = cleaned_content.find('{')
            last_brace = cleaned_content.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_content = cleaned_content[first_brace:last_brace+1]
            else:
                json_content = cleaned_content
            
            import json
            result = json.loads(json_content)
            print("DEBUG: Successfully parsed Claude JSON response")
            
            # Validate Claude's response structure
            required_sections = ['performance_analysis', 'cost_analysis', 'trending_analysis']
            missing_sections = [section for section in required_sections if section not in result]
            
            insights = result.get('insights', [])
            recommendations = result.get('recommendations', [])
            
            print(f"DEBUG: Claude prompt structure validation:")
            print(f"  - Required sections present: {len(required_sections) - len(missing_sections)}/{len(required_sections)}")
            print(f"  - Missing sections: {missing_sections}")
            print(f"  - Total insights: {len(insights)}")
            print(f"  - Total recommendations: {len(recommendations)}")
            
            if missing_sections:
                print(f"WARNING: Claude did not return all required sections!")
                print(f"WARNING: Missing: {missing_sections}")
                return None
            
            print("=" * 60)
            print("DEBUG: LEAD ANALYST OUTPUT (Claude 3.5 Sonnet):")
            print("=" * 60)
            print(f"INSIGHTS ({len(insights)} total):")
            for i, insight in enumerate(insights[:5], 1):  # Show first 5
                if isinstance(insight, dict):
                    print(f"  {i}. {insight.get('insight', insight)}")
                else:
                    print(f"  {i}. {insight}")
            if len(insights) > 5:
                print(f"  ... and {len(insights) - 5} more insights")
                
            print(f"\nRECOMMENDATIONS ({len(recommendations)} total):")
            for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
                if isinstance(rec, dict):
                    print(f"  {i}. {rec.get('recommendation', rec)}")
                    print(f"     Expected ROI: {rec.get('expected_ROI', 'Not specified')}")
                else:
                    print(f"  {i}. {rec}")
            if len(recommendations) > 3:
                print(f"  ... and {len(recommendations) - 3} more recommendations")
                
            print(f"\nBUSINESS CONTEXT: {result.get('business_context', 'Provided')[:100]}...")
            print(f"STRATEGIC IMPLICATIONS: {result.get('strategic_implications', 'Provided')[:100]}...")
            print(f"CONFIDENCE: {result.get('confidence', 'Not specified')}")
            print("=" * 60)
            
            return result
            
        except Exception as e:
            print(f"Claude analysis failed: {e}")
            return None

    def _get_gpt4o_analysis(self, query: str, base_analysis: Dict, analysis_type: str) -> Optional[Dict]:
        """
        Use GPT models to generate intelligent business insights using custom prompts
        """
        try:
            # Load custom prompt from markdown file
            prompt_template = self._load_custom_prompt()
            print(f"DEBUG: Custom prompt loaded, length: {len(prompt_template)} characters")
            print(f"DEBUG: Prompt contains JSON requirements: {'RETURN ONLY VALID JSON' in prompt_template}")
            
            # Prepare data summary for GPT-4o with correct intent level context
            data_summary = self._prepare_data_summary(base_analysis)
            
            # Format the prompt with actual values
            prompt = prompt_template.format(
                query=query,
                analysis_type=analysis_type,
                data_summary=data_summary
            )
            print(f"DEBUG: Final prompt length: {len(prompt)} characters")
            
            # Use a simple system message that doesn't conflict with the custom prompt
            system_message = "You are a Lead Management Analyst. Follow the detailed requirements in the user message exactly. Return ONLY valid JSON format - no markdown, no explanatory text, no code blocks."

            # Try multiple models in order of preference for business analysis
            models_to_try = [
                {"name": "gpt-4", "max_tokens": 4000, "temperature": 0.1},  # More reliable for complex prompts
                {"name": "gpt-4-turbo", "max_tokens": 4000, "temperature": 0.1},  # Faster alternative
                {"name": "gpt-4o", "max_tokens": 4000, "temperature": 0.05},  # Current model
                {"name": "gpt-3.5-turbo", "max_tokens": 3000, "temperature": 0.2}  # Fallback
            ]
            
            response = None
            last_error = None
            
            for model_config in models_to_try:
                try:
                    print(f"DEBUG: Trying model: {model_config['name']}")
                    response = self.openai_client.chat.completions.create(
                        model=model_config["name"],
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=model_config["temperature"],
                        max_tokens=model_config["max_tokens"]
                    )
                    print(f"DEBUG: Successfully used model: {model_config['name']}")
                    break
                except Exception as e:
                    print(f"DEBUG: Model {model_config['name']} failed: {str(e)}")
                    last_error = e
                    continue
            
            if response is None:
                raise Exception(f"All models failed. Last error: {last_error}")
            
            response_content = response.choices[0].message.content
            print(f"DEBUG: GPT-4o response (first 500 chars): {response_content[:500]}")
            
            # Clean the response - extract JSON from various formats
            cleaned_content = response_content.strip()
            
            # Remove markdown code blocks if present
            if '```json' in cleaned_content:
                # Extract JSON from markdown code block
                start = cleaned_content.find('```json') + 7
                end = cleaned_content.find('```', start)
                if end != -1:
                    cleaned_content = cleaned_content[start:end].strip()
                else:
                    cleaned_content = cleaned_content[start:].strip()
            
            # Find JSON object boundaries
            first_brace = cleaned_content.find('{')
            last_brace = cleaned_content.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_content = cleaned_content[first_brace:last_brace+1]
            else:
                json_content = cleaned_content
            
            import json
            result = json.loads(json_content)
            print("DEBUG: Successfully parsed GPT-4o JSON response")
            
            # Validate if GPT-4o returned the detailed structure required by custom prompt
            required_sections = ['performance_analysis', 'cost_analysis', 'trending_analysis']
            missing_sections = [section for section in required_sections if section not in result]
            
            insights = result.get('insights', [])
            recommendations = result.get('recommendations', [])
            
            print(f"DEBUG: Custom prompt structure validation:")
            print(f"  - Required sections present: {len(required_sections) - len(missing_sections)}/{len(required_sections)}")
            print(f"  - Missing sections: {missing_sections}")
            print(f"  - Total insights: {len(insights)}")
            print(f"  - Total recommendations: {len(recommendations)}")
            
            # If missing key sections, this means GPT-4o didn't follow the custom prompt structure
            if missing_sections:
                print(f"WARNING: GPT-4o did not return the detailed JSON structure required by custom prompt!")
                print(f"WARNING: Missing: {missing_sections}")
                print(f"WARNING: This indicates GPT-4o is ignoring your custom prompt requirements!")
                
                # Return incomplete analysis indicator since GPT-4o didn't follow the prompt
                return {
                    'insights': [
                        f"⚠️ GPT-4o ANALYSIS INCOMPLETE: AI returned simplified format instead of detailed business analysis.",
                        f"❌ Missing required sections: {', '.join(missing_sections)}",
                        f"🔧 GPT-4o ignored your custom prompt requirements for detailed performance analysis.",
                        f"📊 Basic insights generated: {len(insights)} simple insights instead of comprehensive analysis."
                    ],
                    'recommendations': [
                        "Check custom prompt formatting in prompts_config.md",
                        "GPT-4o may need more specific instructions or examples",
                        "Consider simplifying custom prompt requirements"
                    ],
                    'business_context': f"GPT-4o failed to follow custom prompt structure. Returned simplified JSON missing: {', '.join(missing_sections)}",
                    'strategic_implications': "Complete detailed analysis required for strategic decision-making. Current output is insufficient.",
                    'confidence': "incomplete_analysis",
                    'analysis_status': 'failed_custom_prompt',
                    'gpt4o_response_summary': f"Returned {len(insights)} insights and {len(recommendations)} recommendations but missing detailed structure"
                }
            
            print("=" * 60)
            print("DEBUG: LEAD ANALYST OUTPUT (GPT-4o):")
            print("=" * 60)
            print(f"INSIGHTS ({len(result.get('insights', []))} total):")
            for i, insight in enumerate(result.get('insights', []), 1):
                print(f"  {i}. {insight}")
            print(f"\nRECOMMENDATIONS ({len(result.get('recommendations', []))} total):")
            for i, rec in enumerate(result.get('recommendations', []), 1):
                if isinstance(rec, dict):
                    print(f"  {i}. {rec.get('recommendation', 'No recommendation text')}")
                    print(f"     Category: {rec.get('category', 'Unknown')}")
                    print(f"     Implementation: {rec.get('implementation_steps', 'No steps')}")
                    print(f"     Expected ROI: {rec.get('expected_ROI', 'No ROI specified')}")
                else:
                    print(f"  {i}. {rec}")
            print(f"\nBUSINESS CONTEXT:")
            print(f"  {result.get('business_context', 'No business context provided')}")
            print(f"\nSTRATEGIC IMPLICATIONS:")
            print(f"  {result.get('strategic_implications', 'No strategic implications provided')}")
            print(f"\nCONFIDENCE:")
            print(f"  {result.get('confidence', 'No confidence level provided')}")
            print("=" * 60)
            return result
            
        except Exception as e:
            print(f"GPT-4o analysis failed: {e}")
            print(f"DEBUG: Prompt sent to GPT-4o:")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print(f"DEBUG: Data summary:")
            print(data_summary[:300] + "..." if len(data_summary) > 300 else data_summary)
            return None
    
    def _prepare_data_summary(self, base_analysis: Dict) -> str:
        """Prepare a comprehensive data summary for GPT-4o with detailed publisher analysis"""
        summary_parts = []
        
        # CRITICAL: Add intent level definitions at the top
        summary_parts.append("INTENT LEVEL DEFINITIONS:")
        summary_parts.append("- Level 1: POOR INTENT (low-quality leads)")
        summary_parts.append("- Level 2: MEDIUM TO HIGH INTENT (moderate-quality leads)")  
        summary_parts.append("- Level 3: BEST INTENT (highest-quality leads)")
        summary_parts.append("- Not Detected: Intent level could not be determined")
        summary_parts.append("- Negative Intent: Explicitly negative or uninterested")
        summary_parts.append("")
        
        # ENHANCED: Detailed publisher-level data for custom prompt requirements
        if self.data is not None:
            summary_parts.append("DETAILED PUBLISHER PERFORMANCE DATA:")
            summary_parts.append("=" * 50)
            
            # Publisher-level analysis with volume thresholds
            publisher_stats = self.data.groupby('PUBLISHER').agg({
                'SALE': lambda x: (x.astype(str).str.upper() == 'YES').sum(),
                'CUSTOMER_INTENT': 'count',
                'AD_MISLED': lambda x: (x.astype(str).str.upper() == 'YES').sum()
            }).rename(columns={
                'SALE': 'total_sales',
                'CUSTOMER_INTENT': 'total_calls',
                'AD_MISLED': 'ad_misled_count'
            })
            
            publisher_stats['conversion_rate'] = (publisher_stats['total_sales'] / publisher_stats['total_calls'] * 100).round(2)
            publisher_stats['ad_misled_rate'] = (publisher_stats['ad_misled_count'] / publisher_stats['total_calls'] * 100).round(2)
            
            # Apply volume thresholds as per custom prompt
            qualifying_publishers = publisher_stats[publisher_stats['total_calls'] >= 30]
            
            summary_parts.append(f"PUBLISHERS MEETING VOLUME THRESHOLD (≥30 calls):")
            for publisher, stats in qualifying_publishers.iterrows():
                summary_parts.append(f"  {publisher}:")
                summary_parts.append(f"    - Total Calls: {stats['total_calls']:,}")
                summary_parts.append(f"    - Total Sales: {stats['total_sales']:,}")
                summary_parts.append(f"    - Conversion Rate: {stats['conversion_rate']:.1f}%")
                summary_parts.append(f"    - Ad Misled Rate: {stats['ad_misled_rate']:.1f}%")
                
                # Intent distribution for this publisher
                publisher_data = self.data[self.data['PUBLISHER'] == publisher]
                intent_dist = publisher_data['CUSTOMER_INTENT'].value_counts(normalize=True) * 100
                summary_parts.append(f"    - Intent Distribution:")
                for intent, pct in intent_dist.items():
                    summary_parts.append(f"      * {intent}: {pct:.1f}%")
                summary_parts.append("")
            
            # Publishers below threshold
            below_threshold = publisher_stats[publisher_stats['total_calls'] < 30]
            if len(below_threshold) > 0:
                summary_parts.append(f"PUBLISHERS BELOW VOLUME THRESHOLD (<30 calls): {len(below_threshold)} publishers")
                summary_parts.append("(Excluded from detailed analysis per volume requirements)")
                summary_parts.append("")
            
            # Overall statistics
            summary_parts.append("OVERALL PERFORMANCE METRICS:")
            summary_parts.append(f"- Total Records: {len(self.data):,}")
            summary_parts.append(f"- Total Publishers: {self.data['PUBLISHER'].nunique()}")
            summary_parts.append(f"- Publishers Meeting Volume Threshold: {len(qualifying_publishers)}")
            
            # Overall intent distribution
            overall_intent = self.data['CUSTOMER_INTENT'].value_counts(normalize=True) * 100
            summary_parts.append("- Overall Intent Distribution:")
            for intent, pct in overall_intent.items():
                summary_parts.append(f"  * {intent}: {pct:.1f}%")
                
            # Overall conversion rate
            overall_sales = (self.data['SALE'].astype(str).str.upper() == 'YES').sum()
            overall_conversion = (overall_sales / len(self.data) * 100)
            summary_parts.append(f"- Overall Conversion Rate: {overall_conversion:.1f}%")
            
            # Performance index calculations (baseline for custom prompt)
            avg_conversion = qualifying_publishers['conversion_rate'].mean()
            summary_parts.append(f"- Average Conversion Rate (Qualifying Publishers): {avg_conversion:.1f}%")
            summary_parts.append("- Performance Index Baseline: 100 = Average Performance")
            
            summary_parts.append("")
            summary_parts.append("ANALYSIS REQUIREMENTS:")
            summary_parts.append("- Apply volume thresholds: 30+ calls for publisher analysis")
            summary_parts.append("- Calculate performance indices relative to average")
            summary_parts.append("- Identify statistical outliers (>2 standard deviations)")
            summary_parts.append("- Provide variance analysis for top performers")
            summary_parts.append("- Answer all 5 business question categories")
        
        return '\n'.join(filter(None, summary_parts))
    
    def _detect_extensive_mode(self, query: str) -> bool:
        """
        Detect if extensive analysis is requested
        """
        extensive_indicators = [
            'extensive', 'comprehensive', 'detailed', 'in-depth', 'thorough',
            'Analysis Depth Required:', 'REPORT FORMAT:', 'ANALYSIS FOCUS:',
            'Key Metrics to Analyze:', 'minimum 1000 words', 'detailed analysis',
            'Business Questions:', 'strategic', 'executive', 'board presentation',
            'Lead Quality Analyst', 'Publisher Performance Analyst', 'Cost Efficiency Analyst',
            'Call Quality Analyst', 'Conversion Funnel Analyst', 'guided analysis',
            'As a', 'perform an extensive analysis', 'comprehensive analysis'
        ]
        
        query_lower = query.lower()
        # Always use extensive mode for guided analysis or when OpenAI is available
        is_extensive = any(indicator.lower() in query_lower for indicator in extensive_indicators)
        
        # Force extensive mode if this looks like a guided analysis query
        if ('as a' in query_lower and 'analyst' in query_lower) or len(query) > 200:
            is_extensive = True
            
        return is_extensive
    
    def _determine_analysis_type(self, query: str) -> str:
        """
        Determine the type of analysis based on query content
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['intent', 'level 1', 'level 2', 'level 3', 'quality']):
            return 'intent_analysis'
        elif any(word in query_lower for word in ['publisher', 'source', 'channel', 'performance']):
            return 'publisher_performance'
        elif any(word in query_lower for word in ['cost', 'roi', 'revenue', 'profit', 'budget']):
            return 'cost_analysis'
        elif any(word in query_lower for word in ['outlier', 'anomaly', 'unusual']):
            return 'outlier_analysis'
        elif any(word in query_lower for word in ['conversion', 'funnel', 'pipeline']):
            return 'conversion_analysis'
        else:
            return 'summary_metrics'
    
    def _get_base_analysis(self, analysis_type: str) -> Dict[str, Any]:
        """Get base statistical analysis"""
        try:
            if analysis_type == 'intent_analysis':
                return {
                    'analysis_type': 'intent_analysis',
                    'data': self.analyzer.calculate_intent_percentages(self.data),
                    'timestamp': datetime.now().isoformat()
                }
            elif analysis_type == 'publisher_performance':
                performance_data = self.analyzer.analyze_publisher_performance(self.data)
                return {
                    'analysis_type': 'publisher_performance',
                    'data': performance_data.to_dict('records') if isinstance(performance_data, pd.DataFrame) else performance_data,
                    'timestamp': datetime.now().isoformat()
                }
            elif analysis_type == 'cost_analysis':
                return {
                    'analysis_type': 'cost_analysis',
                    'data': self.analyzer.analyze_cost_efficiency(self.data),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Summary metrics
                return {
                    'analysis_type': 'summary_metrics',
                    'data': self.analyzer.get_summary_metrics(self.data),
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            return {
                'error': f'Base analysis failed: {str(e)}',
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat()
            }
    
    def _enhance_base_analysis(self, base_analysis: Dict, query: str, extensive_mode: bool) -> Dict[str, Any]:
        """
        Enhance base analysis with rule-based insights when GPT-4o is not available
        """
        analysis_type = base_analysis.get('analysis_type', 'general')
        data = base_analysis.get('data', {})
        
        insights = []
        recommendations = []
        
        # Generate insights based on analysis type
        if analysis_type == 'intent_analysis' and isinstance(data, dict):
            level_1 = data.get('Level 1', 0)
            level_2 = data.get('Level 2', 0)
            level_3 = data.get('Level 3', 0)
            
            insights.extend([
                f"Lead quality analysis shows {level_1:.1f}% Level 1 prospects (high-intent)",
                f"Combined quality score: {level_1 + level_2:.1f}% (Level 1 + Level 2)",
                f"Low-intent traffic: {level_3:.1f}% requires nurturing or filtering"
            ])
            
            if level_1 < 25:
                recommendations.append("URGENT: Improve lead quality - Level 1 intent below 25% threshold")
            if level_1 + level_2 < 50:
                recommendations.append("Focus on traffic source optimization to improve overall intent quality")
                
        elif analysis_type == 'publisher_performance' and isinstance(data, list):
            if len(data) > 0:
                top_performer = data[0]
                avg_conversion = sum(p.get('conversion_rate', 0) for p in data) / len(data)
                
                insights.extend([
                    f"Top performer: {top_performer.get('PUBLISHER', 'Unknown')} with {top_performer.get('conversion_rate', 0):.1f}% conversion",
                    f"Average conversion rate across {len(data)} publishers: {avg_conversion:.1f}%",
                    f"Performance spread indicates {'high' if max(p.get('conversion_rate', 0) for p in data) - min(p.get('conversion_rate', 0) for p in data) > 10 else 'moderate'} variability"
                ])
                
                recommendations.extend([
                    "Scale budget allocation toward top-performing publishers",
                    "Investigate success factors from high-converting sources",
                    "Consider pausing or optimizing underperforming channels"
                ])
        
        # Add generic insights if none generated
        if not insights:
            insights = [
                "Analysis completed with available data",
                "Performance patterns identified for optimization",
                "Opportunities exist for strategic improvements"
            ]
        
        if not recommendations:
            recommendations = [
                "Continue monitoring key performance indicators",
                "Implement data-driven optimization strategies",
                "Review performance trends regularly"
            ]
        
        return {
            **base_analysis,
            'insights': insights,
            'recommendations': recommendations,
            'confidence': 'medium',
            'extensive_mode': extensive_mode,
            'ai_enhanced': False
        } 