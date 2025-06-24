"""
Writer Agent
Specialized agent for formatting analysis results into business communications
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

class WriterAgent:
    """
    Writer Agent - Formats analysis results into business communication formats
    """
    
    def __init__(self):
        self.communication_types = [
            'daily_summary',
            'recommendations', 
            'detailed_analysis',
            'executive_summary'
        ]
    
    def _sanitize_content(self, content: str) -> str:
        """
        Sanitize content to prevent format string errors
        Escapes curly braces that might be interpreted as format placeholders
        """
        if not isinstance(content, str):
            return str(content)
        
        # Replace single curly braces with double curly braces to escape them
        # But preserve valid f-string patterns
        content = re.sub(r'(?<!\{)\{(?!\{)(?![^}]*:)', '{{', content)
        content = re.sub(r'(?<!\})\}(?!\})', '}}', content)
        
        return content
    
    def _format_recommendation(self, rec, extensive_mode=False) -> str:
        """
        Format a recommendation whether it's a string or GPT-4o structured dict
        """
        if isinstance(rec, dict):
            # GPT-4o structured recommendation
            main_text = rec.get('recommendation', 'No recommendation text provided')
            
            if extensive_mode:
                # Add additional details for extensive mode
                details = []
                
                if 'implementation_steps' in rec:
                    details.append(f"   **Implementation:** {rec['implementation_steps']}")
                
                if 'expected_ROI' in rec:
                    details.append(f"   **Expected ROI:** {rec['expected_ROI']}")
                
                if 'category' in rec:
                    details.append(f"   **Category:** {rec['category']}")
                
                if details:
                    return main_text + "\n" + "\n".join(details)
                else:
                    return main_text
            else:
                return main_text
        else:
            # Simple string recommendation
            return str(rec)
    
    def _categorize_recommendation(self, rec) -> str:
        """
        Categorize a recommendation as immediate, strategic, or monitoring
        """
        if isinstance(rec, dict):
            # Use GPT-4o category if available
            category = rec.get('category', '').lower()
            rec_text = rec.get('recommendation', '')
            
            if category == 'immediate' or any(word in rec_text.lower() for word in ['immediate', 'urgent', 'critical', 'now']):
                return 'immediate'
            elif category in ['strategic', 'short-term'] or any(word in rec_text.lower() for word in ['strategic', 'long-term', 'develop', 'build']):
                return 'strategic'
            else:
                return 'monitoring'
        else:
            # Analyze string recommendation
            rec_str = str(rec)
            if any(word in rec_str.lower() for word in ['immediate', 'urgent', 'critical', 'now']):
                return 'immediate'
            elif any(word in rec_str.lower() for word in ['strategic', 'long-term', 'develop', 'build']):
                return 'strategic'
            else:
                return 'monitoring'
    
    def format_analysis(self, analysis_result: Dict[str, Any], format_type: str) -> Dict[str, Any]:
        """
        Format analysis results into specified business communication format
        SIMPLIFIED VERSION - Just return markdown content without complex formatting
        """
        try:
            print("=" * 60)
            print("DEBUG: WRITER AGENT INPUT:")
            print("=" * 60)
            print(f"FORMAT TYPE: {format_type}")
            print(f"ANALYSIS TYPE: {analysis_result.get('analysis_type', 'Unknown')}")
            print(f"EXTENSIVE MODE: {analysis_result.get('extensive_mode', False)}")
            print(f"AI ENHANCED: {analysis_result.get('ai_enhanced', False)}")
            print(f"ANALYSIS STATUS: {analysis_result.get('analysis_status', 'completed')}")
            print(f"INSIGHTS COUNT: {len(analysis_result.get('insights', []))}")
            print(f"RECOMMENDATIONS COUNT: {len(analysis_result.get('recommendations', []))}")
            print(f"CONFIDENCE: {analysis_result.get('confidence', 'Unknown')}")
            
            if format_type not in self.communication_types:
                return {
                    "error": f"Unknown format type: {format_type}",
                    "available_formats": self.communication_types
                }
            
            if "error" in analysis_result:
                return {
                    "error": f"Cannot format analysis with errors: {analysis_result['error']}",
                    "format_type": format_type
                }
            
            # SIMPLIFIED APPROACH - Just create markdown content directly
            result = self._create_simple_markdown_report(analysis_result, format_type)
            
            print("=" * 60)
            print("DEBUG: WRITER AGENT OUTPUT:")
            print("=" * 60)
            print(f"TITLE: {result.get('title', 'No title')}")
            print(f"WORD COUNT: {result.get('word_count', 0)}")
            print(f"CONTENT PREVIEW (first 500 chars):")
            content = result.get('content', '')
            print(content[:500] + "..." if len(content) > 500 else content)
            print("=" * 60)
            
            return result
            
        except Exception as e:
            return {
                "error": f"Error formatting analysis: {str(e)}",
                "format_type": format_type,
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_simple_markdown_report(self, analysis_result: Dict[str, Any], format_type: str) -> Dict[str, Any]:
        """
        Create a simple markdown report without complex string formatting
        """
        analysis_type = analysis_result.get('analysis_type', 'general')
        insights = analysis_result.get('insights', [])
        data = analysis_result.get('data', {})
        recommendations = analysis_result.get('recommendations', [])
        confidence = analysis_result.get('confidence', 'medium')
        extensive_mode = analysis_result.get('extensive_mode', False)
        analyst_config = analysis_result.get('analyst_config', {})
        analysis_status = analysis_result.get('analysis_status', 'completed')
        
        # Build markdown content
        content_lines = []
        
        # Title based on format type
        if format_type == 'detailed_analysis':
            content_lines.append("# Comprehensive Marketing Performance Analysis")
        elif format_type == 'executive_summary':
            content_lines.append("# Executive Strategic Analysis")
        elif format_type == 'recommendations':
            content_lines.append("# Action Recommendations")
        else:
            content_lines.append("# Daily Performance Summary")
        
        content_lines.append("")
        
        # Analyst info if available
        if extensive_mode and analyst_config:
            content_lines.append("## Analysis Details")
            content_lines.append(f"**Analyst:** {analyst_config.get('name', 'Unknown Analyst')}")
            content_lines.append(f"**Specialization:** {analyst_config.get('description', 'General analysis')}")
            content_lines.append(f"**Analysis Depth:** {analyst_config.get('analysis_depth', 'Standard analysis')}")
            content_lines.append("")
        
        # Executive Overview
        content_lines.append("## Executive Overview")
        
        # Check if analysis failed or is incomplete
        if analysis_status == 'failed' or confidence == 'incomplete_analysis':
            content_lines.append("⚠️ **ANALYSIS INCOMPLETE**")
            content_lines.append("")
            content_lines.append("The requested comprehensive analysis could not be completed due to technical issues with the AI analysis system.")
            content_lines.append("The insights shown below are basic data summaries only and **should not be used for strategic decision-making**.")
            content_lines.append("")
            content_lines.append("**Recommended Actions:**")
            content_lines.append("- Retry the analysis to get complete GPT-4o powered insights")
            content_lines.append("- Use the basic analysis dropdown options for immediate data insights")
            content_lines.append("- Contact support if issues persist")
            content_lines.append("")
        elif analysis_type == 'summary_metrics' and isinstance(data, dict):
            total_calls = data.get('total_calls', 0)
            conversion_rate = data.get('conversion_rate', 0)
            total_sales = data.get('total_sales', 0)
            
            content_lines.append(f"This comprehensive analysis examines **{total_calls:,} calls** with a **{conversion_rate:.1f}% conversion rate**, resulting in **{total_sales:,} successful sales**.")
            
            if conversion_rate > 10:
                content_lines.append("Performance indicators show **strong operational efficiency** with above-average conversion rates.")
            elif conversion_rate > 5:
                content_lines.append("Performance indicators show **moderate operational efficiency** with room for improvement.")
            else:
                content_lines.append("Performance indicators show **concerning operational efficiency** requiring immediate attention.")
        
        else:
            # Create intelligent executive overview based on insights and analysis context
            if insights and len(insights) > 0:
                # Use the first key insight to create a meaningful executive overview
                first_insight = insights[0]
                if isinstance(first_insight, dict):
                    insight_text = first_insight.get('insight', '')
                    business_impact = first_insight.get('business_impact', '')
                    business_implication = first_insight.get('business_implication', '')
                    
                    if insight_text and (business_impact or business_implication):
                        content_lines.append(f"**Key Finding:** {insight_text}")
                        impact_text = business_impact or business_implication
                        content_lines.append(f"**Business Impact:** {impact_text}")
                    elif insight_text:
                        content_lines.append(f"**Primary Insight:** {insight_text}")
                    else:
                        content_lines.append("Analysis reveals critical performance patterns requiring strategic attention.")
                elif isinstance(first_insight, str) and len(first_insight.strip()) > 0:
                    # Simple string insight - this is what GPT-4o is actually returning
                    content_lines.append(f"**Key Finding:** {first_insight}")
                else:
                    content_lines.append("Analysis reveals critical performance patterns requiring strategic attention.")
                
                # Add context about the number of insights
                if len(insights) > 1:
                    content_lines.append(f"This analysis identifies **{len(insights)} critical insights** with actionable recommendations for immediate implementation.")
                else:
                    content_lines.append("Detailed analysis provides specific performance insights and strategic recommendations.")
            else:
                # Fallback if no insights available - but this should rarely happen now
                if analysis_type == 'intent_analysis':
                    content_lines.append("**Analysis Focus:** Lead quality and customer intent optimization across your marketing funnel.")
                elif analysis_type == 'publisher_performance':
                    content_lines.append("**Analysis Focus:** Publisher performance evaluation to identify top-performing channels and optimization opportunities.")
                else:
                    content_lines.append(f"**Analysis Focus:** Strategic {analysis_type.replace('_', ' ')} evaluation provides actionable insights for performance optimization.")
                
                content_lines.append("*Note: Limited insights available. Consider refining analysis parameters for more detailed results.*")
        
        content_lines.append("")
        
        # Key Insights
        if insights:
            content_lines.append("## Key Insights")
            for i, insight in enumerate(insights, 1):
                # Ensure insight is a string
                insight_text = str(insight) if not isinstance(insight, str) else insight
                content_lines.append(f"{i}. {insight_text}")
            content_lines.append("")
        
        # Data Analysis
        if data and isinstance(data, dict):
            content_lines.append("## Performance Metrics")
            
            for key, value in data.items():
                if key not in ['error'] and value is not None:
                    formatted_key = key.replace('_', ' ').title()
                    if isinstance(value, (int, float)):
                        if 'rate' in key.lower() or 'percentage' in key.lower():
                            content_lines.append(f"- **{formatted_key}:** {value:.1f}%")
                        elif 'cost' in key.lower() or 'revenue' in key.lower():
                            content_lines.append(f"- **{formatted_key}:** ${value:,.2f}")
                        else:
                            content_lines.append(f"- **{formatted_key}:** {value:,}")
                    else:
                        content_lines.append(f"- **{formatted_key}:** {value}")
            
            content_lines.append("")
        
        elif data and isinstance(data, list) and len(data) > 0:
            content_lines.append("## Performance Data")
            content_lines.append("| Publisher | Conversion Rate | Total Calls | Performance |")
            content_lines.append("|-----------|----------------|-------------|-------------|")
            
            for item in data[:10]:  # Show top 10
                if isinstance(item, dict):
                    publisher = item.get('PUBLISHER', 'Unknown')
                    conversion = item.get('conversion_rate', 0)
                    calls = item.get('total_calls', 0)
                    
                    if conversion > 15:
                        performance = "Excellent"
                    elif conversion > 10:
                        performance = "Good"
                    elif conversion > 5:
                        performance = "Moderate"
                    else:
                        performance = "Poor"
                    
                    content_lines.append(f"| {publisher} | {conversion:.1f}% | {calls:,} | {performance} |")
            
            content_lines.append("")
        
        # Recommendations
        if recommendations:
            content_lines.append("## Strategic Recommendations")
            
            # Categorize recommendations using the new helper function
            immediate_actions = []
            strategic_initiatives = []
            monitoring_actions = []
            
            for rec in recommendations:
                category = self._categorize_recommendation(rec)
                if category == 'immediate':
                    immediate_actions.append(rec)
                elif category == 'strategic':
                    strategic_initiatives.append(rec)
                else:
                    monitoring_actions.append(rec)
            
            if immediate_actions:
                content_lines.append("### Immediate Actions (0-30 days)")
                for i, rec in enumerate(immediate_actions, 1):
                    rec_text = self._format_recommendation(rec, extensive_mode)
                    content_lines.append(f"{i}. {rec_text}")
                    if extensive_mode and not isinstance(rec, dict):
                        content_lines.append("   - **Priority:** HIGH")
                        content_lines.append("   - **Timeline:** Immediate implementation required")
                content_lines.append("")
            
            if strategic_initiatives:
                content_lines.append("### Strategic Initiatives (30-90 days)")
                for i, rec in enumerate(strategic_initiatives, 1):
                    rec_text = self._format_recommendation(rec, extensive_mode)
                    content_lines.append(f"{i}. {rec_text}")
                    if extensive_mode and not isinstance(rec, dict):
                        content_lines.append("   - **Priority:** MEDIUM")
                        content_lines.append("   - **Timeline:** 1-3 months for full implementation")
                content_lines.append("")
            
            if monitoring_actions:
                content_lines.append("### Ongoing Monitoring")
                for i, rec in enumerate(monitoring_actions, 1):
                    rec_text = self._format_recommendation(rec, extensive_mode)
                    content_lines.append(f"{i}. {rec_text}")
                content_lines.append("")
        
        # Success Metrics (for extensive mode)
        if extensive_mode and analyst_config and 'key_metrics' in analyst_config:
            content_lines.append("## Success Metrics & KPIs")
            content_lines.append("**Primary KPIs to Track:**")
            for metric in analyst_config['key_metrics']:
                content_lines.append(f"- {metric}")
                # Add target ranges
                if 'conversion' in metric.lower():
                    content_lines.append("  - Target: >10% (Good), >15% (Excellent)")
                elif 'cost' in metric.lower():
                    content_lines.append("  - Target: <$100 (Good), <$75 (Excellent)")
                elif 'quality' in metric.lower():
                    content_lines.append("  - Target: >70% (Good), >85% (Excellent)")
            content_lines.append("")
        
        # Conclusion
        content_lines.append("## Conclusion")
        
        if analysis_status == 'failed' or confidence == 'incomplete_analysis':
            content_lines.append("⚠️ **This analysis is INCOMPLETE and should not be used for business decisions.**")
            content_lines.append("Please retry the analysis to get comprehensive GPT-4o powered insights.")
        else:
            content_lines.append(f"This analysis provides comprehensive insights with **{confidence}** confidence level.")
            
            if extensive_mode:
                content_lines.append("The insights and recommendations are based on rigorous data analysis and industry best practices.")
                content_lines.append("Implementation of recommended actions should result in measurable improvements in key performance indicators.")
        
        content_lines.append("")
        content_lines.append(f"**Analysis completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if analysis_status == 'failed' or confidence == 'incomplete_analysis':
            content_lines.append(f"**Status:** ❌ INCOMPLETE - Technical Issues")
        else:
            # Handle confidence as either string or dict (Claude returns dict)
            if isinstance(confidence, dict):
                confidence_text = confidence.get('level', 'Unknown')
            else:
                confidence_text = str(confidence)
            content_lines.append(f"**Confidence Level:** {confidence_text.title()}")
        
        # Join all content
        final_content = "\n".join(content_lines)
        
        # Return simple result
        return {
            "format_type": format_type,
            "title": content_lines[0].replace('# ', ''),
            "content": final_content,
            "word_count": len(final_content.split()),
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "extensive_mode": extensive_mode
        }
    
    def _format_daily_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format as daily summary: 'What happened yesterday'"""
        
        analysis_type = analysis_result.get('analysis_type', 'general')
        insights = analysis_result.get('insights', [])
        data = analysis_result.get('data', {})
        
        # Generate daily summary content
        summary_lines = []
        
        # Opening statement
        if analysis_type == 'summary_metrics':
            total_calls = data.get('total_calls', 0)
            publishers = data.get('unique_publishers', 0)
            summary_lines.append(f"Yesterday we processed {total_calls:,} calls from {publishers} publishers.")
        else:
            summary_lines.append("Daily performance analysis completed.")
        
        # Key changes and metrics
        if insights:
            summary_lines.append("\nKey Performance Highlights:")
            for insight in insights[:3]:  # Limit to top 3 insights
                summary_lines.append(f"• {insight}")
        
        # Notable events
        if analysis_type == 'outlier_analysis':
            total_outliers = analysis_result.get('total_outliers', 0)
            if total_outliers > 0:
                summary_lines.append(f"\nNotable Events: {total_outliers} performance outliers detected requiring attention.")
            else:
                summary_lines.append("\nNotable Events: Consistent performance across all publishers.")
        
        elif analysis_type == 'publisher_performance':
            performance_data = analysis_result.get('data', [])
            if performance_data:
                best_publisher = performance_data[0].get('PUBLISHER', 'Unknown')
                summary_lines.append(f"\nNotable Events: {best_publisher} maintained top performance ranking.")
        
        # Closing summary
        confidence = analysis_result.get('confidence', 'medium')
        # Handle confidence as either string or dict (Claude returns dict)
        if isinstance(confidence, dict):
            confidence_text = confidence.get('level', 'Unknown')
        else:
            confidence_text = str(confidence)
        summary_lines.append(f"\nData confidence level: {confidence_text.title()}")
        
        final_content = "\n".join(summary_lines)
        return {
            "format_type": "daily_summary",
            "title": "Daily Performance Summary",
            "content": self._sanitize_content(final_content),
            "word_count": len(" ".join(summary_lines).split()),
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence
        }
    
    def _format_recommendations(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format as actionable recommendations: 'What recommendations to act on'"""
        
        recommendations = analysis_result.get('recommendations', [])
        analysis_type = analysis_result.get('analysis_type', 'general')
        confidence = analysis_result.get('confidence', 'medium')
        
        # Prioritize recommendations
        prioritized_recs = []
        
        # Priority 1: Critical issues (conversion, cost efficiency)
        critical_keywords = ['improve conversion', 'cost-efficient', 'review strategy', 'priority']
        priority_1 = [rec for rec in recommendations if any(keyword in str(rec).lower() for keyword in critical_keywords)]
        
        # Priority 2: Optimization opportunities
        optimization_keywords = ['scale', 'increase budget', 'optimize', 'focus on']
        priority_2 = [rec for rec in recommendations if any(keyword in str(rec).lower() for keyword in optimization_keywords)]
        
        # Priority 3: Monitoring and maintenance
        monitor_keywords = ['monitor', 'maintain', 'investigate']
        priority_3 = [rec for rec in recommendations if any(keyword in str(rec).lower() for keyword in monitor_keywords)]
        
        # Build prioritized recommendation list
        content_lines = []
        
        if priority_1:
            content_lines.append("PRIORITY 1 - IMMEDIATE ACTION REQUIRED:")
            for i, rec in enumerate(priority_1[:2], 1):  # Max 2 critical items
                business_impact = self._estimate_business_impact(rec, 'high')
                content_lines.append(f"{i}. {rec}")
                content_lines.append(f"   Impact: {business_impact}")
                content_lines.append("")
        
        if priority_2:
            content_lines.append("PRIORITY 2 - OPTIMIZATION OPPORTUNITIES:")
            for i, rec in enumerate(priority_2[:2], 1):  # Max 2 optimization items
                business_impact = self._estimate_business_impact(rec, 'medium')
                content_lines.append(f"{i}. {rec}")
                content_lines.append(f"   Impact: {business_impact}")
                content_lines.append("")
        
        if priority_3:
            content_lines.append("MONITOR & MAINTAIN:")
            for rec in priority_3[:2]:  # Max 2 monitoring items
                content_lines.append(f"• {rec}")
        
        # Add confidence and timing
        # Handle confidence as either string or dict (Claude returns dict)
        if isinstance(confidence, dict):
            confidence_text = confidence.get('level', 'Unknown')
        else:
            confidence_text = str(confidence)
        content_lines.append(f"\nRecommendation Confidence: {confidence_text.title()}")
        content_lines.append("Recommended Review Frequency: Daily for Priority 1, Weekly for Priority 2")
        
        final_content = "\n".join(content_lines)
        return {
            "format_type": "recommendations",
            "title": "Action Recommendations",
            "content": self._sanitize_content(final_content),
            "priority_1_count": len(priority_1),
            "priority_2_count": len(priority_2),
            "total_recommendations": len(recommendations),
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence
        }
    
    def _format_detailed_analysis(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format as detailed analysis: 'Detailed analysis of yesterday's actions and insights'"""
        
        analysis_type = analysis_result.get('analysis_type', 'general')
        insights = analysis_result.get('insights', [])
        data = analysis_result.get('data', {})
        recommendations = analysis_result.get('recommendations', [])
        confidence = analysis_result.get('confidence', 'medium')
        
        # Check if this is an extensive analysis request
        extensive_mode = analysis_result.get('extensive_mode', False)
        analyst_config = analysis_result.get('analyst_config', {})
        analysis_focus = analysis_result.get('analysis_focus', 'what_happened')
        
        content_sections = []
        
        # Executive Overview
        content_sections.append("EXECUTIVE OVERVIEW")
        content_sections.append("=" * 50)
        
        if extensive_mode and analyst_config:
            content_sections.append(f"Analysis conducted by: {analyst_config.get('name', 'Unknown Analyst')}")
            content_sections.append(f"Specialization: {analyst_config.get('description', 'General analysis')}")
            content_sections.append(f"Analysis Depth: {analyst_config.get('analysis_depth', 'Standard analysis')}")
            content_sections.append("")
        
        if analysis_type == 'summary_metrics':
            total_calls = data.get('total_calls', 0)
            conversion_rate = data.get('conversion_rate', 0)
            total_sales = data.get('total_sales', 0)
            
            content_sections.append(f"This comprehensive analysis examines {total_calls:,} calls with a {conversion_rate:.1f}% conversion rate, resulting in {total_sales:,} successful sales.")
            
            if extensive_mode:
                content_sections.append(f"The analysis reveals critical performance patterns across multiple dimensions of your marketing operation.")
                performance_level = 'strong' if conversion_rate > 10 else 'moderate' if conversion_rate > 5 else 'concerning'
                content_sections.append(f"Key performance indicators show {performance_level} operational efficiency.")
        else:
            # Create intelligent executive overview for detailed analysis
            if insights and len(insights) > 0:
                first_insight = insights[0]
                if isinstance(first_insight, dict):
                    insight_text = first_insight.get('insight', '')
                    business_impact = first_insight.get('business_impact', '')
                    if insight_text:
                        content_sections.append(f"Key Finding: {insight_text}")
                        if business_impact:
                            content_sections.append(f"Business Impact: {business_impact}")
                    else:
                        content_sections.append("Strategic analysis reveals critical performance patterns requiring immediate attention.")
                else:
                    content_sections.append(f"Key Finding: {str(first_insight)}")
            else:
                if analysis_type == 'intent_analysis':
                    content_sections.append("Lead quality and customer intent analysis reveals optimization opportunities across your marketing funnel.")
                elif analysis_type == 'publisher_performance':
                    content_sections.append("Publisher performance analysis identifies top-performing channels and optimization opportunities.")
                else:
                    content_sections.append(f"Strategic {analysis_type.replace('_', ' ')} analysis provides actionable insights for performance optimization.")
            
            if extensive_mode:
                content_sections.append("This analysis addresses critical business questions and provides actionable insights for strategic decision-making.")
        
        content_sections.append("")
        
        # Performance Deep-Dive
        content_sections.append("PERFORMANCE DEEP-DIVE ANALYSIS")
        content_sections.append("=" * 50)
        
        if analysis_type == 'publisher_performance':
            performance_data = data if isinstance(data, list) else []
            if performance_data:
                content_sections.append("PUBLISHER PERFORMANCE RANKINGS:")
                content_sections.append("-" * 40)
                
                for i, publisher in enumerate(performance_data[:10], 1):  # Top 10
                    name = publisher.get('PUBLISHER', 'Unknown')
                    conversion = publisher.get('conversion_rate', 0)
                    calls = publisher.get('total_calls', 0)
                    strong_leads = publisher.get('strong_lead_rate', 0)
                    
                    content_sections.append(f"{i:2d}. {name}")
                    content_sections.append(f"    • Conversion Rate: {conversion:.1f}%")
                    content_sections.append(f"    • Call Volume: {calls:,}")
                    content_sections.append(f"    • Strong Lead Rate: {strong_leads:.1f}%")
                    
                    # Performance assessment
                    if conversion > 15:
                        assessment = "EXCELLENT - Top tier performer"
                    elif conversion > 10:
                        assessment = "GOOD - Above average performance"
                    elif conversion > 5:
                        assessment = "MODERATE - Room for improvement"
                    else:
                        assessment = "POOR - Requires immediate attention"
                    
                    content_sections.append(f"    • Assessment: {assessment}")
                    content_sections.append("")
                
                if extensive_mode:
                    # Statistical analysis
                    conversion_rates = [p.get('conversion_rate', 0) for p in performance_data]
                    avg_conversion = sum(conversion_rates) / len(conversion_rates) if conversion_rates else 0
                    max_conversion = max(conversion_rates) if conversion_rates else 0
                    min_conversion = min(conversion_rates) if conversion_rates else 0
                    
                    content_sections.append("STATISTICAL PERFORMANCE ANALYSIS:")
                    content_sections.append("-" * 40)
                    content_sections.append(f"Average Conversion Rate: {avg_conversion:.2f}%")
                    content_sections.append(f"Best Performer: {max_conversion:.2f}%")
                    content_sections.append(f"Worst Performer: {min_conversion:.2f}%")
                    content_sections.append(f"Performance Spread: {max_conversion - min_conversion:.2f}%")
                    content_sections.append("")
                    
                    # Performance distribution analysis
                    excellent = len([r for r in conversion_rates if r > 15])
                    good = len([r for r in conversion_rates if 10 < r <= 15])
                    moderate = len([r for r in conversion_rates if 5 < r <= 10])
                    poor = len([r for r in conversion_rates if r <= 5])
                    
                    content_sections.append("PERFORMANCE DISTRIBUTION:")
                    content_sections.append(f"• Excellent (>15%): {excellent} publishers ({excellent/len(conversion_rates)*100:.1f}%)")
                    content_sections.append(f"• Good (10-15%): {good} publishers ({good/len(conversion_rates)*100:.1f}%)")
                    content_sections.append(f"• Moderate (5-10%): {moderate} publishers ({moderate/len(conversion_rates)*100:.1f}%)")
                    content_sections.append(f"• Poor (<5%): {poor} publishers ({poor/len(conversion_rates)*100:.1f}%)")
                    content_sections.append("")
        
        elif analysis_type == 'intent_analysis':
            if isinstance(data, dict):
                content_sections.append("CUSTOMER INTENT QUALITY BREAKDOWN:")
                content_sections.append("-" * 40)
                
                for intent_level, percentage in data.items():
                    if intent_level != 'error':
                        content_sections.append(f"• {intent_level}: {percentage:.1f}%")
                        
                        # Quality assessment
                        if 'Level 1' in intent_level or 'High' in intent_level:
                            if percentage > 40:
                                assessment = "EXCELLENT - Strong lead quality"
                            elif percentage > 25:
                                assessment = "GOOD - Acceptable quality"
                            else:
                                assessment = "POOR - Quality improvement needed"
                        elif 'Level 2' in intent_level:
                            assessment = "MODERATE - Potential for conversion"
                        else:
                            assessment = "LOW - Limited conversion potential"
                        
                        content_sections.append(f"  Assessment: {assessment}")
                content_sections.append("")
                
                if extensive_mode:
                    # Lead quality analysis
                    high_quality = data.get('Level 1', 0)
                    medium_quality = data.get('Level 2', 0)
                    low_quality = data.get('Level 3', 0)
                    
                    content_sections.append("LEAD QUALITY IMPACT ANALYSIS:")
                    content_sections.append("-" * 40)
                    content_sections.append(f"High-Quality Lead Impact: {high_quality:.1f}% of traffic")
                    content_sections.append(f"Expected conversion from high-quality: {high_quality * 0.3:.1f}%")
                    content_sections.append(f"Medium-Quality Lead Impact: {medium_quality:.1f}% of traffic")
                    content_sections.append(f"Expected conversion from medium-quality: {medium_quality * 0.15:.1f}%")
                    content_sections.append(f"Total Expected Conversion: {(high_quality * 0.3 + medium_quality * 0.15):.1f}%")
                    content_sections.append("")
        
        elif analysis_type == 'cost_analysis':
            if isinstance(data, dict):
                total_revenue = data.get('total_revenue', 0)
                total_cost = data.get('total_cost', 0)
                roi = data.get('roi_percentage', 0)
                cost_per_sale = data.get('cost_per_sale', 0)
                
                content_sections.append("FINANCIAL PERFORMANCE ANALYSIS:")
                content_sections.append("-" * 40)
                content_sections.append(f"Total Revenue: ${total_revenue:,.2f}")
                content_sections.append(f"Total Investment: ${total_cost:,.2f}")
                content_sections.append(f"Net Profit: ${total_revenue - total_cost:,.2f}")
                content_sections.append(f"Return on Investment: {roi:.1f}%")
                content_sections.append(f"Cost Per Sale: ${cost_per_sale:.2f}")
                content_sections.append("")
                
                # ROI Assessment
                if roi > 300:
                    roi_assessment = "EXCELLENT - Outstanding financial performance"
                elif roi > 200:
                    roi_assessment = "GOOD - Strong return on investment"
                elif roi > 100:
                    roi_assessment = "MODERATE - Profitable but room for improvement"
                else:
                    roi_assessment = "POOR - Below break-even, immediate action required"
                
                content_sections.append(f"ROI Assessment: {roi_assessment}")
                content_sections.append("")
                
                if extensive_mode:
                    # Financial efficiency analysis
                    content_sections.append("COST EFFICIENCY BREAKDOWN:")
                    content_sections.append("-" * 40)
                    
                    revenue_per_call = data.get('revenue_per_call', 0)
                    cost_per_call = data.get('cost_per_call', 0)
                    profit_per_call = revenue_per_call - cost_per_call
                    
                    content_sections.append(f"Revenue per Call: ${revenue_per_call:.2f}")
                    content_sections.append(f"Cost per Call: ${cost_per_call:.2f}")
                    content_sections.append(f"Profit per Call: ${profit_per_call:.2f}")
                    content_sections.append(f"Profit Margin: {(profit_per_call/revenue_per_call*100):.1f}%" if revenue_per_call > 0 else "Profit Margin: N/A")
                    content_sections.append("")
        
        # Statistical Analysis and Trends
        content_sections.append("STATISTICAL ANALYSIS & INSIGHTS")
        content_sections.append("=" * 50)
        
        if insights:
            content_sections.append("KEY STATISTICAL INSIGHTS:")
            content_sections.append("-" * 30)
            for i, insight in enumerate(insights, 1):
                content_sections.append(f"{i}. {insight}")
                
                if extensive_mode:
                    # Add context and implications for each insight
                    if 'conversion' in insight.lower():
                        content_sections.append("   → Implication: Direct impact on revenue generation")
                        content_sections.append("   → Monitoring: Track weekly for trend analysis")
                    elif 'cost' in insight.lower():
                        content_sections.append("   → Implication: Affects profitability and budget allocation")
                        content_sections.append("   → Monitoring: Review monthly for optimization opportunities")
                    elif 'quality' in insight.lower():
                        content_sections.append("   → Implication: Influences long-term customer value")
                        content_sections.append("   → Monitoring: Assess quarterly for strategic planning")
            content_sections.append("")
        
        if extensive_mode:
            # Risk Analysis Section
            content_sections.append("RISK ANALYSIS & MITIGATION")
            content_sections.append("=" * 50)
            
            if analysis_type == 'publisher_performance':
                content_sections.append("IDENTIFIED RISKS:")
                content_sections.append("• Publisher Concentration Risk: Over-reliance on top performers")
                content_sections.append("• Performance Volatility: Inconsistent conversion rates")
                content_sections.append("• Quality Degradation: Potential decline in lead quality")
                content_sections.append("")
                
                content_sections.append("MITIGATION STRATEGIES:")
                content_sections.append("• Diversify publisher portfolio to reduce concentration")
                content_sections.append("• Implement performance monitoring and alert systems")
                content_sections.append("• Establish quality benchmarks and regular reviews")
                content_sections.append("")
            
            elif analysis_type == 'cost_analysis':
                content_sections.append("FINANCIAL RISKS:")
                content_sections.append("• Cost Inflation: Rising acquisition costs")
                content_sections.append("• Revenue Volatility: Fluctuating conversion values")
                content_sections.append("• Market Competition: Pressure on margins")
                content_sections.append("")
                
                content_sections.append("RISK MITIGATION:")
                content_sections.append("• Implement cost caps and budget controls")
                content_sections.append("• Diversify revenue streams and customer segments")
                content_sections.append("• Monitor competitive landscape and adjust strategies")
                content_sections.append("")
        
        # Recommendations Summary
        if recommendations:
            content_sections.append("STRATEGIC RECOMMENDATIONS")
            content_sections.append("=" * 50)
            
            # Categorize recommendations
            immediate_actions = []
            strategic_initiatives = []
            monitoring_actions = []
            
            for rec in recommendations:
                # Ensure rec is a string before calling .lower()
                rec_str = str(rec) if not isinstance(rec, str) else rec
                if any(word in rec_str.lower() for word in ['immediate', 'urgent', 'critical', 'now']):
                    immediate_actions.append(rec)
                elif any(word in rec_str.lower() for word in ['strategic', 'long-term', 'develop', 'build']):
                    strategic_initiatives.append(rec)
                else:
                    monitoring_actions.append(rec)
            
            if immediate_actions:
                content_sections.append("IMMEDIATE ACTIONS (0-30 days):")
                for i, rec in enumerate(immediate_actions, 1):
                    content_sections.append(f"{i}. {rec}")
                    if extensive_mode:
                        content_sections.append(f"   Priority: HIGH | Timeline: Immediate")
                        content_sections.append(f"   Resources: Assign dedicated team member")
                content_sections.append("")
            
            if strategic_initiatives:
                content_sections.append("STRATEGIC INITIATIVES (30-90 days):")
                for i, rec in enumerate(strategic_initiatives, 1):
                    content_sections.append(f"{i}. {rec}")
                    if extensive_mode:
                        content_sections.append(f"   Priority: MEDIUM | Timeline: 1-3 months")
                        content_sections.append(f"   Resources: Cross-functional team required")
                content_sections.append("")
            
            if monitoring_actions:
                content_sections.append("ONGOING MONITORING (Continuous):")
                for i, rec in enumerate(monitoring_actions, 1):
                    content_sections.append(f"{i}. {rec}")
                    if extensive_mode:
                        content_sections.append(f"   Priority: LOW | Timeline: Ongoing")
                        content_sections.append(f"   Resources: Automated monitoring preferred")
                content_sections.append("")
        
        if extensive_mode:
            # Success Metrics Section
            content_sections.append("SUCCESS METRICS & KPIs")
            content_sections.append("=" * 50)
            
            if analyst_config and 'key_metrics' in analyst_config:
                content_sections.append("PRIMARY KPIS TO TRACK:")
                for metric in analyst_config['key_metrics']:
                    content_sections.append(f"• {metric}")
                    # Add target ranges based on metric type
                    if 'conversion' in metric.lower():
                        content_sections.append("  Target: >10% (Good), >15% (Excellent)")
                    elif 'cost' in metric.lower():
                        content_sections.append("  Target: <$100 (Good), <$75 (Excellent)")
                    elif 'quality' in metric.lower():
                        content_sections.append("  Target: >70% (Good), >85% (Excellent)")
                content_sections.append("")
            
            content_sections.append("MONITORING FREQUENCY:")
            content_sections.append("• Daily: Conversion rates, call volume")
            content_sections.append("• Weekly: Cost per sale, publisher performance")
            content_sections.append("• Monthly: ROI, strategic KPIs")
            content_sections.append("• Quarterly: Comprehensive performance review")
            content_sections.append("")
        
        # Conclusion
        content_sections.append("CONCLUSION")
        content_sections.append("=" * 50)
        content_sections.append(f"This detailed analysis provides a comprehensive view of your marketing performance with {confidence} confidence level.")
        
        if extensive_mode:
            content_sections.append("The insights and recommendations presented are based on rigorous data analysis and industry best practices.")
            content_sections.append("Implementation of the recommended actions should result in measurable improvements in key performance indicators.")
            content_sections.append("Regular monitoring and adjustment of strategies will ensure continued optimization and growth.")
        
        content_sections.append("")
        content_sections.append(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # Handle confidence as either string or dict (Claude returns dict)
        if isinstance(confidence, dict):
            confidence_text = confidence.get('level', 'Unknown')
        else:
            confidence_text = str(confidence)
        content_sections.append(f"Confidence Level: {confidence_text.title()}")
        
        final_content = "\n".join(content_sections)
        
        return {
            "format_type": "detailed_analysis",
            "title": "Comprehensive Marketing Performance Analysis",
            "content": self._sanitize_content(final_content),
            "word_count": len(final_content.split()),
            "sections": ["Executive Overview", "Performance Deep-Dive", "Statistical Analysis", "Risk Analysis", "Recommendations", "Success Metrics"],
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "extensive_mode": extensive_mode
        }
    
    def _format_executive_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format as executive summary: 'Executive summary from detailed analysis'"""
        
        analysis_type = analysis_result.get('analysis_type', 'general')
        insights = analysis_result.get('insights', [])
        data = analysis_result.get('data', {})
        recommendations = analysis_result.get('recommendations', [])
        confidence = analysis_result.get('confidence', 'medium')
        
        # Check if this is an extensive analysis request
        extensive_mode = analysis_result.get('extensive_mode', False)
        analyst_config = analysis_result.get('analyst_config', {})
        analysis_focus = analysis_result.get('analysis_focus', 'what_happened')
        
        content_lines = []
        
        # Executive Header
        content_lines.append("EXECUTIVE SUMMARY")
        content_lines.append("=" * 60)
        
        if extensive_mode and analyst_config:
            content_lines.append(f"Analysis Domain: {analyst_config.get('name', 'Marketing Performance')}")
            content_lines.append(f"Strategic Focus: {analyst_config.get('description', 'Comprehensive analysis')}")
            # Handle confidence as either string or dict (Claude returns dict)
            if isinstance(confidence, dict):
                confidence_text = confidence.get('level', 'Unknown')
            else:
                confidence_text = str(confidence)
            content_lines.append(f"Confidence Level: {confidence_text.title()}")
            content_lines.append("")
        
        # Strategic Overview (2-3 sentences)
        content_lines.append("STRATEGIC OVERVIEW")
        content_lines.append("-" * 30)
        
        if analysis_type == 'summary_metrics':
            total_calls = data.get('total_calls', 0)
            conversion_rate = data.get('conversion_rate', 0)
            total_sales = data.get('total_sales', 0)
            
            if conversion_rate > 15:
                performance_assessment = "exceptional performance with industry-leading metrics"
                strategic_position = "strong competitive advantage"
            elif conversion_rate > 10:
                performance_assessment = "solid performance above industry benchmarks"
                strategic_position = "competitive market position"
            elif conversion_rate > 5:
                performance_assessment = "moderate performance with improvement opportunities"
                strategic_position = "opportunity for competitive gains"
            else:
                performance_assessment = "below-benchmark performance requiring immediate intervention"
                strategic_position = "critical need for strategic restructuring"
            
            content_lines.append(f"Our marketing operations demonstrate {performance_assessment}, processing {total_calls:,} calls with a {conversion_rate:.1f}% conversion rate, yielding {total_sales:,} successful sales.")
            content_lines.append(f"This performance indicates a {strategic_position} in the marketplace.")
            
            if extensive_mode:
                # Calculate business impact
                revenue_potential = data.get('total_revenue', 0)
                if revenue_potential > 0:
                    content_lines.append(f"Current revenue generation of ${revenue_potential:,.0f} represents significant business value with clear optimization pathways.")
                
                content_lines.append("The analysis reveals both operational excellence opportunities and strategic growth vectors that can drive immediate and long-term value creation.")
            
        elif analysis_type == 'publisher_performance':
            performance_data = data if isinstance(data, list) else []
            if performance_data and len(performance_data) > 1:
                conversion_rates = [p.get('conversion_rate', 0) for p in performance_data]
                best_rate = max(conversion_rates) if conversion_rates else 0
                worst_rate = min(conversion_rates) if conversion_rates else 0
                spread = best_rate - worst_rate
                
                if spread > 10:
                    content_lines.append(f"Significant performance variation across our publisher portfolio (range: {worst_rate:.1f}% to {best_rate:.1f}%) presents substantial optimization opportunities.")
                    content_lines.append("Strategic reallocation of resources toward top-performing channels could yield immediate ROI improvements.")
                else:
                    content_lines.append(f"Publisher performance demonstrates consistency across the portfolio with a {spread:.1f}% performance range.")
                    content_lines.append("This stability provides a solid foundation for scaling operations and exploring new growth channels.")
                
                if extensive_mode:
                    top_performers = [p for p in performance_data if p.get('conversion_rate', 0) > 15]
                    content_lines.append(f"Portfolio analysis identifies {len(top_performers)} exceptional performers representing untapped scaling potential.")
        
        elif analysis_type == 'cost_analysis':
            roi = data.get('roi_percentage', 0)
            total_revenue = data.get('total_revenue', 0)
            total_cost = data.get('total_cost', 0)
            
            if roi > 300:
                financial_assessment = "outstanding financial performance with exceptional returns"
                strategic_implication = "strong foundation for aggressive growth investment"
            elif roi > 200:
                financial_assessment = "strong financial performance exceeding industry standards"
                strategic_implication = "opportunity for measured expansion"
            elif roi > 100:
                financial_assessment = "profitable operations with optimization potential"
                strategic_implication = "focus on efficiency improvements before scaling"
            else:
                financial_assessment = "sub-optimal financial performance requiring immediate restructuring"
                strategic_implication = "critical need for cost reduction and process optimization"
            
            content_lines.append(f"Financial analysis reveals {financial_assessment} with {roi:.1f}% ROI on ${total_cost:,.0f} investment generating ${total_revenue:,.0f} in revenue.")
            content_lines.append(f"This performance suggests {strategic_implication}.")
            
            if extensive_mode:
                profit = total_revenue - total_cost
                value_creation_level = 'strong' if profit > total_cost * 0.5 else 'moderate' if profit > 0 else 'concerning'
                content_lines.append(f"Net profit of ${profit:,.0f} demonstrates {value_creation_level} value creation capability.")
        
        else:
            content_lines.append(f"Comprehensive {analysis_type.replace('_', ' ')} analysis reveals key strategic insights for executive decision-making.")
            
            if extensive_mode:
                content_lines.append("The analysis provides data-driven recommendations for optimizing operational performance and achieving strategic objectives.")
        
        content_lines.append("")
        
        # Key Metrics Summary
        content_lines.append("KEY PERFORMANCE INDICATORS")
        content_lines.append("-" * 30)
        
        if analysis_type == 'intent_analysis' and isinstance(data, dict):
            high_quality = data.get('Level 1', 0)
            medium_quality = data.get('Level 2', 0)
            
            content_lines.append(f"• Lead Quality Index: {high_quality:.1f}% premium leads")
            content_lines.append(f"• Conversion Potential: {medium_quality:.1f}% qualified prospects")
            
            if extensive_mode:
                total_quality = high_quality + medium_quality
                content_lines.append(f"• Overall Quality Score: {total_quality:.1f}% (Target: >60%)")
                
                if total_quality > 60:
                    quality_assessment = "STRONG - Exceeds industry benchmarks"
                elif total_quality > 40:
                    quality_assessment = "MODERATE - Meets minimum standards"
                else:
                    quality_assessment = "WEAK - Below acceptable thresholds"
                
                content_lines.append(f"• Quality Assessment: {quality_assessment}")
            
        elif analysis_type == 'cost_analysis' and isinstance(data, dict):
            roi = data.get('roi_percentage', 0)
            cost_per_sale = data.get('cost_per_sale', 0)
            
            content_lines.append(f"• Return on Investment: {roi:.1f}%")
            content_lines.append(f"• Customer Acquisition Cost: ${cost_per_sale:.2f}")
            
            if extensive_mode:
                revenue_per_call = data.get('revenue_per_call', 0)
                profit_margin = data.get('profit_margin', 0)
                
                content_lines.append(f"• Revenue per Lead: ${revenue_per_call:.2f}")
                content_lines.append(f"• Profit Margin: {profit_margin:.1f}%")
                
                # Benchmark against industry standards
                if cost_per_sale < 75:
                    efficiency_rating = "EXCELLENT - Top quartile efficiency"
                elif cost_per_sale < 100:
                    efficiency_rating = "GOOD - Above average efficiency"
                elif cost_per_sale < 150:
                    efficiency_rating = "MODERATE - Industry average"
                else:
                    efficiency_rating = "POOR - Below industry standards"
                
                content_lines.append(f"• Efficiency Rating: {efficiency_rating}")
            
        elif analysis_type == 'summary_metrics':
            total_calls = data.get('total_calls', 0)
            conversion_rate = data.get('conversion_rate', 0)
            publishers = data.get('unique_publishers', 0)
            
            content_lines.append(f"• Call Volume: {total_calls:,} leads processed")
            content_lines.append(f"• Conversion Rate: {conversion_rate:.1f}%")
            content_lines.append(f"• Channel Portfolio: {publishers} active publishers")
            
            if extensive_mode:
                strong_lead_rate = data.get('strong_lead_rate', 0)
                content_lines.append(f"• Lead Quality: {strong_lead_rate:.1f}% high-intent prospects")
                
                # Performance rating
                if conversion_rate > 15:
                    performance_rating = "EXCEPTIONAL - Industry leadership"
                elif conversion_rate > 10:
                    performance_rating = "STRONG - Above market average"
                elif conversion_rate > 5:
                    performance_rating = "MODERATE - Market competitive"
                else:
                    performance_rating = "WEAK - Below market standards"
                
                content_lines.append(f"• Performance Rating: {performance_rating}")
        
        content_lines.append("")
        
        # Critical Decisions Needed
        content_lines.append("CRITICAL DECISIONS REQUIRED")
        content_lines.append("-" * 30)
        
        # Extract and prioritize critical recommendations
        critical_recs = []
        strategic_recs = []
        
        for rec in recommendations:
            # Ensure rec is a string before calling .lower()
            rec_str = str(rec) if not isinstance(rec, str) else rec
            if any(word in rec_str.lower() for word in ['critical', 'immediate', 'urgent', 'priority']):
                critical_recs.append(rec)
            elif any(word in rec_str.lower() for word in ['strategic', 'long-term', 'scale', 'expand']):
                strategic_recs.append(rec)
        
        if critical_recs:
            for i, rec in enumerate(critical_recs[:3], 1):  # Top 3 critical
                content_lines.append(f"{i}. {rec}")
                
                if extensive_mode:
                    # Add business impact context
                    rec_str = str(rec) if not isinstance(rec, str) else rec
                    if 'cost' in rec_str.lower():
                        content_lines.append("   → Impact: Direct effect on profitability and cash flow")
                        content_lines.append("   → Timeline: Implement within 30 days")
                    elif 'performance' in rec_str.lower() or 'conversion' in rec_str.lower():
                        content_lines.append("   → Impact: Revenue generation and market position")
                        content_lines.append("   → Timeline: Implement within 60 days")
                    elif 'quality' in rec_str.lower():
                        content_lines.append("   → Impact: Long-term customer value and brand reputation")
                        content_lines.append("   → Timeline: Implement within 90 days")
        else:
            content_lines.append("• Continue monitoring current performance trends")
            content_lines.append("• Maintain existing operational strategies")
            
            if extensive_mode:
                content_lines.append("• Focus on incremental optimization opportunities")
        
        content_lines.append("")
        
        # Strategic Recommendations
        content_lines.append("STRATEGIC INITIATIVES")
        content_lines.append("-" * 30)
        
        if strategic_recs:
            for i, rec in enumerate(strategic_recs[:3], 1):  # Top 3 strategic
                content_lines.append(f"{i}. {rec}")
                
                if extensive_mode:
                    # Add strategic context
                    rec_str = str(rec) if not isinstance(rec, str) else rec
                    if 'scale' in rec_str.lower() or 'expand' in rec_str.lower():
                        content_lines.append("   → Strategic Value: Market expansion and revenue growth")
                        content_lines.append("   → Investment Required: Moderate to High")
                        content_lines.append("   → Expected ROI: 6-18 months")
                    elif 'optimize' in rec_str.lower() or 'improve' in rec_str.lower():
                        content_lines.append("   → Strategic Value: Operational efficiency and cost reduction")
                        content_lines.append("   → Investment Required: Low to Moderate")
                        content_lines.append("   → Expected ROI: 3-12 months")
        else:
            content_lines.append("• Maintain current strategic direction")
            content_lines.append("• Focus on operational excellence")
            
            if extensive_mode:
                content_lines.append("• Explore adjacent market opportunities")
        
        content_lines.append("")
        
        if extensive_mode:
            # Business Impact Assessment
            content_lines.append("BUSINESS IMPACT ASSESSMENT")
            content_lines.append("-" * 30)
            
            if analysis_type == 'cost_analysis':
                roi = data.get('roi_percentage', 0)
                total_revenue = data.get('total_revenue', 0)
                
                if roi > 200:
                    impact_level = "HIGH POSITIVE IMPACT"
                    business_value = f"Exceptional value creation with ${total_revenue:,.0f} revenue generation"
                elif roi > 100:
                    impact_level = "MODERATE POSITIVE IMPACT"
                    business_value = f"Solid value creation with ${total_revenue:,.0f} revenue generation"
                else:
                    impact_level = "IMPROVEMENT REQUIRED"
                    business_value = f"Sub-optimal value creation requiring strategic intervention"
                
                content_lines.append(f"• Financial Impact: {impact_level}")
                content_lines.append(f"• Business Value: {business_value}")
                
            elif analysis_type == 'publisher_performance':
                performance_data = data if isinstance(data, list) else []
                if performance_data:
                    top_performers = len([p for p in performance_data if p.get('conversion_rate', 0) > 15])
                    total_publishers = len(performance_data)
                    
                    if top_performers / total_publishers > 0.3:
                        portfolio_strength = "STRONG portfolio with multiple high performers"
                    elif top_performers / total_publishers > 0.1:
                        portfolio_strength = "MODERATE portfolio with selective strength"
                    else:
                        portfolio_strength = "WEAK portfolio requiring diversification"
                    
                    content_lines.append(f"• Portfolio Strength: {portfolio_strength}")
                    content_lines.append(f"• Growth Potential: {top_performers} channels ready for scaling")
            
            content_lines.append("")
            
            # Resource Requirements
            content_lines.append("RESOURCE REQUIREMENTS")
            content_lines.append("-" * 30)
            
            total_critical = len(critical_recs)
            total_strategic = len(strategic_recs)
            
            if total_critical > 2:
                urgency_level = "HIGH - Immediate resource allocation required"
            elif total_critical > 0:
                urgency_level = "MODERATE - Focused resource deployment needed"
            else:
                urgency_level = "LOW - Standard resource allocation sufficient"
            
            content_lines.append(f"• Urgency Level: {urgency_level}")
            content_lines.append(f"• Critical Actions: {total_critical} requiring immediate attention")
            content_lines.append(f"• Strategic Initiatives: {total_strategic} for long-term growth")
            
            # Budget implications
            if analysis_type == 'cost_analysis':
                total_cost = data.get('total_cost', 0)
                if total_cost > 100000:
                    budget_scale = "LARGE SCALE - Significant budget management required"
                elif total_cost > 50000:
                    budget_scale = "MEDIUM SCALE - Moderate budget oversight needed"
                else:
                    budget_scale = "SMALL SCALE - Standard budget controls sufficient"
                
                content_lines.append(f"• Budget Scale: {budget_scale}")
            
            content_lines.append("")
        
        # Executive Conclusion
        content_lines.append("EXECUTIVE CONCLUSION")
        content_lines.append("-" * 30)
        
        if extensive_mode:
            # Comprehensive conclusion based on analysis type
            if analysis_type == 'summary_metrics':
                conversion_rate = data.get('conversion_rate', 0)
                if conversion_rate > 10:
                    conclusion = "Our marketing operations demonstrate strong performance with clear pathways for optimization and growth. The data supports strategic expansion initiatives while maintaining operational excellence."
                else:
                    conclusion = "Current marketing performance indicates significant improvement opportunities. Immediate focus on operational optimization will establish foundation for future growth initiatives."
            
            elif analysis_type == 'cost_analysis':
                roi = data.get('roi_percentage', 0)
                if roi > 200:
                    conclusion = "Financial performance exceeds industry benchmarks, providing strong justification for increased investment and strategic expansion. Risk-adjusted returns support aggressive growth strategies."
                else:
                    conclusion = "Financial performance requires optimization before scaling. Focus on efficiency improvements will enhance ROI and create sustainable growth foundation."
            
            elif analysis_type == 'publisher_performance':
                conclusion = "Publisher portfolio analysis reveals optimization opportunities through strategic reallocation and performance-based scaling. Data-driven channel management will maximize returns."
            
            else:
                conclusion = "Analysis provides clear direction for strategic decision-making with actionable insights for immediate implementation and long-term planning."
            
            content_lines.append(conclusion)
        else:
            content_lines.append(f"This analysis provides strategic insights with {confidence} confidence level for executive decision-making and resource allocation.")
        
        content_lines.append("")
        content_lines.append("Next Steps: Review recommendations with leadership team and establish implementation timeline.")
        content_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
        # Handle confidence as either string or dict (Claude returns dict)
        if isinstance(confidence, dict):
            confidence_text = confidence.get('level', 'Unknown')
        else:
            confidence_text = str(confidence)
        content_lines.append(f"Confidence Level: {confidence_text.title()}")
        
        final_content = "\n".join(content_lines)
        
        return {
            "format_type": "executive_summary",
            "title": "Executive Strategic Analysis",
            "content": self._sanitize_content(final_content),
            "word_count": len(final_content.split()),
            "strategic_focus": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "extensive_mode": extensive_mode
        }
    
    def _estimate_business_impact(self, recommendation: str, priority_level: str) -> str:
        """Estimate business impact for recommendations"""
        
        # High impact keywords
        high_impact_keywords = ['conversion', 'cost', 'efficiency', 'revenue', 'scale']
        
        # Medium impact keywords  
        medium_impact_keywords = ['optimize', 'improve', 'increase', 'focus']
        
        # Low impact keywords
        low_impact_keywords = ['monitor', 'maintain', 'investigate', 'review']
        
        recommendation_lower = recommendation.lower()
        
        if any(keyword in recommendation_lower for keyword in high_impact_keywords):
            if priority_level == 'high':
                return "High - Direct revenue/cost impact"
            else:
                return "Medium-High - Significant performance improvement"
        
        elif any(keyword in recommendation_lower for keyword in medium_impact_keywords):
            return "Medium - Operational efficiency gains"
        
        elif any(keyword in recommendation_lower for keyword in low_impact_keywords):
            return "Low-Medium - Risk mitigation and insights"
        
        else:
            return "Medium - General performance enhancement"
    
    def generate_all_formats(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all four communication formats from a single analysis"""
        
        all_formats = {}
        
        for format_type in self.communication_types:
            formatted_result = self.format_analysis(analysis_result, format_type)
            all_formats[format_type] = formatted_result
        
        return {
            "source_analysis": analysis_result.get('analysis_type', 'unknown'),
            "timestamp": datetime.now().isoformat(),
            "formats": all_formats,
            "total_formats": len(self.communication_types)
        } 