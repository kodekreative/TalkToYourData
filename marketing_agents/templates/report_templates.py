"""
Report Templates
Template definitions for business communications
"""

from typing import Dict, List, Any
from datetime import datetime

class ReportTemplates:
    """
    Report Templates - Predefined templates for business communications
    """
    
    def __init__(self):
        self.templates = {
            'daily_summary': self._get_daily_summary_template(),
            'recommendations': self._get_recommendations_template(),
            'detailed_analysis': self._get_detailed_analysis_template(),
            'executive_summary': self._get_executive_summary_template()
        }
    
    def _get_daily_summary_template(self) -> Dict[str, Any]:
        """Daily Summary Template: 'What happened yesterday'"""
        return {
            "title": "Daily Performance Summary",
            "sections": [
                {
                    "name": "opening_statement",
                    "template": "Yesterday we processed {total_calls:,} calls from {unique_publishers} publishers.",
                    "required_fields": ["total_calls", "unique_publishers"]
                },
                {
                    "name": "key_highlights",
                    "template": "Key Performance Highlights:\n{highlights}",
                    "required_fields": ["highlights"]
                },
                {
                    "name": "notable_events",
                    "template": "Notable Events: {events}",
                    "required_fields": ["events"]
                },
                {
                    "name": "confidence_level",
                    "template": "Data confidence level: {confidence}",
                    "required_fields": ["confidence"]
                }
            ],
            "target_length": "2-3 paragraphs",
            "audience": "Operations team",
            "frequency": "Daily"
        }
    
    def _get_recommendations_template(self) -> Dict[str, Any]:
        """Recommendations Template: 'What recommendations to act on'"""
        return {
            "title": "Action Recommendations",
            "sections": [
                {
                    "name": "priority_1",
                    "template": "PRIORITY 1 - IMMEDIATE ACTION REQUIRED:\n{priority_1_items}",
                    "required_fields": ["priority_1_items"]
                },
                {
                    "name": "priority_2", 
                    "template": "PRIORITY 2 - OPTIMIZATION OPPORTUNITIES:\n{priority_2_items}",
                    "required_fields": ["priority_2_items"]
                },
                {
                    "name": "monitor_maintain",
                    "template": "MONITOR & MAINTAIN:\n{monitor_items}",
                    "required_fields": ["monitor_items"]
                },
                {
                    "name": "confidence_timing",
                    "template": "Recommendation Confidence: {confidence}\nRecommended Review Frequency: {frequency}",
                    "required_fields": ["confidence", "frequency"]
                }
            ],
            "target_length": "Maximum 5 prioritized action items",
            "audience": "Management team",
            "frequency": "Daily/Weekly"
        }
    
    def _get_detailed_analysis_template(self) -> Dict[str, Any]:
        """Detailed Analysis Template: 'Detailed analysis of yesterday's actions and insights'"""
        return {
            "title": "Detailed Performance Analysis",
            "sections": [
                {
                    "name": "executive_overview",
                    "template": "EXECUTIVE OVERVIEW\n==================================================\n{overview_content}",
                    "required_fields": ["overview_content"]
                },
                {
                    "name": "performance_deep_dive",
                    "template": "PERFORMANCE DEEP-DIVE\n==================================================\n{performance_content}",
                    "required_fields": ["performance_content"]
                },
                {
                    "name": "statistical_analysis",
                    "template": "STATISTICAL ANALYSIS\n==================================================\n{statistical_content}",
                    "required_fields": ["statistical_content"]
                },
                {
                    "name": "supporting_data",
                    "template": "SUPPORTING DATA\n==================================================\n{data_content}",
                    "required_fields": ["data_content"]
                },
                {
                    "name": "strategic_recommendations",
                    "template": "STRATEGIC RECOMMENDATIONS\n==================================================\n{recommendations_content}",
                    "required_fields": ["recommendations_content"]
                }
            ],
            "target_length": "500-750 words with supporting data tables",
            "audience": "Analytics team and management",
            "frequency": "Weekly"
        }
    
    def _get_executive_summary_template(self) -> Dict[str, Any]:
        """Executive Summary Template: 'Executive summary from detailed analysis'"""
        return {
            "title": "Executive Summary",
            "sections": [
                {
                    "name": "strategic_overview",
                    "template": "{strategic_overview}",
                    "required_fields": ["strategic_overview"]
                },
                {
                    "name": "key_metrics",
                    "template": "KEY METRICS:\n{key_metrics}",
                    "required_fields": ["key_metrics"]
                },
                {
                    "name": "critical_decisions",
                    "template": "CRITICAL DECISIONS NEEDED:\n{critical_decisions}",
                    "required_fields": ["critical_decisions"]
                },
                {
                    "name": "strategic_recommendations",
                    "template": "STRATEGIC RECOMMENDATIONS:\n{strategic_recommendations}",
                    "required_fields": ["strategic_recommendations"]
                },
                {
                    "name": "analysis_confidence",
                    "template": "Analysis Confidence: {confidence}",
                    "required_fields": ["confidence"]
                }
            ],
            "target_length": "100-150 words, strategic focus only",
            "audience": "Executive leadership",
            "frequency": "Weekly/Monthly"
        }
    
    def get_template(self, template_type: str) -> Dict[str, Any]:
        """Get a specific template"""
        return self.templates.get(template_type, {})
    
    def get_all_templates(self) -> Dict[str, Any]:
        """Get all available templates"""
        return self.templates
    
    def validate_template_data(self, template_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that data contains all required fields for a template"""
        template = self.get_template(template_type)
        
        if not template:
            return {
                "valid": False,
                "error": f"Unknown template type: {template_type}",
                "available_templates": list(self.templates.keys())
            }
        
        missing_fields = []
        
        for section in template.get('sections', []):
            required_fields = section.get('required_fields', [])
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)
        
        if missing_fields:
            return {
                "valid": False,
                "error": f"Missing required fields: {missing_fields}",
                "template_type": template_type
            }
        
        return {
            "valid": True,
            "template_type": template_type,
            "sections_count": len(template.get('sections', []))
        } 