"""
Agent Orchestrator
Coordinates interaction between Lead Management Analyst and Writer Agent
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from .agents.lead_analyst import LeadManagementAnalyst
from .agents.writer_agent import WriterAgent
from .data.analyzer import MarketingDataAnalyzer
from .templates.report_templates import ReportTemplates

class AgentOrchestrator:
    """
    Agent Orchestrator - Coordinates the two-agent system for marketing analytics
    """
    
    def __init__(self, data: pd.DataFrame = None):
        self.lead_analyst = LeadManagementAnalyst(data)
        self.writer_agent = WriterAgent()
        self.templates = ReportTemplates()
        self.data = data
        
        # Conversation history
        self.conversation_history = []
        
    def load_data(self, data: pd.DataFrame):
        """Load new data into the system"""
        self.data = data
        self.lead_analyst.load_data(data)
        
        return {
            "status": "success",
            "message": f"Loaded {len(data)} records with {data.shape[1]} columns",
            "data_preview": data.head(3).to_dict(),
            "available_columns": list(data.columns),
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query through the Lead Management Analyst
        """
        if self.data is None:
            return {
                "error": "No data loaded. Please load data first.",
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
        
        # Get analysis from Lead Management Analyst
        analysis_result = self.lead_analyst.analyze_query(query)
        
        # Add to conversation history
        self.conversation_history.append({
            "type": "analysis",
            "query": query,
            "result": analysis_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return analysis_result
    
    def generate_report(self, analysis_result: Dict[str, Any], format_type: str) -> Dict[str, Any]:
        """
        Generate a business report using the Writer Agent
        """
        if "error" in analysis_result:
            return {
                "error": f"Cannot generate report from analysis with errors: {analysis_result['error']}",
                "format_type": format_type
            }
        
        # Format analysis using Writer Agent
        formatted_report = self.writer_agent.format_analysis(analysis_result, format_type)
        
        # Add to conversation history
        self.conversation_history.append({
            "type": "report",
            "format_type": format_type,
            "result": formatted_report,
            "timestamp": datetime.now().isoformat()
        })
        
        return formatted_report
    
    def analyze_and_report(self, query: str, format_type: str = 'daily_summary') -> Dict[str, Any]:
        """
        Complete pipeline: analyze query and generate formatted report
        """
        # Step 1: Analyze query
        analysis_result = self.analyze_query(query)
        
        if "error" in analysis_result:
            return analysis_result
        
        # Step 2: Generate report
        report = self.generate_report(analysis_result, format_type)
        
        return {
            "pipeline_type": "analyze_and_report",
            "query": query,
            "format_type": format_type,
            "analysis": analysis_result,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_all_reports(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate all four report formats from a single analysis
        """
        if "error" in analysis_result:
            return {
                "error": f"Cannot generate reports from analysis with errors: {analysis_result['error']}"
            }
        
        # Generate all formats using Writer Agent
        all_reports = self.writer_agent.generate_all_formats(analysis_result)
        
        # Add to conversation history
        self.conversation_history.append({
            "type": "all_reports",
            "result": all_reports,
            "timestamp": datetime.now().isoformat()
        })
        
        return all_reports
    
    def quick_analysis_suite(self, query: str) -> Dict[str, Any]:
        """
        Complete analysis suite: analyze query and generate all report formats
        """
        # Step 1: Analyze query
        analysis_result = self.analyze_query(query)
        
        if "error" in analysis_result:
            return analysis_result
        
        # Step 2: Generate all reports
        all_reports = self.generate_all_reports(analysis_result)
        
        return {
            "suite_type": "quick_analysis_suite",
            "query": query,
            "analysis": analysis_result,
            "reports": all_reports,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_predefined_analyses(self) -> List[Dict[str, Any]]:
        """
        Get list of predefined analysis types that can be run
        """
        return [
            {
                "name": "Intent Quality Analysis",
                "query": "What's the percentage of Level 2 and Level 3 calls?",
                "description": "Analyze lead quality and intent distribution",
                "recommended_format": "detailed_analysis"
            },
            {
                "name": "Publisher Performance Ranking",
                "query": "Which publishers have the best performance?",
                "description": "Rank publishers by performance index",
                "recommended_format": "executive_summary"
            },
            {
                "name": "Cost Efficiency Analysis",
                "query": "What is cost per sale by publisher?",
                "description": "Analyze cost efficiency across publishers",
                "recommended_format": "recommendations"
            },
            {
                "name": "Performance Outliers",
                "query": "Show me performance outliers",
                "description": "Identify statistical outliers in key metrics",
                "recommended_format": "daily_summary"
            },
            {
                "name": "Conversion Impact Factors",
                "query": "What factors are impacting conversion rates?",
                "description": "Analyze factors affecting conversion performance",
                "recommended_format": "detailed_analysis"
            },
            {
                "name": "Overall Performance Summary",
                "query": "Give me an overview of overall performance",
                "description": "Comprehensive performance metrics summary",
                "recommended_format": "executive_summary"
            }
        ]
    
    def run_predefined_analysis(self, analysis_name: str) -> Dict[str, Any]:
        """
        Run a predefined analysis by name
        """
        predefined_analyses = self.get_predefined_analyses()
        
        # Find the analysis
        selected_analysis = None
        for analysis in predefined_analyses:
            if analysis['name'] == analysis_name:
                selected_analysis = analysis
                break
        
        if not selected_analysis:
            return {
                "error": f"Unknown analysis: {analysis_name}",
                "available_analyses": [a['name'] for a in predefined_analyses]
            }
        
        # Run the analysis
        query = selected_analysis['query']
        format_type = selected_analysis['recommended_format']
        
        return self.analyze_and_report(query, format_type)
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history
        """
        return self.conversation_history[-limit:] if limit else self.conversation_history
    
    def clear_conversation_history(self):
        """
        Clear conversation history
        """
        self.conversation_history = []
        return {"status": "success", "message": "Conversation history cleared"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and capabilities
        """
        status = {
            "system_status": "operational",
            "data_loaded": self.data is not None,
            "agents": {
                "lead_analyst": "operational",
                "writer_agent": "operational"
            },
            "capabilities": {
                "natural_language_queries": True,
                "business_report_generation": True,
                "predefined_analyses": len(self.get_predefined_analyses()),
                "report_formats": len(self.writer_agent.communication_types)
            },
            "conversation_history_length": len(self.conversation_history),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.data is not None:
            status["data_info"] = {
                "records": len(self.data),
                "columns": self.data.shape[1],
                "publishers": self.data['PUBLISHER'].nunique() if 'PUBLISHER' in self.data.columns else 0,
                "total_calls": len(self.data)  # Total calls is just the number of rows
            }
        
        return status
    
    def get_sample_queries(self) -> List[str]:
        """
        Get sample queries that users can try
        """
        return [
            "What's the percentage of Level 2 and Level 3 calls?",
            "Which publishers have the best performance?",
            "What is cost per sale by publisher?",
            "Show me performance outliers",
            "What factors are impacting conversion rates?",
            "Give me an overview of overall performance",
            "Which publisher has the highest conversion rate?",
            "What are the ad misled rates by publisher?",
            "Show me the cost efficiency rankings",
            "What's the intent quality by publisher?"
        ]
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate if a query can be processed
        """
        if not query or len(query.strip()) < 3:
            return {
                "valid": False,
                "error": "Query too short. Please provide a meaningful question.",
                "suggestions": self.get_sample_queries()[:3]
            }
        
        if self.data is None:
            return {
                "valid": False,
                "error": "No data loaded. Please load data first.",
                "required_action": "load_data"
            }
        
        # Check if query contains relevant keywords
        relevant_keywords = [
            'publisher', 'performance', 'conversion', 'cost', 'sale', 'intent', 
            'level', 'quality', 'outlier', 'analysis', 'summary', 'rate'
        ]
        
        query_lower = query.lower()
        has_relevant_keywords = any(keyword in query_lower for keyword in relevant_keywords)
        
        if not has_relevant_keywords:
            return {
                "valid": True,
                "warning": "Query may not contain marketing-specific terms. Results may be general.",
                "suggestions": self.get_sample_queries()[:3]
            }
        
        return {
            "valid": True,
            "message": "Query is valid and ready for analysis"
        } 