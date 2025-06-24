# Marketing Agents System

## ✅ Complete Two-Agent Marketing Analytics System

A complete two-agent AI system for marketing analytics that combines data analysis with business communication generation. Successfully integrated into the existing `test_agents_pds.py` Streamlit application.

## 🏗️ System Architecture

- **Lead Management Analyst Agent**: Analyzes marketing data and answers business questions
- **Writer Agent**: Formats analysis into 4 business communication types  
- **Agent Orchestrator**: Coordinates the two agents seamlessly
- **Data Analyzer**: Core business logic with 7+ analysis functions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────┐ │
│  │ Lead Management │    │   Writer Agent  │    │   Data   │ │
│  │    Analyst      │◄──►│                 │◄──►│ Analyzer │ │
│  │                 │    │                 │    │          │ │
│  └─────────────────┘    └─────────────────┘    └──────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Shared Memory Store                     │
│              (Context, History, Insights)                   │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Lead Management Analyst Agent (`agents/lead_analyst.py`)
- **Purpose**: Analyzes marketing performance data and answers business questions
- **Capabilities**:
  - Intent quality analysis (Level 1, 2, 3 calls)
  - Publisher performance ranking and indexing
  - Cost efficiency analysis
  - Statistical outlier detection
  - Conversion impact factor analysis
  - Natural language query processing

### 2. Writer Agent (`agents/writer_agent.py`)
- **Purpose**: Formats analysis results into business communications
- **Formats**:
  - **Daily Summary**: "What happened yesterday" (2-3 paragraphs)
  - **Recommendations**: "What actions to take" (prioritized action items)
  - **Detailed Analysis**: "Comprehensive insights" (500-750 words)
  - **Executive Summary**: "Strategic overview" (100-150 words)

### 3. Data Analyzer (`data/analyzer.py`)
- **Purpose**: Core business logic for marketing data analysis
- **Functions**:
  - `calculate_intent_percentages()` - Level 2+3 vs total calls
  - `publisher_performance_index()` - Relative performance scoring
  - `outlier_detection()` - Statistical outliers in key metrics
  - `cost_efficiency_analysis()` - Cost per sale/quote analysis
  - `conversion_impact_factors()` - What affects conversion rates
  - `trending_anomaly_detection()` - Time-based performance changes

### 4. Agent Orchestrator (`orchestrator.py`)
- **Purpose**: Coordinates interaction between agents
- **Features**:
  - Query routing and validation
  - Agent communication protocol
  - Conversation history management
  - Predefined analysis workflows
  - Report generation pipeline

## Data Structure

The system expects CSV/Excel data with these columns:

```
PUBLISHER, TARGET, COST PER SALE, CALL COUNT, SALES RATE, QUOTE RATE, 
SALES, QUOTED, LEVEL 3 INTENT–HIGH, LEVEL 2 INTENT–MEDIUM, 
LEVEL 1 INTENT–LOW, AD MISLED, AD MISLED RATE, DURATION, IVR, REACHED AGENT
```

## Business Logic

### Statistical Thresholds
- **Minimum Volume**: 10 calls per day for meaningful statistics
- **Outlier Detection**: Values >2 standard deviations from mean
- **Performance Index**: Normalize conversion rates to 0-100 scale

### Key Calculations
- **Intent Quality Score**: `(Level3 * 3 + Level2 * 2 + Level1 * 1) / Total Calls`
- **Publisher Performance Index**: `(Publisher Conversion Rate / Average Conversion Rate) * 100`
- **Cost Efficiency Rank**: Sort publishers by Cost Per Sale ascending

## 📊 New Page: "🤖 Marketing Agents"

Your existing Streamlit app now has a new page with 4 tabs:

1. **🔍 Natural Language Analysis**: Ask questions in plain English
2. **📊 Predefined Analyses**: 6 ready-to-run business analyses  
3. **📝 Business Reports**: Generate 4 different report formats
4. **📚 History**: Track all conversations and analyses

## 🎯 Business Intelligence Capabilities

### Lead Management Analyst can answer:
- "What's the percentage of Level 2 and Level 3 calls?"
- "Which publishers have the best performance?"
- "What is cost per sale by publisher?"
- "Show me performance outliers"
- "What factors are impacting conversion rates?"

### Writer Agent generates:
- **Daily Summary**: What happened yesterday (2-3 paragraphs)
- **Recommendations**: Prioritized action items with business impact
- **Detailed Analysis**: Comprehensive 500-750 word reports
- **Executive Summary**: Strategic 100-150 word overviews

## 📈 Data Analysis Functions
- Intent quality scoring (Level 1, 2, 3 calls)
- Publisher performance indexing
- Cost efficiency analysis
- Statistical outlier detection
- Conversion impact factor analysis
- Trending anomaly detection

## 🔧 Integration Features
- **File Upload**: Works with your existing Excel/CSV upload
- **Data Validation**: Ensures data quality and provides feedback
- **Error Handling**: Graceful failures with helpful messages
- **Conversation History**: Tracks all queries and results
- **System Status**: Real-time monitoring of agent capabilities

## 📁 File Structure Created
```
marketing_agents/
├── agents/
│   ├── lead_analyst.py      # Lead Management Analyst Agent
│   └── writer_agent.py      # Writer Agent
├── data/
│   └── analyzer.py          # Core business logic
├── templates/
│   └── report_templates.py  # Business communication templates
├── orchestrator.py          # Agent coordination
├── sample_data.py          # Sample data generator (not used in app)
├── requirements_agents.txt  # Dependencies
└── README.md               # Complete documentation
```

## Usage

### 1. Integration with Streamlit App

```python
from marketing_agents.orchestrator import AgentOrchestrator

# Initialize
orchestrator = AgentOrchestrator()

# Load data
result = orchestrator.load_data(your_dataframe)

# Analyze query
analysis = orchestrator.analyze_query("What's the conversion rate by publisher?")

# Generate report
report = orchestrator.generate_report(analysis, 'executive_summary')
```

### 2. Sample Queries

The system can handle these types of questions:

#### Intent Analysis
- "What's the percentage of Level 2 and Level 3 calls?"
- "Which publishers have the best lead quality?"

#### Performance Analysis  
- "Which publishers have the best performance?"
- "Show me publisher performance relative to each other"
- "Which publishers are outliers in performance?"

#### Cost Analysis
- "What is cost per sale by publisher?"
- "What is cost per quote by publisher?"
- "Which publisher is most cost-efficient?"

#### Impact Analysis
- "What ad misled rates are impacting conversions?"
- "What factors are impacting conversion rates?"

### 3. Predefined Analyses

The system includes 6 predefined analyses:

1. **Intent Quality Analysis** → Detailed Analysis format
2. **Publisher Performance Ranking** → Executive Summary format  
3. **Cost Efficiency Analysis** → Recommendations format
4. **Performance Outliers** → Daily Summary format
5. **Conversion Impact Factors** → Detailed Analysis format
6. **Overall Performance Summary** → Executive Summary format

## Report Templates

### Daily Summary Template
```
Yesterday we processed {total_calls} calls from {publishers} publishers.

Key Performance Highlights:
• {insight_1}
• {insight_2}
• {insight_3}

Notable Events: {significant_changes}

Data confidence level: {confidence}
```

### Recommendations Template
```
PRIORITY 1 - IMMEDIATE ACTION REQUIRED:
1. {action} - Impact: {business_impact}

PRIORITY 2 - OPTIMIZATION OPPORTUNITIES:  
1. {action} - Impact: {business_impact}

MONITOR & MAINTAIN:
• {monitoring_item}

Recommendation Confidence: {confidence}
```

### Executive Summary Template
```
{strategic_overview}

KEY METRICS:
• {metric_1}
• {metric_2}

CRITICAL DECISIONS NEEDED:
• {decision_1}
• {decision_2}

STRATEGIC RECOMMENDATIONS:
• {recommendation_1}
• {recommendation_2}

Analysis Confidence: {confidence}
```

## 🚀 How to Use

1. **Upload Data**: Use your existing file upload in the sidebar
2. **Navigate**: Select "🤖 Marketing Agents" from the navigation
3. **Analyze**: Ask questions or run predefined analyses
4. **Generate Reports**: Create business communications in multiple formats
5. **Review History**: Track all conversations and insights

## 💡 Key Benefits

- **No Sample Data**: Uses your real uploaded files
- **Business-Focused**: Designed for marketing performance analysis
- **Multi-Format Output**: Different reports for different audiences
- **Natural Language**: Ask questions in plain English
- **Integrated Workflow**: Lead Analyst → Writer Agent → Business Reports

## Installation

1. Install requirements:
```bash
pip install -r marketing_agents/requirements_agents.txt
```

2. The system is integrated into `test_agents_pds.py` as a new page: "🤖 Marketing Agents"

## ✅ Ready to Run!

The system is now fully integrated and ready to use! Upload your marketing data and navigate to the "🤖 Marketing Agents" page to start using the two-agent system for comprehensive marketing analytics and business communication generation.

## Features

### Natural Language Interface
- Type questions in plain English
- Automatic query classification and routing
- Context-aware responses

### Business Intelligence
- Statistical analysis with confidence levels
- Outlier detection and anomaly identification
- Performance indexing and ranking
- Cost efficiency optimization

### Report Generation
- Multiple business communication formats
- Audience-specific content (operations, management, executives)
- Word count and confidence metadata
- Downloadable results

### Conversation History
- Track all queries and analyses
- Review previous insights
- Clear history functionality

## System Status

The Marketing Agents page shows real-time system status:
- Agent operational status
- Data loading confirmation  
- Analysis capabilities
- Conversation history length

## Error Handling

The system includes comprehensive error handling:
- Data validation and quality checks
- Query validation with suggestions
- Graceful failure modes
- Detailed error messages with context

## Future Enhancements

1. **Predictive Analytics**: Add forecasting capabilities
2. **Real-time Data**: Connect to live data sources
3. **Custom Agents**: Allow users to create specialized agents
4. **Advanced Visualizations**: Enhanced charts and graphs
5. **API Integration**: REST endpoints for external systems 