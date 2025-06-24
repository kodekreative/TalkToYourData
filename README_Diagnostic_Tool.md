# Performance Marketing Diagnostic Tool

A comprehensive Streamlit application that diagnoses performance marketing issues by analyzing Publisher-Buyer-Target combinations to identify whether conversion problems stem from lead quality issues (publisher-side) or sales execution problems (buyer/agent-side).

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_diagnostic.txt

# Generate sample data (optional)
python generate_sample_data.py

# Run the application
streamlit run performance_diagnostic_app.py
```

### 2. Upload Your Data

The tool expects an Excel file with the following columns:

#### Required Columns:
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

#### Optional Columns (for advanced features):
- **QUOTE**: Yes/No - Was quote provided
- **STAGE_1** through **STAGE_5**: Yes/No - Stage progression tracking
- **CALL_DATE**: Date of the call

## 🎯 Key Features

### 1. **Critical Alert System**
Automatically identifies and flags critical issues requiring immediate attention:

- **🚨 Ad Misled Issues**: Tracks publisher compliance problems
- **🚨 Agent Availability Crisis**: Identifies buyers with staffing issues
- **⚠️ High-Value Lead Waste**: Flags wasted Level 2 & 3 leads  
- **⚠️ Poor Agent Performance**: Tracks objection handling issues

### 2. **Combination Performance Analysis**
- **Publisher × Buyer Performance Matrix**: Heatmap visualization
- **Publisher × Buyer × Target Analysis**: 3-way combination insights
- **Top/Bottom Performer Identification**: Statistical ranking
- **Performance Anomaly Detection**: Statistical outlier identification

### 3. **Lead Quality Assessment**
- **Publisher Quality Rankings**: Conversion rate, strong lead rate, billable rate
- **Customer Intent Analysis**: Performance by intent level
- **Quality Score Calculation**: Composite scoring algorithm
- **Ad Compliance Monitoring**: Track and quantify ad misled incidents

### 4. **Sales Execution Analysis**
- **Buyer Performance Rankings**: Conversion rates and efficiency metrics
- **Agent Availability Analysis**: Staffing adequacy assessment
- **Level 3 Conversion Tracking**: High-value lead performance
- **Training Needs Assessment**: Identify buyers requiring intervention

### 5. **Advanced Analytics**
- **Stage Progression Analysis**: Funnel conversion tracking
- **Quote Ratio Analysis**: Quote-to-call and quote-to-sale ratios
- **Statistical Anomaly Detection**: Identify unusual performance patterns
- **Duration Analysis**: Call length correlation with outcomes

### 6. **Executive Reporting**
- **Performance Overview**: Key metrics dashboard
- **Critical Findings Summary**: Automated insight generation
- **Strategic Recommendations**: Actionable business advice
- **Immediate Action Items**: Prioritized intervention list

## 📊 Business Intelligence Framework

### Performance Metrics
- **Conversion Rate**: Sales ÷ Leads (primary KPI)
- **Quote-to-Call Ratio**: Lead quality and agent engagement proxy
- **Quote-to-Sale Ratio**: Closing ability vs lead quality indicator
- **Stage Progression Rates**: Lead and agent quality assessment

### Critical Thresholds
- **Agent Availability**: <80% triggers critical alert
- **Level 3 Conversion**: <60% indicates training issues
- **Ad Misled Rate**: >5% requires immediate publisher review
- **High-Value Lead Waste**: >50% flags optimization opportunities

## 🔍 Diagnostic Logic

### Lead Quality Issues (Publisher-Side)
- **Low Customer Intent Distribution**: High Level 1/Negative Intent rates
- **Poor Billable Rates**: Non-billable leads indicate quality issues
- **High Ad Misled Rates**: Compliance and targeting problems
- **Short Call Durations**: Indicates poor lead qualification

### Sales Execution Issues (Buyer/Agent-Side)
- **Low Agent Availability**: Staffing and capacity problems
- **Poor Level 3 Conversion**: Training and skill gaps
- **High Objection Without Rebuttal**: Sales technique deficiencies
- **Low Quote-to-Sale Ratios**: Closing skill issues

## 📈 Usage Workflows

### Daily Operations
1. Upload latest performance data
2. Review Critical Alert Dashboard
3. Investigate flagged issues
4. Analyze combination performance
5. Generate action items

### Strategic Review
1. Analyze performance trends over time
2. Identify optimization opportunities
3. Review Publisher-Buyer relationships
4. Plan capacity adjustments
5. Generate executive reports

## 🛠️ Technical Architecture

### Core Components
- **PerformanceDiagnostic**: Main analysis engine
- **AdvancedDiagnostics**: Extended analytics capabilities
- **Alert System**: Critical issue detection
- **Visualization Engine**: Interactive charts and heatmaps

### Data Processing
- **Automatic column standardization**
- **Data cleaning and validation**
- **Missing value handling**
- **Derived field generation**

## 📋 Sample Data

Use the included sample data generator to create test data:

```python
python generate_sample_data.py
```

This creates realistic test data with:
- 6 Publishers with different quality profiles
- 4 Buyers with varying efficiency levels
- 4 Target verticals
- Realistic conversion patterns and issues
- Edge cases for testing alerts

## 🎨 User Interface

### Navigation Pages
1. **Dashboard Overview**: Key metrics and quick alerts
2. **Alert Center**: Detailed critical issue management
3. **Combination Analysis**: Publisher-Buyer performance matrix
4. **Lead Quality Analysis**: Publisher quality assessment
5. **Sales Execution Analysis**: Buyer performance evaluation
6. **Advanced Analysis**: Stage progression and quote ratios
7. **Executive Summary**: Strategic insights and recommendations

### Interactive Features
- **Clickable heatmaps** for drill-down analysis
- **Sortable performance tables**
- **Real-time metric updates**
- **Exportable visualizations**
- **Filter controls** for date ranges and segments

## 🚨 Alert Categories

### Critical (Immediate Action)
- Agent availability <80%
- Ad misled rate >5%
- Multiple Level 3 conversion failures

### High Priority
- High-value lead waste >50%
- Poor publisher quality scores
- Significant performance anomalies

### Medium Priority
- Training needs identification
- Optimization opportunities
- Trend changes

## 📊 Key Visualizations

- **Performance Heatmaps**: Publisher × Buyer conversion matrices
- **Funnel Charts**: Stage progression analysis
- **Scatter Plots**: Lead volume vs conversion relationships
- **Bar Charts**: Performance rankings and comparisons
- **Pie Charts**: Customer intent and issue distributions
- **Time Series**: Trend analysis (when date data available)

## 🔧 Customization

The tool can be extended for specific business needs:

- **Custom Alert Thresholds**: Modify business rules
- **Additional Metrics**: Add industry-specific KPIs
- **New Visualizations**: Extend chart capabilities
- **Integration Points**: Connect to existing systems

## 📞 Support

For questions about implementation or customization, refer to the detailed code comments and the business logic framework outlined in the diagnostic engine.

---

**Built with:** Streamlit, Pandas, Plotly, NumPy, and advanced statistical analysis libraries.

**Purpose:** Provide comprehensive visibility into performance marketing operations, enabling data-driven decisions about publisher relationships, buyer capacity, and sales training needs while immediately flagging critical issues. 