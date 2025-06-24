# Performance Marketing Analysis Dictionary

## *Business Logic Reference for Multi-Agent Analysis*

---

## 🎯 Core Entity Fields

### **PUBLISHER** - The affiliate or traffic source generating leads
- **Data Type**: Text
- **Business Impact**: Primary determinant of lead quality
- **Analysis Role**: Source of traffic quality issues vs execution problems

### **BUYER** - The client company purchasing the leads
- **Data Type**: Text  
- **Business Impact**: Controls sales execution and operational capacity
- **Analysis Role**: Responsible for conversion optimization and agent performance

### **TARGET** - The specific call center division within the buyer organization
- **Data Type**: Text
- **Business Impact**: Operational unit handling the leads
- **Analysis Role**: Division-level performance analysis within buyers

### **CAMPAIGN** - The marketing initiative driving the traffic
- **Data Type**: Text
- **Business Impact**: Campaign-level optimization insights

---

## 📈 Primary Success Metrics

### **Conversion Rate** (Primary KPI)
**Formula**: `(Number of Sales ÷ Number of Leads) × 100`
- **Measurement**: Percentage of calls that result in sales
- **Success Threshold**: Varies by industry (typically 5-15%)
- **Analysis Use**: Primary measure of Publisher-Buyer combination success

### **Successful Combinations Analysis**
**Definition**: Publisher-Buyer pairs or Publisher-Buyer-Target triplets with optimal conversion rates
- **Measurement**: Statistical comparison of conversion rates across combinations
- **Success Indicators**:
  - Above-average conversion rates
  - High lead quality scores
  - Efficient sales execution
  - Low issue rates

---

## 🔍 Lead Quality Indicators

### **CUSTOMER INTENT** (Primary Quality Metric)
**Values**: "Level 1", "Level 2", "Level 3", "Negative Intent", "Not Detected"
- **Business Logic**:
  - **Level 3**: Best intent - highest-quality leads requiring immediate attention
  - **Level 2**: Medium to high intent - moderate-quality leads with good conversion potential
  - **Level 1, Negative Intent, Not Detected**: Poor quality leads with minimal conversion potential
- **Quality Threshold**: Only Level 2 & 3 considered strong leads
- **Critical Rule**: **Level 3 non-conversions indicate sales execution problems**
- **Action Required**: All Level 2 & 3 leads need immediate follow-up

### **BILLABLE** 
**Values**: "Yes", "No"
- **Business Logic**: "No" = Publisher sending very poor quality leads
- **Quality Indicator**: Billable rate should be >70% for healthy publishers
- **Action Trigger**: Low billable rates require publisher review

### **DURATION** 
**Format**: "HH:MM:SS"
- **Business Logic**: Longer duration = better lead quality AND better sales execution
- **Quality Proxy**: Extended conversations indicate engagement
- **Benchmark**: >5 minutes typically indicates qualified interest

### **AD MISLED**
**Values**: "Yes", "No"
- **Business Logic**: "Yes" = Critical compliance violation
- **Action Required**: **Immediate publisher review and correction**
- **Reporting**: Must quantify total count and highlight offending publishers
- **Escalation**: **Zero tolerance policy** for misleading advertising

---

## 💼 Sales Execution Indicators

### **Quote-to-Call Ratio**
**Formula**: `(QUOTE = "Yes" count ÷ Total Calls) × 100`
- **Business Logic**: Measures agent engagement and lead qualification
- **Benchmark**: Should be 40-60% for healthy operations

### **Quote-to-Sale Ratio**
**Formula**: `(SALE = "Yes" count ÷ QUOTE = "Yes" count) × 100`
- **Business Logic**: Measures closing ability
- **Critical Analysis**: If Quote-to-Call ≠ Quote-to-Sale, indicates closing problems
- **Action Trigger**: <30% suggests sales training needed

### **OBJECTION WITH NO REBUTTAL**
**Values**: "Yes", "No"
- **Business Logic**: "Yes" = Poor agent performance indicator
- **Action Required**: Sales training and coaching needed
- **Performance Impact**: Directly correlates with lost sales opportunities

---

## 🔧 Operational Efficiency Indicators

### **REACHED AGENT**
**Values**: "Yes", "No"
- **Business Logic**: "No" = Buyer capacity/availability problem
- **Critical Threshold**: **>90% reach rate required**
- **Escalation Trigger**: Immediate management attention for low rates
- **Impact**: Direct revenue loss from missed opportunities

### **IVR**
**Values**: "Yes", "No"  
- **Business Logic**: "Yes" = Buyer may lack available agents
- **Analysis Focus**: Identify buyers susceptible to high IVR rates
- **Capacity Indicator**: High IVR rates suggest staffing issues

### **WAIT IN SEC**
**Data Type**: Numeric (seconds)
- **Business Logic**: Extended wait times reduce conversion rates
- **Benchmark**: <30 seconds optimal, >60 seconds problematic

---

## 📊 Sales Process Quality Indicators

### **Stage Progression Analysis**
**Fields**: STAGE 1 through STAGE 5
- **Business Logic**: Higher stages = better performance (both lead quality and agent skill)
- **Quality Ranking**:
  - **Stage 5 (Enrollment)**: Highest quality outcome
  - **Stage 4 (Plan Detail)**: Strong engagement
  - **Stage 3 (Needs Analysis)**: Good qualification
  - **Stage 2 (Eligibility)**: Basic qualification
  - **Stage 1 (Introduction)**: Initial contact only

### **Stage Progression Rate**
**Formula**: `Stage N completions ÷ Stage N-1 completions × 100`
- **Analysis Use**: Identify where leads drop off in the funnel
- **Performance Indicator**: Higher stage completion = better execution

---

## 📋 Key Performance Ratios & Derived Metrics

### **Lead Quality Score**
**Components**:
- Customer Intent Level (40% weight)
- Billable Rate (25% weight) 
- Duration (20% weight)
- Stage Progression (15% weight)

### **Sales Execution Score**
**Components**:
- Quote-to-Sale Ratio (30% weight)
- Agent Reach Rate (25% weight)
- Stage Progression (25% weight)
- Objection Handling (20% weight)

### **Publisher Quality Index**
**Formula**: `(Level 2+3 Intent Rate × 0.4) + (Billable Rate × 0.3) + (Avg Duration Score × 0.2) + (Stage Progression × 0.1) - (Ad Misled Penalty × 0.5)`

### **Buyer Efficiency Index**  
**Formula**: `(Agent Reach Rate × 0.3) + (Quote-to-Sale Rate × 0.3) + (Stage Progression × 0.2) + (Duration Utilization × 0.2) - (IVR Penalty × 0.3)`

---

## ⚠️ Critical Business Rules & Thresholds

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

### **Performance Benchmarks**
- **Conversion Rate**: 5-15% (industry dependent)
- **Lead Quality Rate**: >50% Level 2+3 intent
- **Sales Execution**: >30% quote-to-sale conversion
- **Operational Efficiency**: >90% agent reach rate
- **Billable Rate**: >70% for healthy publishers
- **Ad Compliance**: 0% tolerance for violations

---

## 🤖 Analysis Framework: Publisher vs Buyer Issues

### **Publisher Quality Problems** (Lead Generation Issues)
**Indicators**:
- Low billable rates (<50%)
- High ad misled rates (>5%)
- Low customer intent levels (<30% Level 2+3)
- Short call durations (<3 minutes average)
- Poor stage 1-2 progression

**Root Cause**: Traffic quality, targeting, or compliance issues

### **Buyer Execution Problems** (Sales Performance Issues)
**Indicators**:
- Good billable rates (>70%) but low conversion
- High Level 2+3 intent but low sales
- Poor quote-to-sale ratios (<30%)
- High objection-no-rebuttal rates
- Poor stage 3-5 progression
- Low agent reach rates

**Root Cause**: Sales training, capacity, or process issues

### **Combination Analysis Success Factors**
**Optimal Combinations Exhibit**:
- High publisher lead quality scores (>70)
- High buyer execution efficiency (>80)
- Conversion rates >1.5x industry average
- Low issue rates across all categories
- Strong stage progression through all 5 stages

**Problem Combinations Show**:
- Misaligned expectations (high-touch leads to low-capacity buyers)
- Quality-efficiency mismatch (premium leads to junior agents)
- Capacity constraints (good leads to understaffed targets)

---

## 💡 Voice Assistant Query Examples

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

### **Performance Analysis**
- *"How many sales did we have today?"*
- *"What's our overall conversion rate?"*
- *"Show me publisher quality rankings"*
- *"Which buyers need more agent capacity?"*

### **Compliance Monitoring**
- *"Any ad misled violations this week?"*
- *"Show me publishers with billable rate issues"*
- *"Which combinations are underperforming?"*

---

## 📈 Multi-Agent Analysis Methodology

This framework enables the multi-agent system to:

1. **Automatically identify** whether issues stem from lead quality (publisher) or sales execution (buyer)
2. **Rank combinations** by success metrics and quality indicators
3. **Trigger immediate actions** based on critical thresholds
4. **Provide actionable insights** for optimization
5. **Monitor compliance** with zero-tolerance policies
6. **Benchmark performance** against industry standards

### **Agent Specializations**:
- **Sales Performance Agent**: Overall conversion analysis, revenue tracking, combination success
- **Quality Analysis Agent**: Lead quality assessment, compliance monitoring, publisher ranking
- **Comparative Analysis Agent**: Publisher vs buyer performance, efficiency analysis, problem identification
- **Executive Summary Agent**: Synthesis of findings, immediate action items, strategic insights

---

*This framework enables sophisticated analysis to distinguish between lead quality issues (publisher responsibility) and sales execution issues (buyer responsibility) while identifying the optimal combinations for maximum conversion success.* 