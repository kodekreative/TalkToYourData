# 💬 Talk to Your Data

A powerful Streamlit application that enables natural language conversations with Excel data. Upload your spreadsheets, ask questions in plain English, and get intelligent responses with visualizations.

## ✨ Features

### 🔧 Core Functionality
- **Excel Import**: Support for both .xlsx and .xls files with multi-sheet handling
- **Natural Language Processing**: Ask questions in plain English and get intelligent responses
- **Smart Visualizations**: Automatic chart generation based on query intent
- **Business Context Mapping**: Map your data columns to business terms for better understanding
- **Conversation Memory**: Maintain context across multiple questions
- **Data Quality Assessment**: Automatic data profiling and quality scoring

### 🎯 Query Capabilities
- **Aggregations**: "What's the total revenue by region?"
- **Trends**: "Show me sales trends over time"
- **Comparisons**: "Compare Q1 vs Q2 performance"
- **Filtering**: "Show me customers from California"
- **Correlations**: "What's the relationship between price and quantity?"
- **Distributions**: "Show me the distribution of order values"

### 📊 Visualization Types
- Interactive bar charts, line charts, scatter plots
- Pie charts for categorical breakdowns
- Histograms for distributions
- Heatmaps for correlations
- Box plots for statistical analysis
- Automatic chart type selection based on data

### 🧠 AI Integration
- **OpenAI GPT**: Primary LLM for query understanding
- **Rule-based Fallback**: Works without API keys using pattern matching
- **Context-aware**: Uses business mappings for better interpretation
- **Code Generation**: Shows pandas code for transparency

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda for package management

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd TalkToYourData
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up API keys (optional but recommended)**
```bash
# Copy the example environment file
cp env_example.txt .env

# Edit .env with your API keys
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
TalkToYourData/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── env_example.txt                 # Environment variables template
├── README.md                       # This file
│
├── src/                           # Source code modules
│   ├── data_processor.py          # Excel import and processing
│   ├── nlp_handler.py             # Natural language processing
│   ├── visualizer.py              # Chart generation
│   ├── config_manager.py          # Business context management
│   └── query_engine.py            # Query coordination
│
├── config/                        # Configuration files
│   ├── business_terms.json        # Business term definitions
│   ├── data_schema.yaml          # Data validation rules
│   └── visualization_templates.json # Chart templates
│
└── data/                          # Data directory
    └── uploads/                   # Temporary file storage
```

## 🔧 Configuration

### Business Terms
The system uses configurable business terms to understand your data better. Edit `config/business_terms.json` to:
- Define field mappings (e.g., "revenue" maps to ["sales", "income", "earnings"])
- Set data types and validation rules
- Configure visualization preferences

### Data Schema
Configure data validation and quality checks in `config/data_schema.yaml`:
- Set missing value thresholds
- Define outlier detection methods
- Configure business metrics and KPIs

## 💡 Usage Examples

### Getting Started
1. **Upload your Excel file** using the sidebar
2. **Configure business context** by mapping columns to business terms
3. **Ask questions** using the chat interface

### Sample Questions
- "What is our total revenue?"
- "Show me sales by region"
- "How has revenue changed over time?"
- "What's the correlation between price and quantity?"
- "Which products are our best sellers?"
- "Show me customer distribution by region"

### Advanced Features
- **Executive Summary**: Generate automatic insights and recommendations
- **Export Reports**: Download analysis results as JSON
- **Data Quality**: View data profiling and quality scores
- **Conversation History**: Review previous queries and results

## 🤖 AI Integration

### OpenAI GPT (Recommended)
For the best experience, configure OpenAI API access:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Alternative LLMs
The application also supports:
- **Anthropic Claude**: Set `ANTHROPIC_API_KEY`
- **Google Gemini**: Set `GOOGLE_API_KEY`
- **Rule-based Fallback**: Works without any API keys

### Without API Keys
The application includes a rule-based query processor that works without external APIs, though with limited natural language understanding.

## 📊 Data Requirements

### Supported Formats
- **Excel Files**: .xlsx, .xls
- **Multiple Sheets**: Automatic sheet selection
- **Data Types**: Numeric, text, dates, categories

### Data Structure
- **Headers**: First row should contain column names
- **Clean Data**: Remove completely empty rows/columns
- **Consistent Types**: Each column should have consistent data types

### Best Practices
- Use descriptive column names
- Ensure data consistency
- Handle missing values appropriately
- Include date columns for time-series analysis

## 🛠️ Development

### Adding New Features
1. **Data Processing**: Extend `src/data_processor.py`
2. **Query Types**: Add patterns to `src/nlp_handler.py`
3. **Visualizations**: Create new chart types in `src/visualizer.py`
4. **Business Logic**: Configure terms in `config/business_terms.json`

### Testing
```bash
# Run basic functionality tests
python -m pytest tests/

# Test with sample data
streamlit run app.py
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🔒 Security & Privacy

### Data Handling
- **Local Processing**: Data stays on your machine
- **No Data Storage**: Files are processed in memory
- **API Security**: API keys stored in environment variables
- **Optional Cloud**: LLM calls are optional (rule-based fallback available)

### Recommendations
- Don't commit API keys to version control
- Use environment variables for configuration
- Review generated code before execution
- Be cautious with sensitive data

## 📈 Performance

### Optimization Tips
- **File Size**: Larger files may take longer to process
- **Memory Usage**: Monitor memory for very large datasets
- **API Limits**: Be aware of LLM API rate limits
- **Caching**: Results are cached within sessions

### Scalability
- **Local Development**: Suitable for datasets up to 100MB
- **Production**: Consider memory and processing requirements
- **Cloud Deployment**: Can be deployed to cloud platforms

## 🆘 Troubleshooting

### Common Issues

**File Upload Errors**
- Ensure file is valid Excel format
- Check for completely empty sheets
- Verify file isn't corrupted

**API Key Issues**
- Verify API key is correctly set in .env file
- Check API key permissions and quotas
- Use rule-based fallback if needed

**Performance Issues**
- Try smaller datasets for testing
- Close other memory-intensive applications
- Consider data preprocessing

**Visualization Errors**
- Ensure data has appropriate types for charts
- Check for missing or invalid values
- Try different chart types manually

### Getting Help
1. Check the error messages in the application
2. Review the console output for detailed errors
3. Verify your data format and structure
4. Test with smaller sample files first

## 📝 License

This project is open source. See LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the excellent web app framework
- **Plotly**: For interactive visualizations
- **OpenAI**: For natural language processing capabilities
- **Pandas**: For data manipulation and analysis

## 🔮 Future Enhancements

- **More Data Sources**: CSV, JSON, database connections
- **Advanced Analytics**: Statistical tests, forecasting
- **Collaboration**: Multi-user sessions and sharing
- **Custom Dashboards**: Saved dashboard templates
- **Mobile Support**: Responsive design improvements 