# TalkToYourData - Modal Fixed Version 🛠️

A comprehensive multi-agent performance marketing analysis tool with **fixed modal/popup UI issues**. This version specifically addresses scrolling problems and button positioning issues in popup wizards and dialogs.

## 🎯 Key Features

### ✅ **Fixed Modal Issues**
- **Proper scrolling** in popup content areas
- **Always visible buttons** at bottom of modals
- **Responsive design** for mobile and desktop
- **Better UX** with keyboard navigation (ESC to close)
- **Automatic detection** and fixing of modal issues

### 📊 **Core Functionality**
- Multi-agent AI analysis system
- Voice input and response
- Data upload and processing
- Performance marketing insights
- Interactive visualizations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd TalkToYourData-ModalFixed

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env_example.txt .env
# Edit .env with your API keys

# Run the application
streamlit run app.py
```

### Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 🛠️ Modal Fix Details

This version includes comprehensive modal fixes that automatically:

1. **Detect modal elements** using multiple selectors
2. **Apply proper CSS styling** for scrolling and positioning
3. **Ensure responsive behavior** on all screen sizes
4. **Handle keyboard navigation** (ESC key to close)
5. **Prevent body scroll** when modals are open

### Files Included:
- `modal_fix.css` - Complete CSS solution
- `modal_fix.js` - JavaScript auto-detection and fixes
- `modal_template.html` - Example of proper modal structure
- `MODAL_FIX_README.md` - Detailed implementation guide

## 📁 Application Files

### Main Applications
- `app.py` - Multi-agent marketing analysis (with modal fixes)
- `simple_app.py` - Simplified data analysis interface
- `pandasai_app.py` - PandasAI integration

### Core Components
- `multi_agent_core.py` - Multi-agent system core
- `multi_agent_voice.py` - Voice interface integration
- `config/` - Configuration files and templates
- `data/` - Sample data and uploads directory
- `src/` - Source utilities

## 🎮 Usage

1. **Start the application**: `streamlit run app.py`
2. **Upload your data** (CSV or Excel format)
3. **Select agent analyses** to run
4. **Use voice input** for natural language queries
5. **View results** with properly scrollable modals

## 🔧 Modal Fix Integration

The modal fixes are automatically applied when you run any of the applications. No additional setup required!

### For Developers

If you want to add modal fixes to other Streamlit apps:

```python
# Add this to your Streamlit app
st.markdown(MODAL_FIX_CSS, unsafe_allow_html=True)
st.markdown(MODAL_FIX_JS, unsafe_allow_html=True)
```

## 📱 Responsive Design

The modal fixes work on:
- ✅ Desktop browsers
- ✅ Mobile devices
- ✅ Tablets
- ✅ Different screen orientations

## 🆘 Troubleshooting

### Modal Still Not Working?
1. Check browser console for errors
2. Ensure JavaScript is enabled
3. Try refreshing the page
4. Verify the modal uses standard Streamlit components

### Installation Issues?
1. Use Python 3.8+ 
2. Install dependencies: `pip install -r requirements.txt`
3. Check API keys in `.env` file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test modal functionality
5. Submit a pull request

## 📄 License

MIT License - feel free to use and modify

## 🙏 Acknowledgments

- Modal fixes designed to solve common Streamlit popup UX issues
- Built on Streamlit, OpenAI, and other open-source tools
- Responsive design principles for better user experience

---

**Ready to use!** 🚀 This version eliminates the frustrating modal scrolling and button positioning issues. # TTYD-Ver-2-pds
