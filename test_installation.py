#!/usr/bin/env python3
"""
Test script to verify the Talk to Your Data installation.
Run this script to check if all dependencies are installed correctly.
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'matplotlib',
        'seaborn',
        'scipy',
        'sklearn',
        'yaml',
        'json',
        'openpyxl',
        'xlrd'
    ]
    
    optional_packages = [
        'openai',
        'anthropic',
        'google.generativeai'
    ]
    
    print("🔍 Testing required package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'yaml':
                importlib.import_module('yaml')
            elif package == 'json':
                import json
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    print("\n🔍 Testing optional package imports...")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ⚠️  {package} (optional - AI features may be limited)")
    
    return failed_imports

def test_file_structure():
    """Test if the required file structure exists."""
    
    print("\n🔍 Testing file structure...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'src/data_processor.py',
        'src/nlp_handler.py',
        'src/visualizer.py',
        'src/config_manager.py',
        'src/query_engine.py',
        'config/business_terms.json',
        'config/data_schema.yaml',
        'config/visualization_templates.json'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    return missing_files

def test_module_imports():
    """Test if custom modules can be imported."""
    
    print("\n🔍 Testing custom module imports...")
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent / "src"))
    
    custom_modules = [
        'data_processor',
        'nlp_handler',
        'visualizer',
        'config_manager',
        'query_engine'
    ]
    
    failed_modules = []
    
    for module in custom_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_modules.append(module)
    
    return failed_modules

def create_sample_data():
    """Create a small sample Excel file for testing."""
    
    print("\n🔍 Creating sample data...")
    
    try:
        import pandas as pd
        
        # Create sample data
        sample_data = {
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'revenue': [1000 + i * 10 + (i % 7) * 50 for i in range(100)],
            'region': ['North', 'South', 'East', 'West'] * 25,
            'product': ['Product A', 'Product B', 'Product C'] * 33 + ['Product A'],
            'customer': [f'Customer_{i//10}' for i in range(100)],
            'quantity': [5 + (i % 10) for i in range(100)]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Create data directory if it doesn't exist
        Path('data').mkdir(exist_ok=True)
        
        # Save to Excel
        df.to_excel('data/sample_data.xlsx', index=False)
        print("  ✅ Sample data created: data/sample_data.xlsx")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to create sample data: {e}")
        return False

def run_basic_functionality_test():
    """Test basic functionality without Streamlit."""
    
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from data_processor import DataProcessor
        import pandas as pd
        
        # Test data processor
        processor = DataProcessor()
        print("  ✅ DataProcessor initialized")
        
        # Create test dataframe
        test_data = pd.DataFrame({
            'revenue': [100, 200, 150, 300],
            'region': ['North', 'South', 'East', 'West'],
            'date': pd.date_range('2023-01-01', periods=4)
        })
        
        # Test business term mapping
        mappings = processor.map_business_terms(test_data)
        print(f"  ✅ Business term mapping: {mappings}")
        
        # Test data profiling
        profile = processor.get_data_profile(test_data)
        print(f"  ✅ Data profiling completed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🚀 Talk to Your Data - Installation Test\n")
    print("=" * 50)
    
    # Test imports
    failed_imports = test_imports()
    
    # Test file structure
    missing_files = test_file_structure()
    
    # Test custom modules
    failed_modules = test_module_imports()
    
    # Create sample data
    sample_created = create_sample_data()
    
    # Test basic functionality
    functionality_ok = run_basic_functionality_test()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    if failed_imports:
        print(f"❌ Missing required packages: {', '.join(failed_imports)}")
        print("   Run: pip install -r requirements.txt")
    else:
        print("✅ All required packages installed")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
    else:
        print("✅ All required files present")
    
    if failed_modules:
        print(f"❌ Module import issues: {', '.join(failed_modules)}")
    else:
        print("✅ All custom modules importable")
    
    if sample_created:
        print("✅ Sample data created successfully")
    else:
        print("❌ Sample data creation failed")
    
    if functionality_ok:
        print("✅ Basic functionality test passed")
    else:
        print("❌ Basic functionality test failed")
    
    # Overall status
    if not failed_imports and not missing_files and not failed_modules and functionality_ok:
        print("\n🎉 INSTALLATION TEST PASSED!")
        print("You can now run: streamlit run app.py")
    else:
        print("\n⚠️  INSTALLATION ISSUES DETECTED")
        print("Please fix the issues above before running the application.")
    
    if sample_created:
        print("\n💡 TIP: You can test the app with data/sample_data.xlsx")

if __name__ == "__main__":
    main() 