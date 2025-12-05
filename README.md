# ArymaEdge AI - Marketing Optimization Agent

A sophisticated AI agent that optimizes marketing spends across multiple channels to achieve target sales using LangGraph, Google Gemini Pro, and advanced optimization algorithms.

## 🚀 Features

- **Multi-Channel Optimization**: Optimizes spends across TV, Meta Video, Paid Search, Outdoor, Display, Radio, and YouTube
- **Two Optimization Types**: 
  - Default optimization based on historical data
  - ROI-based optimization using performance metrics
- **Risk Assessment**: Warns about spends exceeding historical limits
- **Detailed Reporting**: Generates comprehensive reports with LLM analysis
- **Beautiful UI**: Streamlit-based web interface for easy interaction

## 📊 Demo App

### Quick Start

1. **Activate Virtual Environment**:
   ```bash
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source .venv/bin/activate     # Linux/Mac
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Demo App**:
   ```bash
   streamlit run app.py
   ```

4. **Open in Browser**: The app will open at `http://localhost:8501`

### How to Use the Demo

1. **Set Target Sales**: Enter your target sales amount in the sidebar
2. **Choose Optimization Type**: Select between Default or ROI-based optimization
3. **Run Optimization**: Click the "Run Optimization" button
4. **Review Results**: See optimized spends, warnings, and predictions
5. **Generate Report**: Click "Generate Detailed Report" for comprehensive analysis

## 🏗️ Architecture

### Agent Flow
```
Input Parameters → Optimization → Warning Check → Report Generation
```

### Components
- **Agent Nodes**: Modular functions for each step of the optimization process
- **State Management**: LangGraph state management for data flow
- **LLM Integration**: Google Gemini Pro for natural language processing and report generation
- **Data Sources**: Historical marketing data and ROI metrics

## 📁 Project Structure

```
ArymaEdge_AI/
├── Agent_AI/
│   └── agent.py              # Main agent implementation
├── data/
│   ├── marketing_roas.csv    # ROI performance data
│   ├── Model_data_for_simulator.csv  # Historical marketing data
│   └── data_bounds.csv       # Data constraints
├── optimizers/
│   ├── __init__.py
│   ├── default_optimizer.py  # Default optimization algorithm
│   └── ROI_optimizer.py      # ROI-based optimization
├── app.py                    # Streamlit demo application
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Configuration

### Environment Variables
Set your Google API key:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### Data Files
The agent uses three main data files:
- **Model_data_for_simulator.csv**: Historical marketing spends and sales data
- **marketing_roas.csv**: ROI performance metrics for each channel
- **data_bounds.csv**: Constraints and bounds for optimization

## 📈 Optimization Algorithms

### Default Optimizer
- Uses historical data patterns
- Considers seasonal variations
- Optimizes for target sales achievement

### ROI-based Optimizer
- Incorporates ROI performance metrics
- Prioritizes high-performing channels
- Balances efficiency with target achievement

## 🎯 Report Features

The generated reports include:
- **Executive Summary**: Key insights and recommendations
- **Optimization Results Table**: Detailed spend comparisons
- **Channel Analysis**: Individual channel performance analysis
- **Risk Assessment**: Historical limit comparisons
- **Actionable Recommendations**: Next steps and monitoring suggestions

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Generating Graph Visualization
```bash
python generate_graph_image.py
```

### Command Line Usage
```bash
python Agent_AI/agent.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

---

**Built with ❤️ using LangGraph, Google Gemini Pro, and Streamlit**
