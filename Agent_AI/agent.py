import os
import sys
import pandas as pd
import json
from langgraph.graph import StateGraph, END
import google.generativeai as genai

# Add project root for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from optimizers.default_optimizer import default_optimizer
from optimizers.ROI_optimizer import roi_based_optimizer

# ---------- Utility ----------
def get_spend_limits(csv_path):
    df = pd.read_csv(csv_path)
    limits = {}
    for ch in df.columns:
        if "Spends" in ch:
            limits[ch] = {
                "max": df[ch].astype(float).max(),
                "mean": df[ch].astype(float).mean()
            }
    return limits

# ---------- LLM ----------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in environment variables")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-flash"  # Using stable version to avoid blocking issues

def invoke_llm(prompt: str, temperature: float = 0, max_tokens: int = 1000) -> str:
    """Invoke LLM synchronously and return the full response"""
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_tokens,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Check if response has valid content
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return response.text
            else:
                # If blocked, return empty string or handle gracefully
                print(f"Warning: Response blocked. Finish reason: {candidate.finish_reason}")
                return ""
        return ""
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        return ""

def stream_llm(prompt: str, temperature: float = 0, max_tokens: int = 8000):
    """Stream LLM response and yield chunks"""
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_tokens,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            stream=True
        )
        
        has_content = False
        for chunk in response:
            try:
                # Check if chunk has valid content before accessing text
                if chunk.candidates and len(chunk.candidates) > 0:
                    candidate = chunk.candidates[0]
                    if candidate.content and candidate.content.parts:
                        text = chunk.text
                        if text:
                            has_content = True
                            yield text
                    else:
                        # Chunk blocked by safety filters
                        if not has_content:
                            yield f"[Content blocked by safety filters - finish_reason: {candidate.finish_reason}]\n"
                            has_content = True
            except Exception as chunk_error:
                # Handle individual chunk errors
                if not has_content:
                    yield f"[Error processing chunk: {chunk_error}]\n"
                    has_content = True
                continue
        
        if not has_content:
            yield "[No content generated - response may have been blocked]\n"
            
    except Exception as e:
        error_msg = f"Error streaming LLM: {e}\n"
        print(error_msg)
        yield error_msg

# ---------- Graph State ----------
class AgentState(dict):
    target_sales: float
    use_roi: bool
    spends: dict
    warnings: list
    results: dict
    wants_report: bool
    report_data: dict

# ---------- Nodes ----------
def ask_target_and_type(state: AgentState):
    """
    Ask for target sales first, then ask if ROI-based optimization is required.
    Skips prompts if values already present in state (e.g., from Streamlit UI).
    """
    # If provided by caller (e.g., Streamlit), skip interactive prompts
    if state.get("target_sales") is not None and isinstance(state.get("use_roi"), bool):
        # Validate that target_sales is a valid number
        if not isinstance(state["target_sales"], (int, float)) or state["target_sales"] <= 0:
            raise ValueError("target_sales must be a positive number")
        return state

    # Step 1: Ask target sales - keep asking until we get a valid value
    target_sales = None
    while target_sales is None:
        print("Hi! What's your target sales?")
        target_input = input("> ").strip()

        # Try to parse directly first (more reliable)
        try:
            target_sales = int(target_input.replace(",", "").strip())
        except ValueError:
            # Use LLM to extract numeric target sales as fallback
            llm_prompt_sales = f"""
            Extract the target sales value from the user input as an integer without commas.
            Output only the number, or 'null' if not given.

            User: "{target_input}"
            """
            response_sales = invoke_llm(llm_prompt_sales, temperature=0, max_tokens=100)
            target_sales_str = response_sales.strip().lower()

            if target_sales_str and target_sales_str.isdigit():
                target_sales = int(target_sales_str)
            else:
                print("Please enter a valid number for target sales.")
                target_sales = None

    # Step 2: Ask for optimization type
    print("Do you want ROI-based optimization? (yes/no)")
    opt_input = input("> ").strip()

    # Try to parse directly first
    if opt_input.lower() in ["yes", "y", "true", "1"]:
        use_roi = True
    elif opt_input.lower() in ["no", "n", "false", "0"]:
        use_roi = False
    else:
        # Use LLM as fallback for unclear input
        llm_prompt_roi = f"""
        Determine if the user wants ROI-based optimization.
        Output 'true' if yes, 'false' if no, or 'null' if unclear.

        User: "{opt_input}"
        """
        response_roi = invoke_llm(llm_prompt_roi, temperature=0, max_tokens=100)
        roi_str = response_roi.strip().lower()

        if roi_str == "true":
            use_roi = True
        elif roi_str == "false":
            use_roi = False
        else:
            # Default to False if unclear
            use_roi = False
            print("Unclear input, defaulting to standard optimization.")

    # Save to state
    state["target_sales"] = target_sales
    state["use_roi"] = use_roi

    return state

def run_optimizer(state: AgentState):
    # Validate target_sales before running optimizer
    target_sales = state.get("target_sales")
    if target_sales is None or not isinstance(target_sales, (int, float)) or target_sales <= 0:
        raise ValueError(f"Invalid target_sales: {target_sales}. Must be a positive number.")
    
    roi_path = os.path.join("data", "marketing_roas.csv")
    model_path = os.path.join("data", "Model_data_for_simulator.csv")

    if state.get("use_roi", False):
        results = roi_based_optimizer(target_sales, roi_path, model_path)
    else:
        results = default_optimizer(target_sales, model_path)

    if results and not isinstance(results, dict):
        state["results"] = {"spends": results}
    elif results and "spends" not in results:
        state["results"] = {"spends": results}
    else:
        state["results"] = results

    state["spends"] = state["results"]["spends"] if state["results"] else None
    return state

def check_warnings(state: AgentState):
    model_path = os.path.join("data", "Model_data_for_simulator.csv")
    limits = get_spend_limits(model_path)
    warnings = []

    if not state.get("spends"):
        state["warnings"] = ["No spends returned by optimizer."]
        return state

    for ch, val in state["spends"].items():
        if ch in limits:
            if val > limits[ch]["max"]:
                warnings.append(f"{ch} exceeds historical max ({val:.2f} > {limits[ch]['max']:.2f})")
            elif val > limits[ch]["mean"]:
                warnings.append(f"{ch} exceeds historical mean ({val:.2f} > {limits[ch]['mean']:.2f})")

    state["warnings"] = warnings
    return state

def format_output(state: AgentState):
    if not state.get("results"):
        msg = "Hmm, I couldn't find a solution for your request."
    else:
        results = state["results"]
        target_sales = state.get("target_sales", 0)
        target_sales_str = f"{target_sales:,.0f}" if target_sales else "N/A"
        msg = f"Here's what I found for your target sales {target_sales_str}:\n"
        for ch, val in results["spends"].items():
            msg += f"  {ch}: {val:.2f}\n"
        if "seasonality" in results:
            msg += "Seasonality:\n"
            for s, v in results["seasonality"].items():
                msg += f"  {s}: {v}\n"
        if "predicted_sales" in results:
            msg += f"Predicted Sales: {results['predicted_sales']:.2f}\n"
        if state.get("warnings"):
            msg += "\n⚠️ Warnings:\n" + "\n".join(f"- {w}" for w in state["warnings"])
    print(msg)
    return state

def ask_for_report(state: AgentState):
    """
    Ask user if they want a detailed report on spends optimization.
    Skips prompt if 'wants_report' already provided in state (e.g., Streamlit UI).
    """
    # If caller already decided, do not prompt
    if isinstance(state.get("wants_report"), bool):
        return state

    print("\n" + "="*50)
    print("📊 REPORT GENERATION")
    print("="*50)
    choice = input("Would you like a detailed report on your spends optimization? (yes/no): ").strip().lower()
    state["wants_report"] = choice in ["yes", "y"]
    return state

def generate_report(state: AgentState):
    """
    Generate a comprehensive report using LLM with historical data and optimization results
    """
    if not state.get("wants_report"):
        return state

    # Get historical data for comparison
    model_path = os.path.join("data", "Model_data_for_simulator.csv")
    limits = get_spend_limits(model_path)

    # Prepare data for LLM
    report_data = {
        "target_sales": state["target_sales"],
        "optimization_type": "ROI-based" if state["use_roi"] else "Default",
        "optimized_spends": state["spends"],
        "historical_limits": limits,
        "warnings": state["warnings"],
        "results": state["results"]
    }

    # Create LLM prompt for report generation
    llm_prompt = f"""
You are a Aryma Labs optimization Agent. Analyze the following optimization results and provide insights.

TARGET SALES: ${state['target_sales']:,.0f}
OPTIMIZATION TYPE: {'ROI-based' if state['use_roi'] else 'Default'}

OPTIMIZED SPENDS:
{json.dumps(state['spends'], indent=2)}

HISTORICAL DATA (Max and Mean spends):
{json.dumps(limits, indent=2)}

WARNINGS:
{state['warnings']}

RESULTS:
{json.dumps(state['results'], indent=2)}

Generate a clear report with these sections:

## Executive Summary
Brief overview of the optimization results and key findings.

## Optimization Results
Create a table with these columns:
- Channel Name
- Optimized Spend
- Historical Max
- Historical Mean
- % of Max
- % of Mean
- Risk Level (High if <25% of max, Medium if 25-75%, Low if >75%)

## Channel Analysis
For each major channel, explain:
- How the optimized spend compares to historical data
- Whether this represents a risk or opportunity
- Specific recommendations

## Key Insights
- Which channels are getting more/less budget
- Risk areas to monitor
- Overall budget strategy

## Recommendations
- Immediate actions to take
- What to monitor closely
- How to mitigate risks

Keep it concise and actionable. Use markdown formatting.
"""

    # Stream the report generation
    print("\n" + "="*50)
    print("📋 SPENDS OPTIMIZATION REPORT")
    print("="*50)
    
    full_content = ""
    for chunk in stream_llm(llm_prompt, temperature=0.7):
        print(chunk, end="", flush=True)
        full_content += chunk
    
    print("\n" + "="*50)

    # Store report data in state
    state["report_data"] = {
        "content": full_content,
        "data": report_data
    }
    return state

# ---------- LangGraph ----------
graph = StateGraph(AgentState)
graph.add_node("ask_target_and_type", ask_target_and_type)
graph.add_node("run_optimizer", run_optimizer)
graph.add_node("check_warnings", check_warnings)
graph.add_node("format_output", format_output)
graph.add_node("ask_for_report", ask_for_report)
graph.add_node("generate_report", generate_report)

graph.set_entry_point("ask_target_and_type")
graph.add_edge("ask_target_and_type", "run_optimizer")
graph.add_edge("run_optimizer", "check_warnings")
graph.add_edge("check_warnings", "format_output")
graph.add_edge("format_output", "ask_for_report")
graph.add_edge("ask_for_report", "generate_report")
graph.add_edge("generate_report", END)

app = graph.compile()

if __name__ == "__main__":
    app.invoke({})


