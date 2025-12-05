import os
import json
from pathlib import Path
from typing import Literal, Optional
import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from optimizers.predictor import predict_sales_for_spends
from optimizers.default_optimizer import DefaultOptimizer, geometric_hill, CHANNELS, BETA_COLS
from optimizers.ROI_optimizer import ROIOptimizer
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_CSV = DATA_DIR / "Model_data_for_simulator.csv"
ROI_CSV = DATA_DIR / "marketing_roas.csv"
R_CODE_DIR = BASE_DIR / "R code"
SPENDS_DATA_CSV = R_CODE_DIR / "Spends Data.csv"
HYPERPARAMS_CSV = R_CODE_DIR / "Hyperparams.csv"
COEFS_CSV = R_CODE_DIR / "Coefs.csv"

# Global optimizer instances
default_optimizer_instance = None
roi_optimizer_instance = None
optimizers_initialized = False


def get_spend_limits(csv_path: str) -> dict:
    """Get historical max and mean for each spend channel."""
    df = pd.read_csv(csv_path)
    limits = {}
    for ch in df.columns:
        if "Spends" in ch:
            limits[ch] = {
                "max": float(df[ch].astype(float).max()),
                "mean": float(df[ch].astype(float).mean())
            }
    return limits


def check_warnings(spends: dict, csv_path: str) -> list:
    """Check if any spend exceeds historical limits."""
    limits = get_spend_limits(csv_path)
    warnings = []
    
    if not spends:
        return ["No spends returned by optimizer."]
    
    for ch, val in spends.items():
        if ch in limits:
            if val > limits[ch]["max"]:
                warnings.append({
                    "type": "error",
                    "channel": ch,
                    "message": f"{ch} exceeds historical maximum",
                    "detail": f"Optimized: ${val:,.0f} vs Max: ${limits[ch]['max']:,.0f}"
                })
            elif val > limits[ch]["mean"]:
                warnings.append({
                    "type": "warning",
                    "channel": ch,
                    "message": f"{ch} exceeds historical average",
                    "detail": f"Optimized: ${val:,.0f} vs Avg: ${limits[ch]['mean']:,.0f}"
                })
    
    return warnings


class OptimizeRequest(BaseModel):
    target_sales: float = Field(..., gt=0, description="Desired sales figure in dollars")
    optimizer_type: Literal["default", "roi"] = Field(
        "default", description="Choose default optimizer or ROI-based optimizer"
    )
    prediction_month: str = Field(
        "Jan-24", description="Month identifier formatted as Mon-YY (e.g., Jan-24)"
    )


class PredictRequest(BaseModel):
    spends: dict[str, float] = Field(..., description="Dictionary of channel spends")
    prediction_month: str = Field(
        "Jan-24", description="Month identifier formatted as Mon-YY (e.g., Jan-24)"
    )


class ReportRequest(BaseModel):
    target_sales: float = Field(..., gt=0)
    optimizer_type: str = Field("default")
    spends: dict[str, float] = Field(...)
    predicted_sales: Optional[float] = None
    seasonality: Optional[dict] = None
    warnings: Optional[list] = None


class OptimizedSpendsRequest(BaseModel):
    optimized_spends: Optional[dict[str, float]] = Field(default=None, description="Dictionary of optimized spends by channel")


async def initialize_optimizers():
    """Initialize optimizers in the background."""
    global default_optimizer_instance, roi_optimizer_instance, optimizers_initialized
    
    try:
        print("Initializing optimizers in background...")
        
        # Initialize default optimizer
        print("Creating default optimizer...")
        default_optimizer_instance = DefaultOptimizer(
            csv_path=str(MODEL_CSV),
            algorithm='basinhopping',
            niter_bh=50
        )
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, default_optimizer_instance.initialize_sales_range)
        
        # Initialize ROI optimizer
        print("Creating ROI optimizer...")
        roi_optimizer_instance = ROIOptimizer(
            csv_path=str(MODEL_CSV),
            roi_csv_path=str(ROI_CSV),
            algorithm='basinhopping',
            niter_bh=50
        )
        
        await loop.run_in_executor(None, roi_optimizer_instance.initialize_sales_range)
        
        optimizers_initialized = True
        print("Optimizers initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing optimizers: {e}")
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Initialize optimizers in background
    asyncio.create_task(initialize_optimizers())
    yield
    # Shutdown: cleanup if needed
    print("Shutting down...")


app = FastAPI(title="ArymaEdge Optimizer API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "optimizers_initialized": optimizers_initialized
    }


@app.get("/sales-range")
async def get_sales_range():
    """Get the achievable sales range from the optimizers."""
    if not optimizers_initialized:
        return {
            "status": "initializing",
            "message": "Optimizers are still initializing. Please wait..."
        }
    
    return {
        "status": "ready",
        "default_optimizer": {
            "min_sales": default_optimizer_instance.min_achievable_sales,
            "max_sales": default_optimizer_instance.max_achievable_sales
        },
        "roi_optimizer": {
            "min_sales": roi_optimizer_instance.min_achievable_sales,
            "max_sales": roi_optimizer_instance.max_achievable_sales
        }
    }


@app.post("/optimize")
async def optimize_spends(request: OptimizeRequest):
    global default_optimizer_instance, roi_optimizer_instance, optimizers_initialized
    
    # Check if optimizers are initialized
    if not optimizers_initialized:
        # Use fallback: create temporary optimizer
        if request.optimizer_type == "roi":
            optimizer = ROIOptimizer(
                csv_path=str(MODEL_CSV),
                roi_csv_path=str(ROI_CSV),
                algorithm='basinhopping',
                niter_bh=50
            )
            print("Warning: Using fallback ROI optimizer (not initialized yet)")
            optimizer.initialize_sales_range()
            result = optimizer.optimize(request.target_sales)
        else:
            optimizer = DefaultOptimizer(
                csv_path=str(MODEL_CSV),
                algorithm='basinhopping',
                niter_bh=50
            )
            print("Warning: Using fallback default optimizer (not initialized yet)")
            optimizer.initialize_sales_range()
            result = optimizer.optimize(request.target_sales)
    else:
        # Use pre-initialized optimizers
        if request.optimizer_type == "roi":
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                roi_optimizer_instance.optimize, 
                request.target_sales
            )
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                default_optimizer_instance.optimize,
                request.target_sales
            )

    if not result:
        raise HTTPException(status_code=400, detail="Optimization failed to converge.")

    # Check if target was achievable
    if not result.get('target_achievable', True):
        # Target was not achievable, but we returned best possible
        result["warnings"] = [{
            "type": "error",
            "channel": "Overall",
            "message": result.get('message', 'Target sales not achievable'),
            "detail": f"Achievable range: [{result['min_achievable_sales']:,.0f}, {result['max_achievable_sales']:,.0f}]"
        }]
    else:
        # Add spend warnings
        warnings = check_warnings(result.get("spends", {}), str(MODEL_CSV))
        result["warnings"] = warnings

    return jsonable_encoder(result)


@app.post("/predict")
async def predict_sales(request: PredictRequest):
    """Predict sales for given spend values without optimization."""
    try:
        predicted_sales = predict_sales_for_spends(
            spends_dict=request.spends,
            prediction_month=request.prediction_month,
            csv_path=str(MODEL_CSV),
        )
        return {"predicted_sales": predicted_sales}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    """Generate a detailed optimization report."""
    try:
        limits = get_spend_limits(str(MODEL_CSV))
        
        # Build report content
        report_sections = []
        
        # Executive Summary
        total_spend = sum(request.spends.values())
        report_sections.append({
            "title": "Executive Summary",
            "content": f"Optimization completed for target sales of ${request.target_sales:,.0f} using {request.optimizer_type.upper()} optimization. "
                      f"Total recommended spend: ${total_spend:,.0f}. "
                      f"Predicted sales: ${request.predicted_sales:,.0f}." if request.predicted_sales else ""
        })
        
        # Channel Analysis Table
        channel_analysis = []
        for ch, val in request.spends.items():
            if ch in limits:
                pct_of_max = (val / limits[ch]["max"] * 100) if limits[ch]["max"] > 0 else 0
                pct_of_mean = (val / limits[ch]["mean"] * 100) if limits[ch]["mean"] > 0 else 0
                
                if pct_of_max > 100:
                    risk = "High"
                elif pct_of_max > 75:
                    risk = "Medium"
                else:
                    risk = "Low"
                
                channel_analysis.append({
                    "channel": ch,
                    "optimized_spend": val,
                    "historical_max": limits[ch]["max"],
                    "historical_mean": limits[ch]["mean"],
                    "pct_of_max": pct_of_max,
                    "pct_of_mean": pct_of_mean,
                    "risk_level": risk
                })
        
        report_sections.append({
            "title": "Channel Analysis",
            "type": "table",
            "data": channel_analysis
        })
        
        # Key Insights
        high_risk_channels = [c for c in channel_analysis if c["risk_level"] == "High"]
        above_avg_channels = [c for c in channel_analysis if c["pct_of_mean"] > 100]
        below_avg_channels = [c for c in channel_analysis if c["pct_of_mean"] < 50]
        
        insights = []
        if high_risk_channels:
            insights.append(f"⚠️ {len(high_risk_channels)} channel(s) exceed historical maximum spending levels")
        if above_avg_channels:
            insights.append(f"📈 {len(above_avg_channels)} channel(s) are above historical average")
        if below_avg_channels:
            insights.append(f"📉 {len(below_avg_channels)} channel(s) are significantly below average")
        
        report_sections.append({
            "title": "Key Insights",
            "type": "list",
            "items": insights
        })
        
        # Recommendations
        recommendations = []
        for ch in high_risk_channels:
            recommendations.append(f"Monitor {ch['channel']} closely - spending exceeds historical limits")
        if request.warnings:
            recommendations.append("Review all warnings before implementing this budget allocation")
        recommendations.append("Consider A/B testing for channels with significant budget changes")
        recommendations.append("Track performance weekly to validate optimization assumptions")
        
        report_sections.append({
            "title": "Recommendations",
            "type": "list",
            "items": recommendations
        })
        
        return {
            "report": report_sections,
            "generated_at": pd.Timestamp.now().isoformat(),
            "meta": {
                "target_sales": request.target_sales,
                "optimizer_type": request.optimizer_type,
                "total_spend": total_spend,
                "predicted_sales": request.predicted_sales
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/saturation-curves")
async def get_saturation_curves():
    """Generate saturation curve data using R code files - EXACT implementation from saturation_curve.py."""
    try:
        # Load the R code files (EXACT from saturation_curve.py)
        data = pd.read_csv(str(SPENDS_DATA_CSV))
        hyperparams = pd.read_csv(str(HYPERPARAMS_CSV))
        coefs = pd.read_csv(str(COEFS_CSV))
        
        # Arranging the data alphabetically (EXACT from saturation_curve.py)
        first_column = data.iloc[:, 0]
        sorted_columns = sorted(data.columns[1:])
        data_sorted = pd.concat([first_column, data[sorted_columns]], axis=1)
        data = data_sorted
        
        hyperparams_sorted_columns = sorted(hyperparams.columns[:])
        hyperparams_sorted = hyperparams[hyperparams_sorted_columns]
        hyperparams = hyperparams_sorted
        
        coefs_sorted = coefs.sort_values(by="rn")
        coefs = coefs_sorted
        
        # Extracting the hyperparameters (EXACT from saturation_curve.py)
        thetas = hyperparams.filter(like="thetas")
        alphas = hyperparams.filter(like="alphas")
        gammas = hyperparams.filter(like="gammas")
        
        # Get the media channels (EXACT from saturation_curve.py)
        ds = data.columns[0]
        data = data.drop(columns=[ds])
        paid_media_spends = data.columns
        
        # Geometric Adstock Function (EXACT from saturation_curve.py)
        def adstock_geometric(x, theta):
            if isinstance(x, pd.Series):
                x = x.values
            if not np.isscalar(theta):
                raise ValueError("theta must be a single value")
                
            if len(x) > 1:
                x_decayed = np.zeros_like(x)
                x_decayed[0] = x[0]
                for i in range(1, len(x_decayed)):
                    x_decayed[i] = x[i] + theta * x_decayed[i - 1]
                
                theta_vec_cum = np.zeros_like(x)
                theta_vec_cum[0] = theta
                for t in range(1, len(x)):
                    theta_vec_cum[t] = theta_vec_cum[t - 1] * theta
            else:
                x_decayed = x
                theta_vec_cum = np.array([theta])

            inflation_total = np.sum(x_decayed) / np.sum(x)
            return {'x': x, 'x_decayed': x_decayed, 'thetaVecCum': theta_vec_cum, 'inflation_total': inflation_total}
        
        # Compute the inflexions (EXACT from saturation_curve.py)
        def transform(i):
            x = data[paid_media_spends[i]].values
            theta = thetas.iloc[0, i]
            transform = adstock_geometric(x, theta)
            return transform["x_decayed"]
        
        x_trans = {col: transform(i) for i, col in enumerate(paid_media_spends)}
        
        inflexions = {col: np.dot([min(x_trans[col]), max(x_trans[col])], [1 - gammas.iloc[0, i], gammas.iloc[0, i]]) for i, col in enumerate(paid_media_spends)}
        
        # Response function (EXACT from saturation_curve.py)
        def fx_objective(x, coeff, alpha, inflexion, x_hist_carryover, get_sum=True):
            # Ensure x and x_hist_carryover are NumPy arrays for element-wise operations
            x = np.array(x)
            x_hist_carryover = np.array(x_hist_carryover)
            
            # Adstock scales: Adding the mean of x_hist_carryover to the x values
            x_adstocked = x + np.mean(x_hist_carryover)
            inflexion = np.array(inflexion)
            alpha = np.array(alpha)
            # Hill transformation calculation
            if get_sum:
                # Sum of the transformed values
                x_out = coeff * np.sum((1 + (inflexion ** alpha) / (x_adstocked ** alpha)) ** -1)
            else:
                # Individual value calculation
                x_out = coeff * ((1 + (inflexion ** alpha) / (x_adstocked ** alpha)) ** -1)
            
            return x_out
        
        # Simulate for max, avg, and response (EXACT from saturation_curve.py)
        def create_curve_data(i):
            max_spends = data[paid_media_spends[i]].max()
            avg_spends = data[paid_media_spends[i]].mean()
            
            # Calculate saturation point (90% of maximum response) - do this FIRST
            coef = coefs.iloc[i]['coef']
            alpha_val = alphas.iloc[0, i]
            inflexion_val = inflexions[paid_media_spends[i]]
            
            saturation_spend = None
            saturation_response = None
            try:
                if inflexion_val > 0 and alpha_val > 0:
                    # Calculate saturation at 90% of max response
                    saturation_spend = float((inflexion_val**alpha_val / 0.1111)**(1/alpha_val))
                    saturation_response = float(fx_objective([saturation_spend], coef, alpha_val, inflexion_val, 0, get_sum=False)[0])
            except:
                pass
            
            # Calculate get_max_x as 1.2 times the max of (saturation_spend, max_spends)
            sat_spend_for_calc = saturation_spend if saturation_spend else max_spends
            get_max_x = max(max_spends, sat_spend_for_calc) * 1.2
            
            # Generate spend range from 0 to get_max_x
            simulate_spend = np.linspace(0, get_max_x, 100)

            simulate_response = fx_objective(simulate_spend, coefs.iloc[i]['coef'], alphas.iloc[0, i], inflexions[paid_media_spends[i]], 0, get_sum=False)
            simulate_response_max = fx_objective([max_spends], coefs.iloc[i]['coef'], alphas.iloc[0, i], inflexions[paid_media_spends[i]], 0, get_sum=False)
            simulate_response_avg = fx_objective([avg_spends], coefs.iloc[i]['coef'], alphas.iloc[0, i], inflexions[paid_media_spends[i]], 0, get_sum=False)
            
            return {
                "paid_media_spends": paid_media_spends[i],
                "simulate_spend": simulate_spend,
                "simulate_response": simulate_response,
                "max_spend": max_spends,
                "max_response": simulate_response_max,
                "avg_spend": avg_spends,
                "avg_response": simulate_response_avg,
                "saturation_spend": saturation_spend,
                "saturation_response": saturation_response
            }
        
        # Generate all curve data
        all_curves = [create_curve_data(i) for i in range(len(paid_media_spends))]
        
        # Create DataFrames for plotting (EXACT from saturation_curve.py)
        plotDF_scurve_list = []
        max_spends_dt_list = []
        avg_spends_dt_list = []
        sat_spends_dt_list = []
        
        for curve in all_curves:
            plotDF_scurve_list.append(pd.DataFrame({
                "paid_media_spends": [curve["paid_media_spends"]] * len(curve["simulate_spend"]),
                "spend": curve["simulate_spend"],
                "total_response": curve["simulate_response"]
            }))
            
            max_spends_dt_list.append(pd.DataFrame({
                "paid_media_spends": [curve["paid_media_spends"]],
                "max_spend": [curve["max_spend"]],
                "response": [curve["max_response"][0] if isinstance(curve["max_response"], np.ndarray) else curve["max_response"]]
            }))
            
            avg_spends_dt_list.append(pd.DataFrame({
                "paid_media_spends": [curve["paid_media_spends"]],
                "avg_spend": [curve["avg_spend"]],
                "response": [curve["avg_response"][0] if isinstance(curve["avg_response"], np.ndarray) else curve["avg_response"]]
            }))
            
            if curve["saturation_spend"]:
                sat_spends_dt_list.append(pd.DataFrame({
                    "paid_media_spends": [curve["paid_media_spends"]],
                    "sat_spend": [curve["saturation_spend"]],
                    "response": [curve["saturation_response"]]
                }))
        
        plotDF_scurve = pd.concat(plotDF_scurve_list, ignore_index=True)
        max_spends_dt = pd.concat(max_spends_dt_list, ignore_index=True)
        avg_spends_dt = pd.concat(avg_spends_dt_list, ignore_index=True)
        sat_spends_dt = pd.concat(sat_spends_dt_list, ignore_index=True) if sat_spends_dt_list else None
        
        # Define the qualitative color palette (EXACT from saturation_curve.py)
        color_palette = px.colors.qualitative.Set1
        
        # Create Plotly chart (EXACT from saturation_curve.py)
        fig = go.Figure()
        
        # Loop over media and assign a unique color from the palette
        for i, media in enumerate(paid_media_spends):
            # Replace underscores with spaces in the media name for the legend
            media_name = media.replace('_', ' ')
        
            # Get the color from the palette based on the index
            color = color_palette[i % len(color_palette)]
            
            # Create a legend group for each media
            legend_group = f"media_{i}"
        
            # Add line trace with the color
            fig.add_trace(go.Scatter(
                x=plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['spend'],
                y=plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['total_response'],
                mode='lines',
                name=media_name,
                line=dict(color=color),
                hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Spend:</b> <b>{int(spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for spend, response in zip(
                plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['spend'],
                plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['total_response'])],
                hoverinfo='text',
                legendgroup=legend_group))
        
            # Add max spend scatter points (with channel info in customdata)
            max_spend_data = max_spends_dt[max_spends_dt['paid_media_spends'] == media]
            fig.add_trace(go.Scatter(
                x=max_spend_data['max_spend'],
                y=max_spend_data['response'],
                mode='markers',
                showlegend=False,
                marker=dict(symbol="circle", size=8, color=color),
                hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Maximum Spend:</b> <b>{int(max_spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for max_spend, response in zip(
                max_spend_data['max_spend'], max_spend_data['response'])],
                hoverinfo='text',
                legendgroup="markers_max",
                customdata=[media_name] * len(max_spend_data)))
        
            # Add avg spend scatter points (with channel info in customdata)
            avg_spend_data = avg_spends_dt[avg_spends_dt['paid_media_spends'] == media]
            fig.add_trace(go.Scatter(
                x=avg_spend_data['avg_spend'],
                y=avg_spend_data['response'],
                mode='markers',
                showlegend=False,
                marker=dict(symbol="triangle-up", size=8, color=color),
                hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Average Spend:</b> <b>{int(avg_spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for avg_spend, response in zip(
                avg_spend_data['avg_spend'], avg_spend_data['response'])],
                hoverinfo='text',
                legendgroup="markers_avg",
                customdata=[media_name] * len(avg_spend_data)))
            
            # Add saturation spend scatter points (with channel info in customdata)
            if sat_spends_dt is not None:
                sat_spend_data = sat_spends_dt[sat_spends_dt['paid_media_spends'] == media]
                if not sat_spend_data.empty:
                    fig.add_trace(go.Scatter(
                        x=sat_spend_data['sat_spend'],
                        y=sat_spend_data['response'],
                        mode='markers',
                        showlegend=False,
                        marker=dict(symbol="square", size=8, color=color),
                        hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Saturation Point:</b> <b>{int(sat_spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for sat_spend, response in zip(
                        sat_spend_data['sat_spend'], sat_spend_data['response'])],
                        hoverinfo='text',
                        legendgroup="markers_sat",
                        customdata=[media_name] * len(sat_spend_data)))
        
        # Add clickable legend entries for marker types
        fig.add_trace(go.Scatter(x=[None], y=[None],
                                 mode='markers',
                                 name="○ Max",
                                 showlegend=True,
                                 marker=dict(symbol="circle", size=8, color="gray"),
                                 legendgroup="markers_max",
                                 hoverinfo='skip'))
        
        fig.add_trace(go.Scatter(x=[None], y=[None],
                                 mode='markers',
                                 name="△ Avg",
                                 showlegend=True,
                                 marker=dict(symbol="triangle-up", size=8, color="gray"),
                                 legendgroup="markers_avg",
                                 hoverinfo='skip'))
        
        fig.add_trace(go.Scatter(x=[None], y=[None],
                                 mode='markers',
                                 name="◾ Sat",
                                 showlegend=True,
                                 marker=dict(symbol="square", size=8, color="gray"),
                                 legendgroup="markers_sat",
                                 hoverinfo='skip'))
        
        # Update layout - move legend to bottom
        fig.update_layout(
            title="<b>Saturation Curves</b>",
            title_x=0.5,
            title_xanchor='center',
            xaxis_title="<b>Digital Spends (in USD)</b>",
            yaxis_title="<b>Sales Volume</b>",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="lightgray",
                borderwidth=1
            ),
            margin=dict(b=100)
        )
        
        # Return Plotly figure as JSON
        return {"plotly_json": fig.to_json()}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate saturation curves: {str(e)}")


@app.post("/saturation-curves-with-optimized")
async def get_saturation_curves_with_optimized(request: OptimizedSpendsRequest):
    """Generate saturation curve data with optimized markers - EXACT implementation from saturation_curve.py."""
    try:
        # Load the R code files (EXACT from saturation_curve.py)
        data = pd.read_csv(str(SPENDS_DATA_CSV))
        hyperparams = pd.read_csv(str(HYPERPARAMS_CSV))
        coefs = pd.read_csv(str(COEFS_CSV))
        
        # Arranging the data alphabetically (EXACT from saturation_curve.py)
        first_column = data.iloc[:, 0]
        sorted_columns = sorted(data.columns[1:])
        data_sorted = pd.concat([first_column, data[sorted_columns]], axis=1)
        data = data_sorted
        
        hyperparams_sorted_columns = sorted(hyperparams.columns[:])
        hyperparams_sorted = hyperparams[hyperparams_sorted_columns]
        hyperparams = hyperparams_sorted
        
        coefs_sorted = coefs.sort_values(by="rn")
        coefs = coefs_sorted
        
        # Extracting the hyperparameters (EXACT from saturation_curve.py)
        thetas = hyperparams.filter(like="thetas")
        alphas = hyperparams.filter(like="alphas")
        gammas = hyperparams.filter(like="gammas")
        
        # Get the media channels (EXACT from saturation_curve.py)
        ds = data.columns[0]
        data = data.drop(columns=[ds])
        paid_media_spends = data.columns
        
        # Geometric Adstock Function (EXACT from saturation_curve.py)
        def adstock_geometric(x, theta):
            if isinstance(x, pd.Series):
                x = x.values
            if not np.isscalar(theta):
                raise ValueError("theta must be a single value")
                
            if len(x) > 1:
                x_decayed = np.zeros_like(x)
                x_decayed[0] = x[0]
                for i in range(1, len(x_decayed)):
                    x_decayed[i] = x[i] + theta * x_decayed[i - 1]
                
                theta_vec_cum = np.zeros_like(x)
                theta_vec_cum[0] = theta
                for t in range(1, len(x)):
                    theta_vec_cum[t] = theta_vec_cum[t - 1] * theta
            else:
                x_decayed = x
                theta_vec_cum = np.array([theta])

            inflation_total = np.sum(x_decayed) / np.sum(x)
            return {'x': x, 'x_decayed': x_decayed, 'thetaVecCum': theta_vec_cum, 'inflation_total': inflation_total}
        
        # Compute the inflexions (EXACT from saturation_curve.py)
        def transform(i):
            x = data[paid_media_spends[i]].values
            theta = thetas.iloc[0, i]
            transform = adstock_geometric(x, theta)
            return transform["x_decayed"]
        
        x_trans = {col: transform(i) for i, col in enumerate(paid_media_spends)}
        
        inflexions = {col: np.dot([min(x_trans[col]), max(x_trans[col])], [1 - gammas.iloc[0, i], gammas.iloc[0, i]]) for i, col in enumerate(paid_media_spends)}
        
        # Response function (EXACT from saturation_curve.py)
        def fx_objective(x, coeff, alpha, inflexion, x_hist_carryover, get_sum=True):
            # Ensure x and x_hist_carryover are NumPy arrays for element-wise operations
            x = np.array(x)
            x_hist_carryover = np.array(x_hist_carryover)
            
            # Adstock scales: Adding the mean of x_hist_carryover to the x values
            x_adstocked = x + np.mean(x_hist_carryover)
            inflexion = np.array(inflexion)
            alpha = np.array(alpha)
            # Hill transformation calculation
            if get_sum:
                # Sum of the transformed values
                x_out = coeff * np.sum((1 + (inflexion ** alpha) / (x_adstocked ** alpha)) ** -1)
            else:
                # Individual value calculation
                x_out = coeff * ((1 + (inflexion ** alpha) / (x_adstocked ** alpha)) ** -1)
            
            return x_out
        
        optimized_spends = request.optimized_spends or {}
        
        # Calculate global max for chart scaling
        global_max_spend = 0
        global_max_response = 0
        
        # Simulate for max, avg, optimized, and response (based on saturation_curve.py)
        def create_curve_data(i):
            max_spends = data[paid_media_spends[i]].max()
            avg_spends = data[paid_media_spends[i]].mean()
            
            # Calculate saturation point (90% of maximum response) - do this FIRST
            coef = coefs.iloc[i]['coef']
            alpha_val = alphas.iloc[0, i]
            inflexion_val = inflexions[paid_media_spends[i]]
            
            saturation_spend = None
            saturation_response = None
            try:
                if inflexion_val > 0 and alpha_val > 0:
                    saturation_spend = float((inflexion_val**alpha_val / 0.1111)**(1/alpha_val))
                    saturation_response = float(fx_objective([saturation_spend], coef, alpha_val, inflexion_val, 0, get_sum=False)[0])
            except:
                pass
            
            # Calculate get_max_x as 1.2 times the max of (saturation_spend, max_spends)
            sat_spend_for_calc = saturation_spend if saturation_spend else max_spends
            get_max_x = max(max_spends, sat_spend_for_calc) * 1.2
            
            # Generate spend range from 0 to get_max_x
            simulate_spend = np.linspace(0, get_max_x, 100)
            
            simulate_response = fx_objective(simulate_spend, coefs.iloc[i]['coef'], alphas.iloc[0, i], inflexions[paid_media_spends[i]], 0, get_sum=False)
            simulate_response_max = fx_objective([max_spends], coefs.iloc[i]['coef'], alphas.iloc[0, i], inflexions[paid_media_spends[i]], 0, get_sum=False)
            simulate_response_avg = fx_objective([avg_spends], coefs.iloc[i]['coef'], alphas.iloc[0, i], inflexions[paid_media_spends[i]], 0, get_sum=False)
            
            # Calculate optimized spend response (if provided)
            opt_spend = optimized_spends.get(paid_media_spends[i], 0)
            opt_response = None
            if opt_spend > 0:
                opt_response = float(fx_objective([opt_spend], coef, alpha_val, inflexion_val, 0, get_sum=False)[0])
            
            return {
                "paid_media_spends": paid_media_spends[i],
                "simulate_spend": simulate_spend,
                "simulate_response": simulate_response,
                "max_spend": max_spends,
                "max_response": simulate_response_max,
                "avg_spend": avg_spends,
                "avg_response": simulate_response_avg,
                "saturation_spend": saturation_spend,
                "saturation_response": saturation_response,
                "opt_spend": opt_spend,
                "opt_response": opt_response
            }
        
        # Generate all curve data
        all_curves = [create_curve_data(i) for i in range(len(paid_media_spends))]
        
        # Create DataFrames for plotting
        plotDF_scurve_list = []
        max_spends_dt_list = []
        avg_spends_dt_list = []
        sat_spends_dt_list = []
        opt_spends_dt_list = []
        
        for curve in all_curves:
            plotDF_scurve_list.append(pd.DataFrame({
                "paid_media_spends": [curve["paid_media_spends"]] * len(curve["simulate_spend"]),
                "spend": curve["simulate_spend"],
                "total_response": curve["simulate_response"]
            }))
            
            max_spends_dt_list.append(pd.DataFrame({
                "paid_media_spends": [curve["paid_media_spends"]],
                "max_spend": [curve["max_spend"]],
                "response": [curve["max_response"][0] if isinstance(curve["max_response"], np.ndarray) else curve["max_response"]]
            }))
            
            avg_spends_dt_list.append(pd.DataFrame({
                "paid_media_spends": [curve["paid_media_spends"]],
                "avg_spend": [curve["avg_spend"]],
                "response": [curve["avg_response"][0] if isinstance(curve["avg_response"], np.ndarray) else curve["avg_response"]]
            }))
            
            if curve["saturation_spend"]:
                sat_spends_dt_list.append(pd.DataFrame({
                    "paid_media_spends": [curve["paid_media_spends"]],
                    "sat_spend": [curve["saturation_spend"]],
                    "response": [curve["saturation_response"]]
                }))
            
            if curve["opt_spend"] > 0:
                opt_spends_dt_list.append(pd.DataFrame({
                    "paid_media_spends": [curve["paid_media_spends"]],
                    "opt_spend": [curve["opt_spend"]],
                    "response": [curve["opt_response"]]
                }))
        
        plotDF_scurve = pd.concat(plotDF_scurve_list, ignore_index=True)
        max_spends_dt = pd.concat(max_spends_dt_list, ignore_index=True)
        avg_spends_dt = pd.concat(avg_spends_dt_list, ignore_index=True)
        sat_spends_dt = pd.concat(sat_spends_dt_list, ignore_index=True) if sat_spends_dt_list else None
        opt_spends_dt = pd.concat(opt_spends_dt_list, ignore_index=True) if opt_spends_dt_list else None
        
        # Define the qualitative color palette
        color_palette = px.colors.qualitative.Set1
        
        # Create Plotly chart
        fig = go.Figure()
        
        # Loop over media and assign a unique color from the palette
        for i, media in enumerate(paid_media_spends):
            media_name = media.replace('_', ' ')
            color = color_palette[i % len(color_palette)]
            legend_group = f"media_{i}"
        
            # Add line trace
            fig.add_trace(go.Scatter(
                x=plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['spend'],
                y=plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['total_response'],
                mode='lines',
                name=media_name,
                line=dict(color=color),
                hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Spend:</b> <b>{int(spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for spend, response in zip(
                plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['spend'],
                plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['total_response'])],
                hoverinfo='text',
                legendgroup=legend_group))
        
            # Add max spend scatter points (with channel info in customdata)
            max_spend_data = max_spends_dt[max_spends_dt['paid_media_spends'] == media]
            fig.add_trace(go.Scatter(
                x=max_spend_data['max_spend'],
                y=max_spend_data['response'],
                mode='markers',
                showlegend=False,
                marker=dict(symbol="circle", size=8, color=color),
                hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Maximum Spend:</b> <b>{int(max_spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for max_spend, response in zip(
                max_spend_data['max_spend'], max_spend_data['response'])],
                hoverinfo='text',
                legendgroup="markers_max",
                customdata=[media_name] * len(max_spend_data)))
        
            # Add avg spend scatter points (with channel info in customdata)
            avg_spend_data = avg_spends_dt[avg_spends_dt['paid_media_spends'] == media]
            fig.add_trace(go.Scatter(
                x=avg_spend_data['avg_spend'],
                y=avg_spend_data['response'],
                mode='markers',
                showlegend=False,
                marker=dict(symbol="triangle-up", size=8, color=color),
                hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Average Spend:</b> <b>{int(avg_spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for avg_spend, response in zip(
                avg_spend_data['avg_spend'], avg_spend_data['response'])],
                hoverinfo='text',
                legendgroup="markers_avg",
                customdata=[media_name] * len(avg_spend_data)))
            
            # Add saturation spend scatter points (with channel info in customdata)
            if sat_spends_dt is not None:
                sat_spend_data = sat_spends_dt[sat_spends_dt['paid_media_spends'] == media]
                if not sat_spend_data.empty:
                    fig.add_trace(go.Scatter(
                        x=sat_spend_data['sat_spend'],
                        y=sat_spend_data['response'],
                        mode='markers',
                        showlegend=False,
                        marker=dict(symbol="square", size=8, color=color),
                        hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Saturation Point:</b> <b>{int(sat_spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for sat_spend, response in zip(
                        sat_spend_data['sat_spend'], sat_spend_data['response'])],
                        hoverinfo='text',
                        legendgroup="markers_sat",
                        customdata=[media_name] * len(sat_spend_data)))
            
            # Add optimized spend scatter points (with channel info in customdata)
            if opt_spends_dt is not None:
                opt_spend_data = opt_spends_dt[opt_spends_dt['paid_media_spends'] == media]
                if not opt_spend_data.empty:
                    fig.add_trace(go.Scatter(
                        x=opt_spend_data['opt_spend'],
                        y=opt_spend_data['response'],
                        mode='markers',
                        showlegend=False,
                        marker=dict(symbol="star", size=10, color="gold", line=dict(color=color, width=2)),
                        hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Optimized Spend:</b> <b>{int(opt_spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for opt_spend, response in zip(
                        opt_spend_data['opt_spend'], opt_spend_data['response'])],
                        hoverinfo='text',
                        legendgroup="markers_opt",
                        customdata=[media_name] * len(opt_spend_data)))
        
        # Add clickable legend entries for marker types
        fig.add_trace(go.Scatter(x=[None], y=[None],
                                 mode='markers',
                                 name="○ Max",
                                 showlegend=True,
                                 marker=dict(symbol="circle", size=8, color="gray"),
                                 legendgroup="markers_max",
                                 hoverinfo='skip'))
        
        fig.add_trace(go.Scatter(x=[None], y=[None],
                                 mode='markers',
                                 name="△ Avg",
                                 showlegend=True,
                                 marker=dict(symbol="triangle-up", size=8, color="gray"),
                                 legendgroup="markers_avg",
                                 hoverinfo='skip'))
        
        fig.add_trace(go.Scatter(x=[None], y=[None],
                                 mode='markers',
                                 name="◾ Sat",
                                 showlegend=True,
                                 marker=dict(symbol="square", size=8, color="gray"),
                                 legendgroup="markers_sat",
                                 hoverinfo='skip'))
        
        if opt_spends_dt is not None and not opt_spends_dt.empty:
            fig.add_trace(go.Scatter(x=[None], y=[None],
                                     mode='markers',
                                     name="⭐ Opt",
                                     showlegend=True,
                                     marker=dict(symbol="star", size=10, color="gold"),
                                     legendgroup="markers_opt",
                                     hoverinfo='skip'))
        
        # Update layout - move legend to bottom
        fig.update_layout(
            title="<b>Saturation Curves</b>",
            title_x=0.5,
            title_xanchor='center',
            xaxis_title="<b>Digital Spends (in USD)</b>",
            yaxis_title="<b>Sales Volume</b>",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="lightgray",
                borderwidth=1
            ),
            margin=dict(b=100)
        )
        
        # Return Plotly figure as JSON
        return {"plotly_json": fig.to_json()}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate saturation curves: {str(e)}")
