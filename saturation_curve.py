# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:14:06 2025

@author: Aryma6
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:29:44 2025

@author: Aryma6
"""

import pandas as pd
import numpy as np
from shiny import App, render, ui, reactive
import plotly.graph_objects as go
import nest_asyncio
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
from shinywidgets import output_widget, render_widget  

# Define the layout of the application
app_ui = ui.page_fluid(
    # Add CSS styling for the progress bar
    ui.tags.style("""
        /* Ensure the progress bar container is tall enough */
        .progress-bar-container {
            height: 40px !important;  /* Adjust the height of the entire progress bar */
            line-height: 40px;  /* Align the text vertically */
        }
        
        /* Ensure the progress bar itself is tall enough */
        .progress-bar {
            height: 100% !important;  /* Full height of the container */
        }
        
        /* Center and enlarge the text */
        .progress-bar-text {
            font-size: 18px !important;  /* Adjust font size for readability */
            color: #000 !important;  /* Set text color for visibility */
            text-align: center !important;  /* Ensure the text is centered */
            line-height: 40px;  /* Ensure vertical centering of the text */
        }
    """),

    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file("data_saturation", "Choose Data File", accept=".csv"),
            ui.input_file("data_hyperparameters", "Choose Hyperparameters CSV File", accept=".csv"),
            ui.input_file("data_coefficients", "Choose Coefficients CSV File", accept=".csv"),
            ui.input_action_button("generate", "Generate"),
            width=400
            #style="height: 100vh; overflow-y: auto;"  # Full height and handle overflow
        ),
        output_widget("generate_saturation_curve")
    )
)




# Define server logic
def server(input, output, session):
    @render_widget
    @reactive.event(input.generate)
    def generate_saturation_curve():
        #print("Generate button clicked!")
        # Check if the required files are uploaded
        if input.data_saturation() is None or input.data_hyperparameters() is None or input.data_coefficients() is None:
            return
        #print("Files are uploaded successfully.")
        # Load the datasets
        data = pd.read_csv(input.data_saturation()[0]['datapath'])
        hyperparams = pd.read_csv(input.data_hyperparameters()[0]['datapath'])
        coefs = pd.read_csv(input.data_coefficients()[0]['datapath'])
        
        #Arranging the data alphabetically
        # Step 1: Extract the first column (to be kept as is)
        first_column = data.iloc[:, 0]

        # Step 2: Sort the remaining columns alphabetically
        sorted_columns = sorted(data.columns[1:])
        
        # Step 3: Recombine the first column with the sorted columns
        data_sorted = pd.concat([first_column, data[sorted_columns]], axis=1)
        data = data_sorted
        
        hyperparams_sorted_columns = sorted(hyperparams.columns[:])
        hyperparams_sorted = hyperparams[hyperparams_sorted_columns]
        hyperparams = hyperparams_sorted
        
        coefs_sorted = coefs.sort_values(by = "rn")
        coefs = coefs_sorted
        
        #Extracting the hyperparameters
        thetas = hyperparams.filter(like="thetas")
        alphas = hyperparams.filter(like="alphas")
        gammas = hyperparams.filter(like="gammas")

        # Get the media channels
        ds = data.columns[0]
        data = data.drop(columns=[ds])
        paid_media_spends = data.columns

        # Geometric Adstock Function
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

        # Compute the inflexions
        def transform(i):
            #print(f"Processing {paid_media_spends[i]} at index {i}")  # Add a print statement for debugging
            x = data[paid_media_spends[i]].values
            theta = thetas.iloc[0,i]  #theta = thetas.iloc[i].values[0]
            transform = adstock_geometric(x, theta)
            return transform["x_decayed"]


        #print(data.shape)  # Inspect the shape of the dataframe
        #print(paid_media_spends)  # Check the column names of paid media spends


        x_trans = {col: transform(i) for i, col in enumerate(paid_media_spends)}
        #x_trans = {col: transform(i) for i, col in enumerate(paid_media_spends[:len(paid_media_spends)])}

        inflexions = {col: np.dot([min(x_trans[col]), max(x_trans[col])], [1 - gammas.iloc[0,i], gammas.iloc[0,i]]) for i, col in enumerate(paid_media_spends)}

        # Response function
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

        # Simulate for max, avg, and response
        def create_curve_data(i):
            max_spends = data[paid_media_spends[i]].max()
            avg_spends = data[paid_media_spends[i]].mean()
            get_max_x = max_spends * 1.2
            simulate_spend = np.linspace(0, get_max_x, 100)

            simulate_response = fx_objective(simulate_spend, coefs.iloc[i]['coef'], alphas.iloc[0,i], inflexions[paid_media_spends[i]], 0, get_sum=False)
            simulate_response_max = fx_objective([max_spends], coefs.iloc[i]['coef'], alphas.iloc[0,i], inflexions[paid_media_spends[i]], 0, get_sum=False)
            simulate_response_avg = fx_objective([avg_spends], coefs.iloc[i]['coef'], alphas.iloc[0,i], inflexions[paid_media_spends[i]], 0, get_sum=False)
            
            return pd.DataFrame({
                "paid_media_spends": [paid_media_spends[i]] * len(simulate_spend),
                "spend": simulate_spend,
                "total_response": simulate_response
            }), pd.DataFrame({
                "paid_media_spends": [paid_media_spends[i]],
                "max_spend": max_spends,
                "response": simulate_response_max
            }), pd.DataFrame({
                "paid_media_spends": [paid_media_spends[i]],
                "avg_spend": avg_spends,
                "response": simulate_response_avg
            })

        plotDF_scurve, max_spends_dt, avg_spends_dt = zip(*[create_curve_data(i) for i in range(len(paid_media_spends))])

        # Merge the results into a single DataFrame
        plotDF_scurve = pd.concat(plotDF_scurve, ignore_index=True)
        max_spends_dt = pd.concat(max_spends_dt, ignore_index=True)
        avg_spends_dt = pd.concat(avg_spends_dt, ignore_index=True)

        
        
       
        # Define the qualitative color palette (you can use others like 'Set2', 'Set3', etc.)
        color_palette = px.colors.qualitative.Set1  # 'Set1' contains 9 distinct colors
        
        # Create Plotly chart
        fig = go.Figure()
        
        # Loop over media and assign a unique color from the palette
        for i, media in enumerate(paid_media_spends):
            # Replace underscores with spaces in the media name for the legend
            media_name = media.replace('_', ' ')  # Remove underscores
        
            # Get the color from the palette based on the index
            color = color_palette[i % len(color_palette)]  # Cycle through colors if there are more media than colors
            
            # Create a legend group for each media to ensure line, circle, and triangle are linked
            legend_group = f"media_{i}"
        
            # # Add line trace with the color
            # fig.add_trace(go.Scatter(x=plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['spend'],
            #                          y=plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['total_response'],
            #                          mode='lines',
            #                          name=media_name,  # Use the modified name without underscores
            #                          line=dict(color=color),
            #                          legendgroup=legend_group))  # Apply the same legend group to the line
            
            # Add line trace with the color
            fig.add_trace(go.Scatter(
                x=plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['spend'],
                y=plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['total_response'],
                mode='lines',
                name=media_name,  # Use the modified name without underscores
                line=dict(color=color),
                hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Spend:</b> <b>{int(spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for spend, response in zip(
                plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['spend'],
                plotDF_scurve[plotDF_scurve['paid_media_spends'] == media]['total_response'])],
                hoverinfo='text',  # Show the custom hovertext
                legendgroup=legend_group))  # Apply the same legend group to the line
        
            # Add max spend scatter points with the same color as the line
            max_spend_data = max_spends_dt[max_spends_dt['paid_media_spends'] == media]
            # fig.add_trace(go.Scatter(x=max_spend_data['max_spend'],
            #                          y=max_spend_data['response'],
            #                          mode='markers',
            #                          showlegend=False,  # Do not show in the legend for each media
            #                          marker=dict(symbol="circle", size=8, color=color),
            #                          legendgroup=legend_group))  # Apply the same legend group to max markers
            
            fig.add_trace(go.Scatter(
            x=max_spend_data['max_spend'],
            y=max_spend_data['response'],
            mode='markers',
            showlegend=False,  # Do not show in the legend for each media
            marker=dict(symbol="circle", size=8, color=color),
            hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Maximum Spend:</b> <b>{int(max_spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for max_spend, response in zip(
            max_spend_data['max_spend'], max_spend_data['response'])],
            hoverinfo='text',  # Show the custom hovertext for max spend
            legendgroup=legend_group))  # Apply the same legend group to max markers
        
            # Add avg spend scatter points with the same color as the line
            avg_spend_data = avg_spends_dt[avg_spends_dt['paid_media_spends'] == media]
            # fig.add_trace(go.Scatter(x=avg_spend_data['avg_spend'],
            #                          y=avg_spend_data['response'],
            #                          mode='markers',
            #                          showlegend=False,  # Do not show in the legend for each media
            #                          marker=dict(symbol="triangle-up", size=8, color=color),
            #                          legendgroup=legend_group))  # Apply the same legend group to avg markers
            
            fig.add_trace(go.Scatter(
            x=avg_spend_data['avg_spend'],
            y=avg_spend_data['response'],
            mode='markers',
            showlegend=False,  # Do not show in the legend for each media
            marker=dict(symbol="triangle-up", size=8, color=color),
            hovertext=[f"<b>Media:</b> <b>{media_name}</b><br><b>Average Spend:</b> <b>{int(avg_spend):,}</b><br><b>Sales Volume:</b> <b>{int(response):,}</b>" for avg_spend, response in zip(
            avg_spend_data['avg_spend'], avg_spend_data['response'])],
            hoverinfo='text',  # Show the custom hovertext for avg spend
            legendgroup=legend_group))  # Apply the same legend group to avg markers
        
        # Add a single scatter trace for Max Spend (circle) to show in legend (static)
        fig.add_trace(go.Scatter(x=[None], y=[None],  # Empty x and y values to only show in legend
                                 mode='markers',
                                 name="Max Spend",  # Single legend entry for Max Spend
                                 showlegend=True,    # Show in legend
                                 marker=dict(symbol="circle", size=8, color=color_palette[0]),
                                 legendgroup="max_spend_group"))  # Link to a single group
        
        # Add a single scatter trace for Avg Spend (triangle) to show in legend (static)
        fig.add_trace(go.Scatter(x=[None], y=[None],  # Empty x and y values to only show in legend
                                 mode='markers',
                                 name="Avg Spend",  # Single legend entry for Avg Spend
                                 showlegend=True,    # Show in legend
                                 marker=dict(symbol="triangle-up", size=8, color=color_palette[0]),
                                 legendgroup="avg_spend_group"))  # Link to a single group
        
        # Update layout to remove gridlines
        fig.update_layout(title="<b>Saturation Curves</b>",
                          title_x=0.5,  # Center the title
                          title_xanchor='center',  # Anchor the title at the center
                          xaxis_title="<b>Digital Spends (in USD)</b>",
                          yaxis_title="<b>Sales Volume</b>",
                          xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False),
                          paper_bgcolor="white",  # Set the background color of the paper to white
                          plot_bgcolor="white")  # Set the background color of the plot area to white)
        
        # Return figure
        return fig
                     


# Create the Shiny app
app = App(app_ui, server)

nest_asyncio.apply()

#app.run()

# Run the app
if __name__ == "__main__":
    app.run()
