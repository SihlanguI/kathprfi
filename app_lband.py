import dash
from dash import dcc, html, Input, Output, dash_table
from statsmodels.distributions.empirical_distribution import ECDF
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Sample data (replace with your actual datasets)
band = "L band"  # Example band
time_range = np.arange(24)
frequency2 = np.load('frequency_lband.npy')
month_labels = ['AUG 2024', 'SEPT 2024', 'OCT 2024', 'NOV 2024', 'DEC 2024']

# Load the summary table
summary_df = pd.read_csv("RFI_summary_statistics_lband.csv")

# Define file paths
#  "December 2022": "assets/RFI_Datasets/fractional_RFI_December_2022.csv",
file_paths = {
    "August 2024": "assets/l_band_data/RFI_DATASETS_lband/fractional_RFI_AugustL_2024.csv",
    "September 2024": "assets/l_band_data/RFI_DATASETS_lband/fractional_RFI_SeptemberL_2024.csv",
    "October 2024": "assets/l_band_data/RFI_DATASETS_lband/fractional_RFI_OctoberL_2024.csv",
    "November 2024": "assets/l_band_data/RFI_DATASETS_lband/fractional_RFI_NovemberL_2024.csv",
    "December 2024": "assets/l_band_data/RFI_DATASETS_lband/fractional_RFI_DecemberL_2024.csv"
}

file_paths_freqs = {
    "August 2024": "assets/l_band_data/RFI_FREQ_datasets_lband/fractional_RFI_August_2024.csv",
    "September 2024": "assets/l_band_data/RFI_FREQ_datasets_lband/fractional_RFI_September_2024.csv",
    "October 2024": "assets/l_band_data/RFI_FREQ_datasets_lband/fractional_RFI_October_2024.csv",
    "November 2024": "assets/l_band_data/RFI_FREQ_datasets_lband/fractional_RFI_November_2024.csv",
    "December 2024": "assets/l_band_data/RFI_FREQ_datasets_lband/fractional_RFI_December_2024.csv"
}

datasets_freqs = {
    name: pd.read_csv(path).iloc[:, 1:].values  # Exclude the first column using iloc
    for name, path in file_paths_freqs.items()
}

# Compute nanmedian for each frequency dataset
frequency_datasets = {name: np.median(data,axis=0) for name, data in datasets_freqs.items()}
# Shaded frequency bands (example values, adjust to match your figure)
shaded_regions = [(880 , 915, 'GSM UP'),
                  (925, 960, 'GSM DOWN'),
                  (962 , 1213, 'Aircraft transpondes'),
                  (1565, 1585, 'GPS L1'),
                  (1217, 1237, 'GPS L2'),
                  (1375, 1387, 'GPS L3'),
                  (1166, 1186, 'GPS L5'),
                  (1592, 1610, 'Glonass L1'),
                  (1242, 1249, 'Glonass L2'),
                  (1502,1503, 'Glonass L3'),
                  (1575,1576, 'Galileo E1'),
                  (1176, 1177,'Galileo E5a'),
                  (1207,1208 ,'Galileo E5b')
]

# Load datasets into a dictionary
loaded_datasets = {
    name: pd.read_csv(path).iloc[:, 1:].values  # Exclude the first column using iloc
    for name, path in file_paths.items()
}
# Compute nanmean for each dataset
datasets = {name: np.nanmedian(data, axis=0) for name, data in loaded_datasets.items()}

# Prepare data for boxplot and bar plot
datasets_mean = {name: np.nanmean(data, axis=0) for name, data in loaded_datasets.items()}
data_for_boxplot = datasets_mean.values()
# Remove NaNs from data_for_boxplot
clean_data = [pd.Series(month_data).dropna().values for month_data in data_for_boxplot]
#data_for_boxplot = [data.flatten() for data in loaded_datasets.values()]
average_occupancy = [np.nanmean(m_data) for m_data in clean_data]
yerr = [np.nanstd(d) for d in loaded_datasets.values()]

# Compute ECDF data
ecdf_data = {name: ECDF(data.flatten()) for name, data in loaded_datasets.items()}
# Combine frequency datasets into a dictionary
# 'DEC 2022': pd.DataFrame(dataset_F[0][0]).median(),

# Create a 2D dataset of time and frequency for the heatmap
time_frequency_datasets = {
    "AUG 2024": np.outer(datasets["August 2024"], frequency_datasets['August 2024']),
    "SEPT 2024": np.outer(datasets["September 2024"], frequency_datasets['September 2024']),
    "OCT 2024": np.outer(datasets["October 2024"], frequency_datasets['October 2024']),
    "NOV 2024": np.outer(datasets["November 2024"], frequency_datasets['November 2024']),
    "DEC 2024": np.outer(datasets["December 2024"], frequency_datasets['December 2024']),
}

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("RFI Monitoring Dashboard"), 
    # Summary Section
    html.Div([
        html.H3("Summary"),
        html.P("""
            Welcome to the RFI monitoring dashboard! This tool monitors Radio Frequency Interference (RFI) activities on the MeerKAT site, 
                providing up-to-date statistics on RFI trends over the months. Using the KATHPRFI pipelines as its foundation, 
                the dashboard offers insights into RFI patterns to help us monitor and manage interference more effectively. 

        """)]),
    dcc.Graph(id='time-plot'),
    dcc.Graph(id='box-plot'),
    dcc.Graph(id='bar-plot'),
    dcc.Graph(id='ecdf-plot'),
    dcc.Graph(id='frequency-plot'),
    dcc.Graph(id='time_frequency-plot'),

    html.H3("Monthly Reports"),
    html.Ul([
        html.Li(html.A("August 2024 Report", href="/assets/l_band_data/RFI_Reports_21024_2025/RFI-Report-August-2024.pdf", target="_blank")),
        html.Li(html.A("September 2024 Report", href="/assets/l_band_data/RFI_Reports_21024_2025/RFI-Report-September-2024.pdf", target="_blank")),
        html.Li(html.A("October 2024 Report", href="/assets/l_band_data/RFI_Reports_21024_2025/RFI-Report-October-2024.pdf", target="_blank")),
        html.Li(html.A("November 2024 Report", href="/assets/l_band_data/RFI_Reports_21024_2025/RFI-Report-November-2024.pdf", target="_blank")),
        html.Li(html.A("December 2024 Report", href="/assets/l_band_data/RFI_Reports_21024_2025/RFI-Report-December-2024.pdf", target="_blank")),
        html.Li(html.A("January 2025 Report", href="/assets/l_band_data/RFI_Reports_21024_2025/RFI-Report-January-2025.pdf", target="_blank")),
        # Add links for all other months
    ]),
    html.H3("Monthly Fractional RFI Flagging Datasets"),
    html.Ul([
        html.Li(html.A("August 2024 Dataset", href="/assets/l_band_data/RFI_DATASETS_lband/fractional_RFI_AugustL_2024.csv", target="_blank")),
        html.Li(html.A("September 2024 Dataset", href="/assets/l_band_data/RFI_DATASETS_lband/fractional_RFI_SeptemberL_2024.csv", target="_blank")),
        html.Li(html.A("October 2024 Dataset", href="/assets/l_band_data/RFI_DATASETS_lband/fractional_RFI_OctoberL_2024.csv", target="_blank")),
        html.Li(html.A("November 2024 Dataset", href="/assets/l_band_data/RFI_DATASETS_lband/fractional_RFI_NovemberL_2024.csv", target="_blank")),
        html.Li(html.A("December 2024 Dataset", href="/assets/l_band_data/RFI_DATASETS_lband/fractional_RFI_DecemberL_2024.csv", target="_blank")),
    ]),
    html.H3("Key RFI Statistics"),
    dash_table.DataTable(
        data=summary_df.to_dict("records"),
        columns=[{"name": col, "id": col} for col in summary_df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "10px"},
        style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"}
    )
])

# Callback to update plots based on selected month
@app.callback(
    [Output('time-plot', 'figure'),
     Output('box-plot', 'figure'),
     Output('bar-plot', 'figure'),
     Output('ecdf-plot', 'figure'),
     Output('frequency-plot', 'figure'),
     Output('time_frequency-plot', 'figure')],
    [Input('time-plot', 'clickData')]
)
def update_plots(clickData):
    selected_month = None
    if clickData:
        selected_month = month_labels[clickData['points'][0]['curveNumber']]

    # Time plot
    time_fig = go.Figure()
    for month, data in datasets.items():
        time_fig.add_trace(go.Scatter(
            x=time_range, y=data, mode='lines', name=month,
            line=dict(width=10 if month == selected_month else 3, shape='hv'),
            opacity=1 if month == selected_month else 0.5
        ))
    time_fig.update_layout(
        title=f'RFI as a function of time for {band}-band HH',
        xaxis_title='Time of the day [UTC]',
        yaxis_title='Fractional RFI flagged',
    )

    # Box plot
    box_fig = go.Figure()
    for i, month in enumerate(month_labels):
        box_fig.add_trace(go.Box(
            y=clean_data[i], name=month,
            marker_color='blue' if month == selected_month else 'gray'
        ))
    box_fig.update_layout(
        title='Box Plot of RFI distribution',
        xaxis_title='Months',
        yaxis_title='Fractional RFI flagged',
    )

    # Bar plot
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=month_labels, y=average_occupancy, name='Average RFI Level',
        error_y=dict(type='data', array=yerr, visible=True),
        marker_color=['blue' if month == selected_month else 'gray' for month in month_labels]
    ))
    bar_fig.update_layout(
        title='Average RFI Levels with Variance',
        yaxis_type='log',
        xaxis_title='Month',
        yaxis_title='RFI Level',
    )

    # ECDF plot
    ecdf_fig = go.Figure()
    for month, ecdf_vals in ecdf_data.items():
        ecdf_fig.add_trace(go.Scatter(
            x=ecdf_vals.x, y=ecdf_vals.y, mode='lines', name=month,
            line=dict(width=10 if month == selected_month else 3),
            opacity=1 if month == selected_month else 0.5
        ))
    ecdf_fig.update_layout(
        yaxis_type="log",
        title='ECDF of RFI Levels (severity of received RFI)',
        xaxis_title='RFI Level',
        yaxis_title='Cumulative Probability',
    )

    # Frequency plot
    freq_fig = go.Figure()
    for month, data in frequency_datasets.items():
        freq_fig.add_trace(go.Scatter(
            x=frequency2, y=data, mode='lines', name=month,
            line=dict(width=10 if month == selected_month else 3),
            opacity=1 if month == selected_month else 0.5
        ))
    
    # # Add shaded bands and labels on the same axis as the frequency data
    # for xmin, xmax, label in shaded_regions:
    # # Shaded region
    #     freq_fig.add_shape(
    #     type="rect",
    #     x0=xmin, x1=xmax, y0=0, y1=1,  # Extend to data range
    #     fillcolor="gray",
    #     opacity=0.3,
    #     layer="below",
    #     line_width=0
    # )

    # # Add label in the same x-axis
    # freq_fig.add_annotation(
    #     x=(xmin + xmax) / 2,  # Positioning label slightly below max Y
    #     text=label,
    #     showarrow=False,
    #     font=dict(size=10),
    #     textangle=90,
    #     xanchor='center', yanchor='top'
    # )
    freq_fig.update_layout(
        title=f'RFI as a function of frequency for {band}-band HH',
        xaxis_title='Frequency [MHz]',
        yaxis_title='Fractional RFI flagged',
    )

    # Create a time_vs_frequency figure
    time_freq_fig = go.Figure()

    # Add heatmaps for each month (initially hidden except AUG 2024)
    months = list(time_frequency_datasets.keys())
    traces = []

    for month in months:
        heatmap = go.Heatmap(
            x=time_range,
            y=frequency2,
            z=time_frequency_datasets[month].T,
            colorscale="Viridis",
            colorbar=dict(title="Intensity"),
            visible=(month == "AUG 2024"),  # Only show the first month initially
            name=month,
        )
        traces.append(heatmap)

    time_freq_fig.add_traces(traces)

    # Create dropdown menu
    dropdown_buttons = [
        {
            "label": month,
            "method": "update",
            "args": [
                {"visible": [m == month for m in months]},  # Show only the selected month
                {"title.text": f"Time vs Frequency Heatmap for {month}"},  # Update title properly
            ],
        }
        for month in months
    ]

    # Update layout with dropdown
    time_freq_fig.update_layout(
        title=dict(text="Time vs Frequency Heatmap"),  # Set title correctly
        xaxis_title="Time",
        yaxis_title="Frequency",
        updatemenus=[
            {
                "buttons": dropdown_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "y": 1.15,
            }
        ],
    )


    return time_fig, box_fig, bar_fig, ecdf_fig, freq_fig, time_freq_fig

# Run the app
import webbrowser
if __name__ == '__main__':
    webbrowser.open("http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)

