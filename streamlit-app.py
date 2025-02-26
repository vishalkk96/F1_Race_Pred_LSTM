import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from Models import Lap_Sim_Funcs as lsf
import warnings

# Set page config for a wider layout
st.set_page_config(
    page_title="F1 Race Predictor LSTM",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the app more visually appealing with F1 colors
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF1801 !important;
        color: white !important;
    }
    h1, h2, h3 {
        color: #15151E;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Create header
st.title("🏎️ F1 Race Predictor LSTM")
st.markdown("""
    <div style="background-color: #15151E; padding: 10px; border-radius: 5px; margin-bottom: 20px; color: white;">
        <h3 style="margin: 0;">A two-stage LSTM Model to predict driver speeds and final outcomes of Formula 1 Races</h3>
    </div>
""", unsafe_allow_html=True)

# Create sidebar for inputs with F1 styling
st.sidebar.markdown("""
    <div style="background-color: #FF1801; padding: 10px; border-radius: 5px; margin-bottom: 20px; color: white;">
        <h2 style="margin: 0; text-align: center;">Race Selection</h2>
    </div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Data/F1_Sector_Dataset.csv')
        race, qual, drql, drvr = lsf.preprocess_spd_model(df)
        test_df = race.loc[race['Date'] >= '2024-05-01'].copy().reset_index(drop=True)
        
        # Get list of unique drivers in test period from drvr DataFrame
        # (which contains the driver information)
        test_drivers = sorted(drvr[(drvr['Date'] >= '2024-05-01')]['Driver'].unique().tolist())
        
        return df, race, qual, drql, drvr, test_df, test_drivers
    except Exception as e:
        st.error(f"Error in load_data: {str(e)}")
        raise e

try:
    df, race, qual, drql, drvr, test_df, available_drivers = load_data()
    available_drivers = sorted(available_drivers)
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please make sure the data file 'Data/F1_Sector_Dataset.csv' exists and is valid.")
    data_loaded = False

if data_loaded:
    # Get available races for the test period
    available_races = sorted(test_df['Race'].unique().tolist())

    # Create input components
    vw_race = st.sidebar.selectbox(
        "Select Race to View",
        options=available_races,
        index=available_races.index('United States Grand Prix') if 'United States Grand Prix' in available_races else 0
    )

    # Filter drivers to only those who participated in the selected race
    race_drivers = sorted(drvr[((drvr['Date'] >= '2024-05-01') & (drvr['Race'] == vw_race))]['Driver'].unique().tolist())
    
    default_drivers = ['VER', 'NOR', 'LEC', 'PIA']
    # Make sure all default drivers exist in available_drivers, otherwise use the first 4
    default_selection = [d for d in default_drivers if d in race_drivers]
    if len(default_selection) < 1:
        default_selection = race_drivers[:min(4, len(race_drivers))]

    drv_lst = st.sidebar.multiselect(
        "Select Drivers to View (4 recommended)",
        options=race_drivers,
        default=default_selection
    )

    if not drv_lst:  # If no driver is selected, use the default selection
        drv_lst = default_selection
        st.sidebar.warning("Using default driver selection")

    dsp_drvr = st.sidebar.selectbox(
        "Select Driver for Error Metrics",
        options=race_drivers,
        index=race_drivers.index('VER') if 'VER' in race_drivers else 0
    )

    nlaps = st.sidebar.number_input(
        "Laps after which simulation starts",
        min_value=10,
        max_value=30,
        value=20
    )

    # Add some explanatory text
    st.sidebar.markdown("""
        <div style="background-color: #15151E; padding: 10px; border-radius: 5px; margin-top: 20px; color: white;">
            <h4 style="margin-top: 0;">How to use this app:</h4>
            <ol>
                <li>Select a race from the dropdown</li>
                <li>Choose up to 4 drivers to visualize</li>
                <li>Select a driver to show detailed error metrics</li>
                <li>Adjust the lap count for when simulation begins</li>
                <li>Explore the results in the tabs on the right</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    # Show info about model architecture
    with st.sidebar.expander("Model Architecture"):
        st.markdown("""
            - **Stage 1 LSTM:** Predicts average lap speed for all drivers
            - 3 layers deep with 32 units per layer
            - Takes track and lap conditions as input
            
            - **Stage 2 LSTM:** Predicts individual driver lap speed
            - 3 layers deep with 32 units per layer
            - Takes Stage 1 output and driver characteristics as input
        """)

    # Run the model section
    if st.sidebar.button("Run Simulation", type="primary"):
        # Create a progress bar for computation
        progress_text = "Running simulation. Please wait..."
        progress_bar = st.progress(0, text=progress_text)

        # Stage 1 LSTM Model Definition
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out

        # Model Parameters
        input_size_1 = 16 
        hidden_size_1 = 32
        num_layers_1 = 3
        output_size_1 = 3 

        spd_model = LSTMModel(input_size_1, hidden_size_1, num_layers_1, output_size_1)

        input_size_2 = 23 
        hidden_size_2 = 32
        num_layers_2 = 3
        output_size_2 = 3

        pos_model = LSTMModel(input_size_2, hidden_size_2, num_layers_2, output_size_2)

        try:
            # Stage 1 LSTM Execution
            with st.spinner('Processing Stage 1 LSTM...'):
                tab = lsf.process_laps('Models/mean_lstm_32_3.pth', spd_model, test_df, qual, nlaps)
                progress_bar.progress(33, text=progress_text)

            # Process driver race data
            with st.spinner('Processing Stage 2 LSTM...'):
                drv_race = lsf.preprocess_pos_model(tab, race, qual, drql, drvr)
                test_df_2 = drv_race.loc[drv_race['Date'] >= '2024-05-01'].copy().reset_index(drop=True)
                prg = lsf.driver_laps('Models/pos_lstm_32_3.pth', pos_model, test_df_2, nlaps)
                progress_bar.progress(66, text=progress_text)

            # Calculate standings
            with st.spinner('Calculating final standings...'):
                std = lsf.rank_drivers(prg)
                rce_std = std.loc[std['Race'] == vw_race].reset_index(drop=True)
                progress_bar.progress(100, text="Simulation complete!")

            # Filter data for the selected race and drivers
            spd_dsp = tab.loc[tab['Race'] == vw_race].reset_index(drop=True)
            mean_val = spd_dsp['Err_Lap_Spd'].mean()

            pos_dsp = prg.loc[(prg['Race'] == vw_race) & (prg['Driver'].isin(drv_lst))].reset_index(drop=True)
            drv_dsp = prg.loc[(prg['Race'] == vw_race) & (prg['Driver'] == dsp_drvr)].reset_index(drop=True)
            err_sec = drv_dsp['Err_Drv_Spd'].mean()
            err_pos = drv_dsp['Err_Position'].mean()

            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Stage 1: Lap Speed", "Stage 2: Driver Speed", "Stage 2: Position", "Race Standings"])

            with tab1:
                st.subheader("Stage 1 LSTM: Average Lap Speed Prediction")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Stage 1 Metrics Visualization
                    fig1, ax1 = plt.subplots(figsize=(12, 6))
                    ax1.plot(spd_dsp['Lap'], spd_dsp['Real_Spd'], label='Real Mean Lap Speed', linestyle='-', marker='.')
                    ax1.plot(spd_dsp['Lap'], spd_dsp['Calc_Spd'], label='Calc Mean Lap Speed', linestyle='--', marker='.')
                    ax1.set_xlabel('Lap')
                    ax1.set_ylabel('Speed')
                    ax1.set_title('Stage 1 LSTM')
                    ax1.axvline(x=nlaps, color='black', linestyle='-', linewidth=1.2, label='Simulation Starts')
                    ax1.text(nlaps + 0.2, max(spd_dsp['Real_Spd'].min(), spd_dsp['Calc_Spd'].min()), 'Simulation Starts',
                             verticalalignment='center', horizontalalignment='left', color='green')
                    ax1.set_xlim(spd_dsp['Lap'].min() - 1, spd_dsp['Lap'].max() + 1)
                    ax1.legend()
                    ax1.grid(True)
                    st.pyplot(fig1)
                
                with col2:
                    # Stage 1 Error Visualization
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    ax2.bar(spd_dsp['Lap'], spd_dsp['Err_Lap_Spd'], label='Real Speed', color='orangered')
                    ax2.set_xlabel('Lap')
                    ax2.set_ylabel('Error %')
                    ax2.set_title('Stage 1 LSTM Errors')
                    ax2.axvline(x=nlaps, color='black', linestyle='-', linewidth=1.2, label='Simulation Starts')
                    ax2.axhline(y=mean_val, color='black', linestyle='--', linewidth=1, label='M.A.P.E')
                    ax2.text(x=0.5, y=mean_val + 0.1, s=f'M.A.P.E = {mean_val:.2f}', color='black', fontsize=9)
                    ax2.text(nlaps + 0.2, (spd_dsp['Err_Lap_Spd'].min() - 0.5), 'Simulation Starts',
                             verticalalignment='center', horizontalalignment='left', color='green')
                    ax2.set_xlim(spd_dsp['Lap'].min(), spd_dsp['Lap'].max()+1)
                    ax2.set_ylim(min(spd_dsp['Err_Lap_Spd'].min() - 1, -5), max(spd_dsp['Err_Lap_Spd'].max() + 1, 5))
                    ax2.grid(True)
                    st.pyplot(fig2)
                
                st.markdown("""
                    <div style="background-color: #15151E; padding: 15px; border-radius: 5px; margin-top: 20px; color: white;">
                        <h4>Stage 1 Explanation:</h4>
                        <p>This stage predicts the average lap speed for all drivers. The model takes track and weather conditions into account to forecast how the overall pace will evolve during the race.</p>
                        <p>The left chart shows the actual vs. predicted average lap speed. The right chart shows the percentage error for each lap, with the Mean Absolute Percentage Error (MAPE) indicated by the dashed line.</p>
                    </div>
                """, unsafe_allow_html=True)

            with tab2:
                st.subheader("Stage 2 LSTM: Driver Speed Prediction")
                
                # Driver colors assignment
                drivers = pos_dsp['Driver'].unique()
                colors = plt.cm.jet(np.linspace(0, 1, len(drivers)))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Stage 2 Real Speed Visualization
                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    for driver, color in zip(drivers, colors):
                        driver_data = pos_dsp[pos_dsp['Driver'] == driver]
                        ax3.plot(driver_data['Lap'], driver_data['Drv_Real_Spd'], label=f'{driver} Real_Spd', marker='x', color=color)
                    ax3.set_title('Driver Real Speed')
                    ax3.set_xlabel('Lap')
                    ax3.set_ylabel('Real Speed')
                    ax3.legend()
                    ax3.grid(True)
                    st.pyplot(fig3)
                
                with col2:
                    # Stage 2 Calculated Speed Visualization
                    fig4, ax4 = plt.subplots(figsize=(12, 6))
                    for driver, color in zip(drivers, colors):
                        driver_data = pos_dsp[pos_dsp['Driver'] == driver]
                        ax4.plot(driver_data['Lap'], driver_data['Drv_Calc_Spd'], label=f'{driver} Calc_Spd', marker='.', linestyle='--', color=color)
                    ax4.axvline(x=nlaps, color='black', linestyle='-', linewidth=1.2, label='Simulation Starts')
                    ax4.text(nlaps + 0.2, pos_dsp['Drv_Calc_Spd'].min(), 'Simulation Starts',
                             verticalalignment='center', horizontalalignment='left', color='green')
                    ax4.set_xlim(pos_dsp['Lap'].min() - 1, pos_dsp['Lap'].max() + 1)
                    ax4.set_title('Driver Pred Speed')
                    ax4.set_xlabel('Lap')
                    ax4.set_ylabel('Pred Speed')
                    ax4.legend()
                    ax4.grid(True)
                    st.pyplot(fig4)
                
                # Stage 2 Speed Errors for Selected Driver
                fig5, ax5 = plt.subplots(figsize=(12, 6))
                ax5.bar(drv_dsp['Lap'], drv_dsp['Err_Drv_Spd'], label='Real Speed', color='orangered')
                ax5.set_xlabel('Lap')
                ax5.set_ylabel('Error %')
                ax5.set_title(f'{dsp_drvr} Lap Speed Error %')
                ax5.axvline(x=nlaps, color='black', linestyle='-', linewidth=1.2, label='Simulation Starts')
                ax5.axhline(y=err_sec, color='black', linestyle='--', linewidth=1, label='M.A.P.E')
                ax5.text(x=0.5, y=err_sec + 0.1, s=f'M.A.P.E = {err_sec:.2f}', color='black', fontsize=9)
                ax5.text(nlaps + 0.2, (drv_dsp['Err_Drv_Spd'].min() - 0.5), 'Simulation Starts',
                         verticalalignment='center', horizontalalignment='left', color='green')
                ax5.set_xlim(drv_dsp['Lap'].min(), drv_dsp['Lap'].max()+1)
                ax5.set_ylim(min(drv_dsp['Err_Drv_Spd'].min() - 1, -5), max(drv_dsp['Err_Drv_Spd'].max() + 1, 5))
                ax5.grid(True)
                st.pyplot(fig5)
                
                st.markdown("""
                    <div style="background-color: #15151E; padding: 15px; border-radius: 5px; margin-top: 20px; color: white;">
                        <h4>Stage 2 Speed Explanation:</h4>
                        <p>This stage predicts individual driver speeds based on the output from Stage 1. It accounts for driver-specific factors like driving style, car performance, and tire management.</p>
                        <p>The top charts compare actual (left) vs. predicted (right) speeds for each selected driver. The bottom chart shows the error percentage for the driver selected for error metrics.</p>
                    </div>
                """, unsafe_allow_html=True)

            with tab3:
                st.subheader("Stage 2 LSTM: Driver Position Prediction")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Stage 2 Real Position Visualization
                    fig6, ax6 = plt.subplots(figsize=(12, 6))
                    for driver, color in zip(drivers, colors):
                        driver_data = pos_dsp[pos_dsp['Driver'] == driver]
                        ax6.plot(driver_data['Lap'], driver_data['Position_T'], label=f'{driver} True_Pos', marker='x', color=color)
                    ax6.set_title('Driver Real Position')
                    ax6.set_xlabel('Lap')
                    ax6.set_ylabel('Position (Z-Score)')
                    ax6.legend()
                    ax6.grid()
                    st.pyplot(fig6)
                
                with col2:
                    # Stage 2 Calculated Position Visualization
                    fig7, ax7 = plt.subplots(figsize=(12, 6))
                    for driver, color in zip(drivers, colors):
                        driver_data = pos_dsp[pos_dsp['Driver'] == driver]
                        ax7.plot(driver_data['Lap'], driver_data['Position_P'], label=f'{driver} Calc_Pos', marker='.', linestyle='--', color=color)
                    ax7.axvline(x=nlaps, color='black', linestyle='-', linewidth=1.2, label='Simulation Starts')
                    ax7.text(nlaps + 0.2, pos_dsp['Position_P'].min(), 'Simulation Starts',
                             verticalalignment='center', horizontalalignment='left', color='green')
                    ax7.set_xlim(pos_dsp['Lap'].min() - 1, pos_dsp['Lap'].max() + 1)
                    ax7.set_title('Driver Pred Position')
                    ax7.set_xlabel('Lap')
                    ax7.set_ylabel('Position (Z-Score)')
                    ax7.legend()
                    ax7.grid()
                    st.pyplot(fig7)
                
                # Stage 2 Position Errors for Selected Driver
                fig8, ax8 = plt.subplots(figsize=(12, 6))
                ax8.bar(drv_dsp['Lap'], drv_dsp['Err_Position'], label='Position Error', color='orangered')
                ax8.set_xlabel('Lap')
                ax8.set_ylabel('Error (Z-Score)')
                ax8.set_title(f'{dsp_drvr} Position Error (Z-Score)')
                ax8.axvline(x=nlaps, color='black', linestyle='-', linewidth=1.2, label='Simulation Starts')
                ax8.axhline(y=err_pos, color='black', linestyle='--', linewidth=1, label='M.A.E')
                ax8.text(x=0.5, y=err_pos + 0.1, s=f'M.A.E = {err_pos:.2f}', color='black', fontsize=9)
                ax8.text(nlaps + 0.2, (drv_dsp['Err_Position'].min() - 0.5), 'Simulation Starts',
                         verticalalignment='center', horizontalalignment='left', color='green')
                ax8.set_xlim(drv_dsp['Lap'].min(), drv_dsp['Lap'].max()+1)
                ax8.set_ylim(min(drv_dsp['Err_Position'].min() - 1, -5), max(drv_dsp['Err_Position'].max() + 1, 5))
                ax8.grid(True)
                st.pyplot(fig8)
                
                st.markdown("""
                    <div style="background-color: #15151E; padding: 15px; border-radius: 5px; margin-top: 20px; color: white;">
                        <h4>Stage 2 Position Explanation:</h4>
                        <p>This section shows how drivers' positions (represented as Z-scores) change throughout the race. Lower Z-scores indicate better positions (closer to P1).</p>
                        <p>The top charts compare actual (left) vs. predicted (right) positions for each selected driver. The bottom chart shows the error for the driver selected for error metrics.</p>
                    </div>
                """, unsafe_allow_html=True)

            with tab4:
                st.subheader("Race Standings Prediction")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Race Predicted v/s Actual Standings
                    fig9, ax9 = plt.subplots(figsize=(8, 10))
                    ax9.axis('off')
                    
                    x_left = 0.1
                    x_right = 0.9
                    n_drivers = len(rce_std)
                    
                    df_left = rce_std.sort_values(by=['Standing_P'], ascending=False).reset_index(drop=True)
                    df_left['y_left'] = np.linspace(0, 1, n_drivers)
                    df_right = rce_std.sort_values(by=['Standing_T'], ascending=False).reset_index(drop=True)
                    df_right['y_right'] = np.linspace(0, 1, n_drivers)
                    
                    driver_positions = {}
                    for idx, row in rce_std.iterrows():
                        driver = row['Driver']
                        y_left = df_left[df_left['Driver'] == driver]['y_left'].values[0]
                        y_right = df_right[df_right['Driver'] == driver]['y_right'].values[0]
                        driver_positions[driver] = {'y_left': y_left, 'y_right': y_right}
                    
                    for driver, pos in driver_positions.items():
                        y_left = pos['y_left']
                        y_right = pos['y_right']
                        
                        ax9.text(x_left - 0.05, y_left, driver, ha='right', va='center')
                        ax9.text(x_right + 0.05, y_right, driver, ha='left', va='center')
                    
                        verts = [
                            (x_left, y_left),
                            ((x_left + x_right) / 2, y_left),
                            ((x_left + x_right) / 2, y_right),
                            (x_right, y_right),
                        ]
                        codes = [Path.MOVETO,
                                 Path.CURVE4,
                                 Path.CURVE4,
                                 Path.CURVE4,
                                 ]
                        path = Path(verts, codes)
                        patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor='gray', alpha=0.5)
                        ax9.add_patch(patch)
                    
                    ax9.text(x_left - 0.05, 1.05, 'Pred Standings', ha='right', va='bottom', fontsize=12, fontweight='bold')
                    ax9.text(x_right + 0.05, 1.05, 'True Standings', ha='left', va='bottom', fontsize=12, fontweight='bold')
                    
                    st.pyplot(fig9)
                
                with col2:
                    # Post Race Correlation
                    fig10, ax10 = plt.subplots(figsize=(8, 10))
                    x = rce_std['Standing_T']
                    y = rce_std['Standing_P']
                    
                    corr_coef, _ = pearsonr(x, y)
                    
                    ax10.scatter(x, y, color='blue', label='Data Points')
                    
                    slope, intercept = np.polyfit(x, y, 1)
                    
                    x_line = np.linspace(0, 21, 100) 
                    y_line = slope * x_line + intercept
                    
                    ax10.plot(x_line, y_line, color='black', linestyle='--', label='Regression Line')
                    
                    ax10.set_xlabel('True Standing')
                    ax10.set_ylabel('Pred Standing')
                    ax10.set_title('Predicted Race Outcome')
                    
                    ax10.text(1, 20, f"Correlation = {corr_coef:.2f}", fontsize=12, color='black')
                    
                    ax10.set_xlim(0, 21)
                    ax10.set_ylim(0, 21)
                    ax10.set_xticks(range(0, 22, 2))
                    ax10.set_yticks(range(0, 22, 2))
                    ax10.grid()
                    
                    st.pyplot(fig10)
                
                st.markdown("""
                    <div style="background-color: #15151E; padding: 15px; border-radius: 5px; margin-top: 20px; color: white;">
                        <h4>Race Standings Explanation:</h4>
                        <p>The left chart shows how drivers moved between their predicted standings (left) and actual final standings (right). Lines crossing indicate prediction errors.</p>
                        <p>The right chart plots true standings vs. predicted standings, with the correlation coefficient showing how accurately the model ranked drivers. A perfect model would have all points on the diagonal line (correlation = 1.0).</p>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error running simulation: {e}")
            st.info("Please make sure all model files exist and are valid.")
    else:
        # Display initial welcome screen with project info
        st.markdown("""
            <div style="text-align: center; margin-top: 2rem; margin-bottom: 2rem;">
                <img src="https://raw.githubusercontent.com/vishalkk96/F1_Race_Pred_LSTM/main/Images/Race_Terminology.png" width="600">
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style="background-color: #15151E; padding: 20px; border-radius: 10px; color: white;">
                    <h3>📊 Two-Stage LSTM Model</h3>
                    <p>This project uses a sophisticated two-stage LSTM neural network to predict Formula 1 race outcomes:</p>
                    <ul>
                        <li><strong>Stage 1:</strong> Predicts average lap speed for all drivers</li>
                        <li><strong>Stage 2:</strong> Predicts individual driver lap speed and position</li>
                    </ul>
                    <p>The model is trained on F1 data from 2019 to April 2024 and validated on races from May 2024 onward.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background-color: #FF1801; padding: 20px; border-radius: 10px; color: white; margin-top: 20px;">
                    <h3>🏁 How to Use</h3>
                    <ol>
                        <li>Use the sidebar to select a race to analyze</li>
                        <li>Choose drivers to visualize (max 4 recommended)</li>
                        <li>Select a driver for detailed error metrics</li>
                        <li>Set how many laps to use before simulation begins</li>
                        <li>Click "Run Simulation" to see the results</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style="background-color: #15151E; padding: 20px; border-radius: 10px; color: white;">
                    <h3>📈 Key Features</h3>
                    <ul>
                        <li><strong>Lap Speed Prediction:</strong> Forecasts how fast each driver will go on each lap</li>
                        <li><strong>Position Tracking:</strong> Tracks relative positions throughout the race</li>
                        <li><strong>Race Outcome:</strong> Predicts final race standings</li>
                        <li><strong>Error Analysis:</strong> Measures prediction accuracy with M.A.P.E and M.A.E</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="background-color: #15151E; padding: 20px; border-radius: 10px; margin-top: 20px; color: white;">
                    <h3>🧠 Technical Background</h3>
                    <ul>
                        <li><strong>Neural Network:</strong> 3-layer LSTM with 32 units per layer</li>
                        <li><strong>Data Source:</strong> FastF1 library with publicly available race data</li>
                        <li><strong>Implementation:</strong> Built with PyTorch and deployed with Streamlit</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

# Add a footer with project info
st.markdown("---")
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h4>F1 Race Predictor LSTM</h4>
            <p>A two-stage LSTM Model for predicting Formula 1 race outcomes</p>
        </div>
        <div>
            <a href="https://github.com/vishalkk96/F1_Race_Pred_LSTM" target="_blank">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30">
                GitHub Repository
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)
