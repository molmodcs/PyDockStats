import streamlit as st
from pydockstats import generate_plots
from app_utils import plot_curve
import plotly.graph_objects as go

def _generate_curves(program_name):
    scores, activity = st.session_state.data[program_name]['scores'], st.session_state.data[program_name]['activity']
    pc_data, roc_data = generate_plots(program_name, scores, activity)
    return pc_data, roc_data

def generate_charts():
    st.header("Charts")
    
    # Program selection
    option = st.radio("Select the program", ['All']+st.session_state.programs)

    # Plot the PC and the ROC
    st.subheader("Predictiveness Curve")
    pc_container = st.container()
    fig_pc = go.Figure(layout=dict(title=dict(text="PC"), xaxis_title="Quantile", yaxis_title="Activity probability",
                                    legend=dict(orientation="h"), height=600, width=1000, font_family="Overpass", font_size=25),
                                    )
    

    st.subheader("ROC (Receiver Operating Characteristic)")
    roc_container = st.container()
    fig_roc = go.Figure(layout=dict(title=dict(text="ROC"), xaxis_title="False Positive Rate", 
                                    yaxis_title="True Positive Rate", legend=dict(orientation="h"), height=600, width=900,
                                    font_family="Overpass", font_size=25))
    
    
    if option == 'All':
        for program_name in st.session_state.programs:
            pc_data, roc_data = _generate_curves(program_name)
            with pc_container:
                plot_curve(fig_pc, program_name, pc_data, "Predictiveness Curve")

            with roc_container:
                plot_curve(fig_roc, program_name, roc_data, "ROC")
                
    else:
        pc_data, roc_data = _generate_curves(option)
        with pc_container:
            plot_curve(fig_pc, option, pc_data, "Predictiveness Curve")

        with roc_container:
            plot_curve(fig_roc, option, roc_data, "ROC")
        
    pc_container.plotly_chart(fig_pc)
    roc_container.plotly_chart(fig_roc)

    return fig_pc, fig_roc