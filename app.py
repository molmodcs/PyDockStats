import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from docking_load import load_file
from pydockstats import preprocess, generate_data

def _max_width_(prcnt_width:int = 70):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .appview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )

_max_width_(70)

# Set the title
st.title("PyDockStats")
st.subheader("A Python tool for Virtual Screening performance analysis")

# Add description
st.markdown("""
            PyDockStats is an easy and versatile Python tool that builds ROC (Receiver operating characteristic) and Predictiveness Curve curves.

            It creates a logistic regression model from the data, and with the predictions, it creates the graphical plots. 
            - ROC curve is a graphical plot that describes the performance of a binary classifier by plotting the relationship between the true positive rate and 
            the false positive rate. 
            - PC (Predictiveness Curve) curve is a graphical plot that measures the ability of a Virtual Screening program to separate 
            the data into true positives (true active) and false positives (decoys) by plotting a Cumulative Distribution Function (CDF). 
            
            Therefore, the tool is useful when verifying Virtual Screening programs' performance and gaining more confidence when making inferences.""")
option = st.selectbox(
    'Selecione o programa',
    ('program 1', 'program 2', 'program 3'))

# Create two columns layout
col1, col2 = st.columns(2, gap='large')

data = dict()

# Column 1: Form to upload .csv file
with col1:
    st.subheader("Upload a .csv or .lst file")
    uploaded_file = st.file_uploader("Choose a .csv file", type=['csv', 'lst'])

    # Check if file is uploaded
    if uploaded_file is not None:
        df = load_file(uploaded_file.name)
        st.success("File uploaded successfully.")

        # Preprocess the data
        names, scores, actives = preprocess('mpro2.csv')

        # Calculate the PC and the ROC
        pc, roc = generate_data(names, scores, actives)

        data['pc'] = pc
        data['roc'] = roc

        # Display the data
        st.subheader("Data Preview")
        st.dataframe(df)




# Column 2: Graphs
with col2:
    st.subheader("Gráficos")


    st.subheader("Predictiveness Curve")
    # Plot Predictiveness Curve
    fig_pc = go.Figure()

    if data:
        for i, (x, y) in enumerate(zip(data['pc']['x'], data['pc']['y'])):
            fig_pc.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"Program {i} Predictiveness Curve"))
    
    fig_pc.update_layout(
        title="Predictiveness Curve and ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(orientation="h"),
        height=700,
        width=1000
    )
    st.plotly_chart(fig_pc)


    st.subheader("ROC (Receiver Operating Characteristic))")
    # Plot ROC curve
    fig_roc = go.Figure()
    if data:
        for i, (x, y) in enumerate(zip(data['roc']['x'], data['roc']['y'])):
            fig_roc.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"Program {i} ROC"))

    fig_roc.update_layout(
        title="Predictiveness Curve and ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(orientation="h"),
        height=700,
        width=1000
    )

    st.plotly_chart(fig_roc)
    # Customize layout




    # Display the figure
    
