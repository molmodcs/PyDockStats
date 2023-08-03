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

# Column 1: Form to upload ligands and decoys files separately
with col1:
    st.subheader("Upload Ligands and Decoys")
    ligands_file = st.file_uploader("Choose a .csv or .lst file for ligands", type=['csv', 'lst'])
    decoys_file = st.file_uploader("Choose a .csv or .lst file for decoys", type=['csv', 'lst'])

    # Check if both files are uploaded
    if ligands_file and decoys_file:
        ligands_df = load_file(ligands_file.name)
        ligands_df.assign(activity=1)
        decoys_df = load_file(decoys_file.name)
        decoys_df.assign(activity=0)
        
        st.success("Files uploaded successfully.")

        # Combine ligands and decoys data into a single dataframe
        df = pd.concat([ligands_df, decoys_df], ignore_index=True)

        # Preprocess the combined data
        names, scores, actives = preprocess(df)

        # Calculate the PC and the ROC
        pc, roc = generate_data(names, scores, actives)

        data['pc'] = pc
        data['roc'] = roc

        # Display the combined data
        st.subheader("Data Preview")
        st.dataframe(df)

# Column 2: Graphs (you can keep the plotting code as it is, but use data['pc'] and data['roc'])
with col2:
    # Plot Predictiveness Curve
    st.subheader("Predictiveness Curve")
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

    # Plot ROC curve
    st.subheader("ROC (Receiver Operating Characteristic)")
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
