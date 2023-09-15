import streamlit as st
import app_utils as utils
from docking_load import load_file
import pandas as pd
from pydockstats import preprocess_data

def input_programs(max=15):
    program_name_container = st.container()
    with program_name_container:
        name_program = st.text_input("Enter the name of the program (optional)", placeholder="Autodock Vina", 
                                     max_chars=30, value="")
        add_program = st.button("Add program", help="Add a new program tab", 
                                type='secondary', disabled=len(st.session_state.programs) > max)
        if (add_program or name_program) and name_program not in st.session_state.programs:
            if name_program:
                st.session_state.programs.append(name_program)
            else:
                program_name = f"Program {len(st.session_state.programs)+1}"
                st.session_state.programs.append(program_name)
            
    
def _program_tab_content(program_name):
    col1, col2 = st.columns([4, 1], gap='medium')
    with col1:
        st.subheader(f"Upload ligands and decoys for {program_name}")

    with col2:
        remove_program = st.button("Remove program", key=f'remove_{program_name}_tab', 
                                   type='primary', help="Remove this program tab",
                                   disabled=len(st.session_state.programs) <= 1)

    # Upload files and display data preview
    ligands_file, decoys_file= utils.upload_files(program_name)

    if ligands_file and decoys_file:
        ligands_df = load_file(ligands_file.name)
        ligands_df['activity'] = 1
        decoys_df = load_file(decoys_file.name)
        decoys_df['activity'] = 0
        
        # Combine ligands and decoys data into a single dataframe
        df = pd.concat([ligands_df, decoys_df], ignore_index=True).sample(frac=1)

        # Preprocess the combined data
        scores, activity = preprocess_data(df)

        # store data of the program in session state
        st.session_state.data[program_name] = dict(scores=scores, activity=activity)

    if remove_program:
        # remove the program from session state
        st.session_state.programs.remove(program_name)
        del remove_program
        try:
            del st.session_state.data[program_name] 
        except KeyError:
            pass
        st.experimental_rerun()

def generate_tabs():
    # User can "create" tabs when adding programs
    if st.session_state['programs']:
        programs_tabs = st.tabs(st.session_state.programs)
        for i in range(len(programs_tabs)):
            with programs_tabs[i]:
                _program_tab_content(st.session_state.programs[i])