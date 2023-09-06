import pandas as pd
import numpy as np
import re

def _remove_extra_whitespaces(string):
    # Remove leading and trailing whitespaces
    string = string.strip()

    # Replace multiple whitespaces with a comma
    string = re.sub('\s+', ',', string)

    return string

def _read_lst(lst_file: str):
    result = ""
    header = ""
    # Open the file for reading
    with open(lst_file, 'r') as file:
        # Iterate over the lines of the file
        for line in file:
            if not line.startswith("#"):
            # Process each line
                result += _remove_extra_whitespaces(line) + "\n"
            else:
                header = _remove_extra_whitespaces(line)[2:]

    result = header + result
    new_filename = lst_file.replace(".lst", ".csv")
    with open(new_filename, 'w') as file:
        file.write(result)

    return new_filename

    # The file will be automatically closed after the 'with' block


def load_file(filename: str) -> pd.DataFrame:
    if filename.endswith((".xlsx", ".ods")):
        return pd.read_excel(filename)
    elif filename.endswith(".lst"):
        return pd.read_csv(_read_lst(filename), sep=',')

              
    return pd.read_csv(filename, sep=None, engine='python')