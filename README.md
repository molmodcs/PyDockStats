# About PyDockStats

<p align="center">
  <img src="/images/PDS.png" alt="PyDockStats" style="max-width:0.5rem;max-height:0.5rem;"/>
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/molmodcs/roc-auc-pc/blob/main/LICENSE)

PyDockStats is a versatile and easy-to-use Python tool that builds [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) (Receiver Operating Characteristic) and [Predictiveness Curve](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0100-8) plots. It also calculates BEDROC and enrichment factor values.

The script starts by creating a logistic regression model from the input data, and with the predictions, it generates graphical plots. The ROC curve visually evaluates the performance of a binary classifier by plotting the true positive rate versus the false positive rate. The Predictiveness Curve (PC) measures the ability of a Virtual Screening program to distinguish true positives (active compounds) from false positives (decoys) through a [Cumulative Distribution Function (CDF)](https://en.wikipedia.org/wiki/Cumulative_distribution_function).

This tool is highly valuable for verifying the performance of Virtual Screening programs and gaining confidence in data-driven inferences.

**The web version of PyDockStats:**
https://pydockstats.streamlit.app/

## Recommended First Step: Isolated Environment (Optional)

This step is optional, but it is highly recommended.

Using an isolated environment helps avoid conflicts with other Python packages on your machine and keeps your system clean.

### Option A: Conda

1. Create an environment:

```bash
conda create --name pydockstats-env python=3.12
```

2. Activate it:

```bash
conda activate pydockstats-env
```

### Option B: venv

1. Create an environment:

```bash
python -m venv .venv
```

2. Activate it:

Linux/MacOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

## Installation

Install directly from GitHub with pip:

```bash
pip install git+https://github.com/molmodcs/PyDockStats
```

After installation, run `pydockstats` directly (without `python -m`).

Dependencies are installed automatically by pip in your active environment.

## Usage

Run with an input data file in `.csv`, `.xlsx`, or `.ods` format.

### Primary command

```bash
pydockstats -f data_file
```

### Alternative command

```bash
python -m PyDockStats -f data_file
```

### Optional arguments

- `-p` or `--programs`: Names of the programs.
- `-o` or `--output`: Output image filename.
- `-m` or `--model`: Model type (default: logistic_regression).

If no optional arguments are specified, the script uses default parameters.

Example:

```bash
pydockstats -f input_data.csv -p gold,vina,dockthor -o out.png -m logistic_regression
```

where _gold, vina and dockthor_ are related to the molecular docking programs used. Please this sequence must be the same to the input data file.

## Input Data Format

PyDockStats builds Predictiveness and ROC curves from datasets containing decoys/ligands IDs, docking scores (decimals separated by dots), and activity labels (0 or 1).

Input data (`.csv`, `.xlsx`, `.ods`) must include columns for each docking program:

| id_program1 | scores_program1 | activity_program1 | id_program2 | scores_program2 | activity_program2 |
| ----------- | --------------- | ----------------- | ----------- | --------------- | ----------------- |
| molecule1   | -12.3           | 0                 | molecule4   | 3.6             | 0                 |

### Example Input

| surf_id   | surf_scores | surf_actives | icm_id    | icm_scores | icm_actives | vina_id   | vina_scores | vina_actives |
| --------- | ----------- | ------------ | --------- | ---------- | ----------- | --------- | ----------- | ------------ |
| decoy1565 | 16.76       | 0            | decoy428  | -54.926393 | 0           | decoy564  | -13.9       | 0            |
| ligand83  | 16.56       | 1            | decoy564  | -53.988434 | 0           | decoy2783 | -13.8       | 0            |
| ligand82  | 16.56       | 1            | ligand16  | -52.584761 | 1           | decoy298  | -13.7       | 0            |
| ligand13  | 16.42       | 1            | decoy2783 | -52.546666 | 0           | ligand18  | -13.4       | 1            |

Note: Molecules in different programs do not need to be aligned, as the algorithm sorts them independently.

## Result Plots

The results of the analysis, including ROC and Predictiveness Curve plots, will be saved as image files in the specified output directory. Example of synthetic plots are displayed below:

<p align="center">
  <img src="images/pc.png" alt="ROC Curve" width="49%">
  <img src="images/roc.png" alt="Predictiveness Curve" width="49%">
</p>

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct and the process for submitting pull requests.

## Authors

- **Matheus Campos de Mattos** - [GitHub](https://github.com/matheuscamposmtt)
- **Luciano T. Costa** - [Website](http://www.molmodcs.uff.br/) | [GitHub](https://github.com/molmodcs)

See the list of [contributors](https://github.com/molmodcs/roc-auc-pc/blob/3936564b42f2626d41962c3b16ef074d166d8582/contributors) who participated in this project.

## License

This project is licensed under the GNU Lesser General Public License - see the [LICENSE.md](LICENSE.md) file for details.

PyDockStats version 1.0 (746241f), compiled by `matheuscamposmattos@id.uff.br` on 2022-07-25.

## Acknowledgments

This program evaluates and classifies the results from virtual screening. For a deeper understanding of its operation, check the [paper](https://doi.org/10.1186/s13321-015-0100-8) upon which it is based.

## References

Empereur-Mot, C., Guillemain, H., Latouche, A. et al. Predictiveness curves in virtual screening. J Cheminform 7, 52 (2015). https://doi.org/10.1186/s13321-015-0100-8

## For Developers

Use this section only if you want to work on the project source code.

1. Clone the repository:

```bash
git clone https://github.com/molmodcs/PyDockStats.git
cd PyDockStats
```

2. Create and activate an isolated environment (choose one):

- Using Conda:

```bash
conda create --name PyDockStats python=3.10
conda activate PyDockStats
```

- Using venv:

```bash
python -m venv .venv
```

Linux/MacOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

3. Install in editable mode:

```bash
pip install -e .
```

4. Run locally:

```bash
pydockstats -f data_file
```
