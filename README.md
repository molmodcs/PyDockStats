# PyDockStats

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/molmodcs/roc-auc-pc/blob/main/LICENSE)

PyDockStats is a Python tool that builds a ROC (Receiver operating characteristic) curve and a [Predictiveness Curve](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0100-8).

The script starts creating a logistic regression model from the data, and with the predictions it creates the curves. ROC is a curve that describes the performance of a binary classifier by plotting the relationship between the true positive rate and the false positive rate.
PC (Predictiveness Curve) is a curve that measures the ability of a Virtual Screening program to separate the data in true positives (true active) and false positive (decoys) by plotting a Cumulative Distribution Function (CDF).

## Getting Started

```git clone https://github.com/molmodcs/roc-auc-pc.git```

The main file is the "pc_roc.py" which generates the Predictiveness curves and the ROC curves given a dataset of decoys and ligands with their respectives IDs or names, docking scores (decimals separated by dot (.) ) and activities (0 or 1).The script is compatible with any number of programs, but be aware that if there are too many programs the plot will be confusing. The input data (.csv) must be separated into columns for each docking program:

|id_program1|scores_program1|activity_program1|id_program2|scores_program2|activity_program2|
|-----------|---------------|-----------------|-----------|---------------|-----------------|
|molecule1  |-12.3          |0                |molecule4  |3.6            |0                |

Example:

|surf_id                                                                                    |surf_scores|surf_actives|icm_id   |icm_scores|icm_actives|vina_id  |vina_scores|vina_actives|
|-------------------------------------------------------------------------------------------|-----------|------------|---------|----------|-----------|---------|-----------|------------|
|decoy1565                                                                                  |16.76      |0           |decoy428 |-54.926393|0          |decoy564 |-13.9      |0           |
|ligand83                                                                                   |16.56      |1           |decoy564 |-53.988434|0          |decoy2783|-13.8      |0           |
|ligand82                                                                                   |16.56      |1           |ligand16 |-52.584761|1          |decoy298 |-13.7      |0           |
|ligand13                                                                                   |16.42      |1           |decoy2783|-52.546666|0          |ligand18 |-13.4      |1           |
|decoy564                                                                                   |16.35      |0           |ligand43 |-50.703975|1          |ligand16 |-13.3      |1           |
|decoy309                                                                                   |16.07      |0           |decoy2539|-50.534748|0          |decoy429 |-13.2      |0           |
|ligand70                                                                                   |15.82      |1           |ligand66 |-50.454789|1          |ligand19 |-13.1      |1           |
|ligand81                                                                                   |15.7       |1           |ligand53 |-50.225887|1          |ligand82 |-13.1      |1           |
|ligand8                                                                                    |15.55      |1           |ligand9  |-49.668177|1          |ligand21 |-13.1      |1           |
|decoy541                                                                                   |15.5       |0           |ligand70 |-48.878551|1          |decoy526 |-13.1      |0           |

OBS: The molecules of the programs doesn't need to be align, because the algorithm sort them for each program and the alignment will be lost.

### Prerequisites

[matplotlib](https://matplotlib.org/) (3.5.2)<br/>
[NumPy](https://numpy.org/) (1.22.3)<br/>
[pandas](https://pandas.pydata.org/) (1.4.2)<br/>
[scikit-learn](https://scikit-learn.org/stable/) (1.1.0)<br/>

## Running

The code runs at the command line:

python pc_roc.py -f data_file

There are optional arguments such as:

-n or --names: names of the programs
-o or --output: output image filename

If not specified the script will use the default parameters.

For example:

> python pc_roc.py -f input_data.csv -n gold,vina,dockthor -o out.png

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Matheus Campos de Mattos** - (https://github.com/matheuscamposmtt)
* **Luciano T. Costa** - (https://http://www.molmodcs.uff.br/)

See also the list of [contributors](https://github.com/molmodcs/roc-auc-pc/blob/3936564b42f2626d41962c3b16ef074d166d8582/contributors) who participated in this project.

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details

* xtb version 6.5.0 (746241f) compiled by 'ehlert@majestix' on 2022-05-15

   xtb is free software: you can redistribute it and/or modify it under
   the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   xtb is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

## Acknowledgments

This program is used for evaluating virtual screening programs, if you want to know more deeply how it works, check the paper which the program is based on {link}.

   
## References
Empereur-mot, C., Guillemain, H., Latouche, A. et al. Predictiveness curves in virtual screening. J Cheminform 7, 52 (2015). https://doi.org/10.1186/s13321-015-0100-8
