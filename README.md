# ROC-AUC-PC

[![Test](https://github.com/delta-io/delta/actions/workflows/test.yaml/badge.svg)](https://github.com/delta-io/delta/actions/workflows/test.yaml)
[![License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg)](https://github.com/molmodcs/roc-auc-pc/LICENSE.txt)
[![PyPI](https://img.shields.io/pypi/v/delta-spark.svg)](https://pypi.org/project/delta-spark/)

{NAME} is a Python tool to build a ROC (Receiver operating characteristic) curve and a Predictiveness Curve(link).

ROC is a curve that describes the performance of a binary classifier by plotting the relationship between the true positive rate and the false positive rate.
PC (Predictiveness Curve) is a curve that measures the ability of a Virtual Screening program to separate the data in true positives (true active) and false positive(decoys) by plotting the Cumulative Distribution Function (CDF) of the data.

## Getting Started

```git clone https://github.com/molmodcs/roc-auc-pc.git```

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

``
matplotlib==3.5.2
numpy==1.22.3
pandas==1.4.2
scikit_learn==1.1.0
scipy==1.8.0
statsmodels==0.13.2
``

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Matheus Campos de Mattos** - (https://github.com/matheuscamposmtt)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

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
