# Deep-Learning-powered-Iterative-Combinatorial Auctions

This is a piece of software used for computing the outcome of the Deep Neural Networks based Pseudo VCG Mechanism (PVM). The algorithm is described in detail in the following paper:

**Deep Learning-powered Iterative Combinatorial Auctions**  
Jakob Weissteiner and Sven Seuken. In Proceedings of the Thirty-fourth AAAI Conference on Artificial Intelligence (AAAI-20), New York, NY, February 2020. Forthcoming. Working paper version from Dec 2019: [[pdf](https://arxiv.org/pdf/1907.05771.pdf)]


If you use this software for academic purposes, please cite the above in your work. Bibtex for this reference is as follows:

```
@InProceedings{weissteiner2019workingpaper,
	author = {Weissteiner, Jakob and Seuken, Sven},
	title = {Deep Learning-powered Iterative Combinatorial Auctions},
	booktitle = {Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI-20). Forthcoming.},
	year = {2020},
}
```

## Installation

Requires Python 3.6 and Java 8 (or later).
Dependencies are CPLEX and SATS (http://spectrumauctions.org/). The file sats.jar is already provided in source/lib. The file cplex.jar needs to be added to the same folder: source/lib.

## Example: How to run PVM for a specific valuation model provided from the Spectrum Auction Test Suite (SATS).

Can be found in the file Test_PVM_github.py
