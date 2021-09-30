# Deep-Learning-powered-Iterative-Combinatorial Auctions

This is a piece of software used for computing the outcome of the Pseudo VCG Mechanism (PVM) based on deep neural networks (DNNs). The algorithm is described in detail in the following paper:

**Deep Learning-powered Iterative Combinatorial Auctions**  
Jakob Weissteiner and Sven Seuken. In Proceedings of the Thirty-fourth AAAI Conference on Artificial Intelligence (AAAI-20), New York, NY, February 2020. Working paper version from Oct 2020: [[pdf](https://arxiv.org/pdf/1907.05771v5.pdf)]

If you use this software for academic purposes, please cite the above in your work. Bibtex for this reference is as follows:

```tex
@InProceedings{weissteiner2020deep,
    author = {Weissteiner, Jakob and Seuken, Sven},
    title = {Deep Learning-powered Iterative Combinatorial Auctions},
    booktitle = {Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI-20).},
    year = {2020},
}
```

## Requirements

* Python 3.6
* Java 8 (or later)
  * Java environment variables set as described [here](https://pyjnius.readthedocs.io/en/stable/installation.html#installation)
* JAR-files ready (they should already be)
  * CPLEX (>=12.8.0): The file cplex.jar is provided in source/lib.
  * [SATS](http://spectrumauctions.org/) (>=0.6.4): The file sats.jar is provided in source/lib.
* CPLEX Python API installed as described [here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html)

## Dependencies

Prepare your python environment (whether you do that with `conda`, `virtualenv`, etc.) and enter this environment. You need to install Cython manually before to make sure the following command works.

Using pip:
```bash
$ pip install Cython==0.28.2
...
$ pip install -r requirements.txt
...
```

## Example: How to run PVM for a specific valuation model provided from the Spectrum Auction Test Suite (SATS)

To demonstrate how to configure our algorithm, we include an example of the Global Synergy Value Model (GSVM) (Goeree and Holt 2010). GSVM consists of 6 regional bidders, 1 national bidder, and 18 items.

First, we set some parameters of the GSVM model. IN GSVM bidders valuations are in the interval [0,500] thus we do not need to scale these values and set scaler  = False.

```python
model_name = 'GSVM'
N = 7  # number of bidders
M = 18  # number of items
bidder_types = 2  # regional and national bidders
bidder_ids = list(range(0, 7))  # bidder ids
scaler = False  # no scaling of bidders valuations
```

Then, we define the parameters of the deep neural networks (DNNs). Recall, we use for each bidder type a distinct DNN.

```python
epochs = 300  # epochs for training the DNNs
batch_size = 32  # batch size for training the DNNs
regularization_type = 'l2'  # 'l1', 'l2' or 'l1_l2'  # regularization of the affine mappings betweenthe layer: L1, L2 or both.
# DNN for the national bidder GSVM:id=6
regularization_N = 0.00001  # regularization parameter
learning_rate_N = 0.01  # learning rate for ADAM
layer_N = [10, 10, 10]  # we define a three hiddern layer DNN with 10 hidden nodes per hidden layer
dropout_N = True  # dropout for trainnig the DNN (regularization)
dropout_prob_N = 0.05  # dropout-rate
# DNNs for the regional bidders GSVM:id=0-5
regularization_R = 0.00001
learning_rate_R = 0.01
layer_R = [16, 16]   # we define a two hiddern layer DNN with 16 hidden nodes per hidden layer
dropout_R = True
dropout_prob_R = 0.05
DNN_parameters = {}
for bidder_id in bidder_ids:
    if bidder_id == 6:
        DNN_parameters['Bidder_{}'.format(bidder_id)] = (
            regularization_N, learning_rate_N, layer_N, dropout_N,dropout_prob_N)
    else:
        DNN_parameters['Bidder_{}'.format(bidder_id)] = (
            regularization_R, learning_rate_R, layer_R, dropout_R,dropout_prob_R)
sample_weight_on = False  # no datapoint-specific weights
sample_weight_scaling = None  # no datapoint-specific weights
```

Then, we define parameters for the MIPs defined in our paper in (OP2). We can select the following three different methods (presented in increasing order of runtime) for tightening the bounds of the big-M constraints:
i.) Mip_bounds_tightening = False, no bound tightening, global L big-m constant.
ii.) Mip_bounds_tightening = 'IA', interval arithmetic (box relaxations).
ii.) Mip_bounds_tightening = 'LP', interval arithmetic (box relaxations) + linear programming relxations per node depending on all previous nodes.

```python
L = 3000  # global big-M constant
Mip_bounds_tightening = 'IA'   # Bound tightening: False ,'IA' or 'LP'
warm_start = True  # boolean, should previous solution be used as a warm start.
```

Next, we define PVM specific parameters. The number of initial bundle-value pairs sampled across bidders is defined as c_0:=caps[0].
This initial bundle-value pairs are the same for all bidders. The maximal number of possible value queries per elicitation thread is defined as c_e:=caps[1].
This results in a maxmium possible number of total value queries in PVM per bidder of (in GSVM  N:=7): c_0+N*c_e.

```python
caps = [40, 10]  # [c_0, c_e] with initial bids c0 and maximal number of value queries ce
seed_instance = 12  # seed for the auction instance generated in SATS
min_iteration = 1  # no restriction on minimum iterations per elicitation thread.
```

Finally, we run the PVM algorithm:

```python
RESULT = PVM(
    scaler=scaler, caps=caps, L=L, parameters=DNN_parameters, epochs=epochs,
    batch_size=batch_size, model_name=model_name, sample_weight_on=sample_weight_on,
    sample_weight_scaling=sample_weight_scaling, min_iteration=min_iteration,
    seed_instance=seed_instance, regularization_type=regularization_type,
    Mip_bounds_tightening=Mip_bounds_tightening, warm_start=warm_start)
```

The full example can be found [here](tests/test_pvm.py).


## Contact

Maintained by Jakob Weissteiner (weissteiner)
