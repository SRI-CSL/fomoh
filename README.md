# Fomoh

PyTorch-based library that implements nested forward AD and interfaces with
PyTorch models.

* Link to paper coming soon!

* Perform forward-mode automatic differentiation over functions in PyTorch to get first and second order information.
* Train neural network models using forward-mode automatic differentiation.
* Includes first-order [FGD](https://arxiv.org/pdf/2202.08587) and our proposed approach of FoMoH-KD.

To run the code, from this directory location you can install the fomoh Python package:
```
pip install .
```

## Notebook Tutorials:
* For basic usage, check out the notebook: `Fomoh_Rosenblock_Example.ipynb`
* For an example training a neural network check out: `FoMoH_NeuralNetwork.ipynb`

## Blog Posts:
* They are on their way!

Examples:
* To run the Rosenbrock example from the paper, in the `./scripts/Rosenbrock` location, run:
``` python Rosenbrock_ND_dim_comparison.py --epochs 100 --dim-obj 10 --save-file ./plots/rosenbrock_comparison_plane_dim_comparison_10D.pt --newton ```
* To run a logistic regression example from the paper, e.g. FoMoH, in the `./scripts/logistic_regression` location, first make the new folder `best_results`, then run:
```./train_model.sh 0.1362 1024 FoMoH 0```
* To run a cnn example from the paper, e.g. FoMoH, in the `./scripts/cnn` location, first make the new folder `best_results_3000`, then run:
```./train_model.sh 0.544 2048 FoMoH 0```

## How to cite?

Please consider citing the following paper if you use `Fomoh` in your research:

* Link to paper coming soon!

<!-- ```
@article{cobb2024second,
  title={Second-Order Forward-Mode Automatic Differentiation for Optimization},
  author={Cobb, Adam D and Baydin, Atılım Güneş and Pearlmutter, Barak A. and Jha, Susmit},
  journal={arXiv},
  year={2024}
}
``` -->

## Acknowledgements

This material is based upon work supported by the United
States Air Force and DARPA under Contract No. FA8750-23-C-0519. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect
the views of the United States Air Force and DARPA.
