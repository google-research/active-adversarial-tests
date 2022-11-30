# Increasing Confidence in Adversarial Robustness Evaluations

This is the official repository of the paper _Increasing Confidence in Adversarial Robustness Evaluations_
by Zimmermann et al. 2022.

The reference implementation of our proposed active test is in
[active_tests/decision_boundary_binarization.py](active_tests/decision_boundary_binarization.py),
and the code to reproduce our experimental findings is in [case_studies](case_studies). Note, that when evaluating
the defense of our authors we always used their reference implementation and only performed the _minimal_ modification
to integrate our test in their respective code base.
##
![](https://zimmerrol.github.io/active-tests/img/Figure_1.svg)

## Citing
If you use this library, you can cite our [paper](https://openreview.net/forum?id=NkK4i91VWp).
Here is an example BibTeX entry:

```bibtex
@inproceedings{zimmermann2022increasing,
    title={Increasing Confidence in Adversarial Robustness Evaluations},
    author={Roland S. Zimmermann and Wieland Brendel and Florian Tramer and Nicholas Carlini},
    booktitle={Advances in Neural Information Processing Systems},
    editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
    year={2022},
    url={https://openreview.net/forum?id=NkK4i91VWp}
}
```

_Disclaimer: This is not an official Google product._