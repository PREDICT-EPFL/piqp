---
title: Citing PIQP
layout: default
nav_order: 6
---

If you found PIQP useful in your scientific work, we encourage you to cite our main paper:

{% raw %}
```
@INPROCEEDINGS{schwan2023piqp,
  author={Schwan, Roland and Jiang, Yuning and Kuhn, Daniel and Jones, Colin N.},
  booktitle={2023 62nd IEEE Conference on Decision and Control (CDC)}, 
  title={{PIQP}: A Proximal Interior-Point Quadratic Programming Solver}, 
  year={2023},
  volume={},
  number={},
  pages={1088-1093},
  doi={10.1109/CDC49753.2023.10383915}
}
```
{% endraw %}

In case you are specifically using the `sparse_multistage` KKT solver backend, we encourage you to cite the specific paper:

{% raw %}
```
@misc{schwan2025piqp_multistage,
  author={Schwan, Roland and Kuhn, Daniel and Jones, Colin N.},
  title={Exploiting Multistage Optimization Structure in Proximal Solvers}, 
  year={2025},
  eprint = {arXiv:2503.12664}
}
```
{% endraw %}
