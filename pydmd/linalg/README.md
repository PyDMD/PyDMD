# Generic linear algebra support

This submodule enables generic linear algebra in PyDMD, by means of a static factory which provides
a `LinalgBase`, whose interface can be found in `linalg_base.py`. PyDMD supports generic linear algebra
on all the methods exposed by `LinalgBase`, but we could add more if needed.

The static factory method is the method `build_linalg_module` in `linalg.py`. The function returns a
concrete implementation of `LinalgBase` depending on the type of the array passed as the first 
argument.

### Supported DMD variants

- CDMD
- DMD
- DMDc
- FbDMD
- HankelDMD
- HODMD
- MrDMD
- RDMD
- SubspaceDMD

### TODO
- HAVOK
- OptDMD
- SPDMD

Some more refined DMD variants (e.g. `MrDMD`, `HODMD`) do not support backpropagation. This is caused
by the dependency of the endpoints (e.g. `dmd.reconstructed_data`, `dmd.amplitudes`) on the imaginary
part of the DMD eigenvalues. This dependency vanishes in some DMD variants, however numerical manipulations
propagate and amplify the dependency such that the underlying backend is not able to compute the gradient
with an acceptable degree of accuracy. Future versions of PyTorch might fix this problem.

Note that even though those variants block backpropagation, they still benefit from the computational
boost given by the newly supported backends.

## Tensorized/batched DMD

`pydmd.linalg` enables batched (or tensorized) DMD training. Like all PyTorch functions, supported DMD
variants now can be trained with tensors of size `(*, M, N)`, where `*` is called *batch dimension*.
In order to enable batched training, you need to supply the additional parameter `batch=True` to `fit()`
(by default it is `false`).

The following snippets are equivalent:
```python
>>> X.shape()
(10, 100, 20)
>>> Y = dmd.fit(X, batch=True).reconstructed_data
```

```python
>>> Y = torch.stack([dmd.fit(Xi).reconstructed_data for Xi in X])
```

The benefit of tensorized training are:
- Performance boost;
- The DMD instance retains information on all the slices of `X`;
- Coinciseness.

## Developers guide

### Things to keep in mind

Due to the strong requirements of `torch.mul` and `torch.linalg.multi_dot`, the implementation of these
two functions in `pytorch_linalg.py` forces a cast to the biggest **complex** type found in the argumnets.
We decided to take this path instead of placing the burden on user/implementors since for some algorithms
it's hard to control consistently whether the output is complex or real (e.g. `torch.linalg.eig`) and casts
will happen internally quite often. This damages memory efficiency and performance, but ensures correct 
results. It will be subject of investigation if we receive complains from our users.

This kind of casts is logged, in order to get the logs enable the `INFO` logging level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Guidelines

**Calling `build_linalg_module()`**
Be careful on the argument on which you call `build_linalg_module()`. It may happen that some NumPy arrays
are created *en passant* to be used as arguments for more complicated functions. These are not good candidates
for `build_linalg_module()`, as they clearly do not convey information about user preferences on array typing.

**Check aggressively**
Always check that the user is providing appropriate array pairs/triplets in PyDMD entrypoints (e.g. `fit()`).
`linalg.py` provides some utility functions (`is_array(X)`, `assert_same_linalg_type(X,*args)`) to facilitate writing
this kind of checks.

**... but trust the team**
No need to check the output of internal functions like `DMDBase._optimal_dmd_matrices()`. This clutters the
code and provides no additional value, our PRs are carefully reviewed by developers from the core team of
PyDMD.
