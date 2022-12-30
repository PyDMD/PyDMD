# Generic linear algebra support
This submodule enables generic linear algebra in PyDMD, by means of a static factory which provides
a `LinalgBase`, whose interface can be found in `linalg_base.py`. PyDMD supports generic linear algebra
on all the methods exposed by `LinalgBase`, but we could add more if needed.

The static factory method is the method `build_linalg_module` in `linalg.py`. The function returns a
concrete implementation of `LinalgBase` depending on the type of the array passed as the first 
argument.

## Things to keep in mind
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

## Guidelines
### Calling `build_linalg_module()`
Be careful on the argument on which you call `build_linalg_module()`. It may happen that some NumPy arrays
are created *en passant* to be used as arguments for more complicated functions. These are not good candidates
for `build_linalg_module()`, as they clearly do not convey information about user preferences on array typing.

### Check aggressively ...
Always check that the user is providing appropriate array pairs/triplets in PyDMD entrypoints (e.g. `fit()`).
`linalg.py` provides some utility functions (`is_array(X)`, `assert_same_linalg_type(X,*args)`) to facilitate writing
this kind of checks.

### ... but trust the team
No need to check the output of internal functions like `DMDBase._optimal_dmd_matrices()`. This clutters the
code and provides no additional value, our PRs are carefully reviewed by developers from the core team of
PyDMD.