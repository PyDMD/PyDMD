# Generic linear algebra support
This submodule enables generic linear algebra in PyDMD, by means of a static factory which provides
a `LinalgBase`, whose interface can be found in `linalg_base.py`. PyDMD supports generic linear algebra
on all the methods exposed by `LinalgBase`, but we could add more if needed.

The static factory method is the method `build_linalg_module` in `linalg.py`. The function returns a
concrete implementation of `LinalgBase` depending on the type of the array passed as the first 
argument.

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