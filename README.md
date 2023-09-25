# smart-concat

Unless you use sophisticated data pipelines from the beginning of projects or have the masochistic bent to use scikit-learn's `ColumnTransformer` in an iterative manner, you'll be finding at the rough exploratory stage that many common data transformations will eat your smartly-indexed and maybe metadata-annotated and give back a plain numpy ndarray. This became a source of pain this week, so I developed this: a smart way of absorbing ndarrays into pd.DataFrame, complete with a prefix assignment system that's locality sensitive for small changes.

An example:

```python
x = pd.DataFrame(np.arange(21).reshape(7,3), columns=["A","B","C"])
y = np.random.normal(size=(x.shape[0],28))
noise = np.random.uniform(-0.1, 0.1, size = y.shape )
u = smart_concat(x, y)
v = smart_concat(x, y+noise)
assert set(u.columns) == set(v.columns)
```

## Installation

Either download the `smart_operations.py` file from this repo or copy-paste its contents directly into your editor. The code has a standard permissive license detailed in the `LICENSE`. 
