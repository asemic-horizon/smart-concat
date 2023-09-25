# smart-concat

This is a nonrandom, content-aware column tagger to cleverly concatenate numpy arrays into pd.DataFrames.

Unless you use sophisticated data pipelines from the beginning of projects or have the masochistic bent to use scikit-learn's `ColumnTransformer` in an iterative manner, you'll be finding at the rough exploratory stage that many common data transformations will eat your smartly-indexed and maybe metadata-annotated and give back a plain numpy ndarray. This became a source of pain this week, so I developed this: a smart way of absorbing ndarrays into pd.DataFrame, complete with a prefix assignment system that's locality sensitive for small changes.

## Usage

An example:

```python
# just some data
x = pd.DataFrame(np.arange(21).reshape(7,3), columns=["A","B","C"])

# some more data, but in numpy.ndarray format -- no column names.
y = np.random.normal(size=(x.shape[0],28))

# and wait, some corruption too!
noise = np.random.uniform(-0.1, 0.1, size = y.shape )

u = smart_concat(x, y)
v = smart_concat(x, y+noise)

assert set(u.columns) == set(v.columns) # yay locality awareness!
```

Note that we don't use any special locality sensitive hashing schemes; `smart_operations.py` has a docstring with more details. The code is not too obscure either. 

## Installation

The only external dependencies here are numpy and pandas. You should try numpy and pandas sometime if you haven't already.

Either download the `smart_operations.py` file from this repo or copy-paste its contents directly into your editor. The code has a standard permissive license detailed in the `LICENSE`. 
