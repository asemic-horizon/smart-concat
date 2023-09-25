# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def smart_prefix(arr, existing_prefixes, length=4, preferred_prefix=None):
    """Non-random, content-aware, reproducible tagger for arrays.
       Meant to introduce a naming scheme for arrays to be transformed
       in pd.DataFrames.


    Args:
        arr: ndarray
        existing_prefixes: Set of existing prefixes
        preferred_prefix (optional): Suggested prefix.

    Returns:
        prefix: str  


    --- 
    Notes:

    - We use the SHA256 hash function to create a reproducible fingerprint
      from general statistical properties of the array (shape and quintiles)
      after rescaling and rounding (to improve locality sensitivity.), e.g. 

     >>     x = pd.DataFrame(np.arange(21).reshape(7,3), columns=["A","B","C"])
     >>    u = smart_concat(x, x.values)
     >>    v = smart_concat(x, np.random.uniform(-0.1,0.1,size=x.shape)+x.values)
     >>    assert set(u.columns) == set(v.columns)
    
     The goal of this is to reduce the chances of names changing due to errors
     introduced by numerical transformations.
     
    """

    arr = (arr/arr.std()).round(1)
    stats =  partial(np.quantile, q=[.2, .8])
    fingerprint = str(arr.shape) +''.join(map(str, stats(arr)) )

    valid_chars =  partial(list.__contains__, list(map(chr,range(65,91)))+[':','&'])
    prefix = preferred_prefix
    while prefix is None or prefix in existing_prefixes: # almost never runs more than once
        hasher = hashlib.sha256()
        hasher.update((fingerprint + str(prefix)).encode('utf-8'))
        base64_hash = base64.b64encode(hasher.digest()).decode('utf-8')
        prefix = ''.join(filter(valid_chars, base64_hash))
    logger.info(f"Content fingerprint: {fingerprint}; prefix suggestion: {prefix}")     
    prefix = prefix[:length] + "_"
    return prefix

def smart_concat(old_df, new_arr):
    """Concatenates a new array to a dataframe while introducing a smart prefix.

    Args:
        old_df: pd.DataFrame
        new_arr: ndarray
    
    Returns:
        combined_df: pd.DataFrame
    """

    existing_prefixes = set(col.split('_')[0] for col in old_df.columns)
    new_prefix = smart_prefix(new_arr, existing_prefixes)
    
    new_df = pd.DataFrame(new_arr).add_prefix(new_prefix)
    combined_df = pd.concat([old_df, new_df], axis=1)
    return combined_df
