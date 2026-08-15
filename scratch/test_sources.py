from datasets import load_dataset
import time

test_specs = [
    {"name": "FineWeb-Edu (General)", "path": "HuggingFaceTB/smollm-corpus", "name_kw": "fineweb-edu-dedup"},
    {"name": "OpenHermes (Q&A)", "path": "teknium/openhermes", "name_kw": None},
    {"name": "Cosmopedia-v2 (Q&A/Gen)", "path": "HuggingFaceTB/cosmopedia-v2", "name_kw": None},
    {"name": "SimpleMath (Math)", "path": "ProCreations/SimpleMath", "name_kw": None},
    {"name": "GSM8K (Math)", "path": "gsm8k", "name_kw": "main"},
    {"name": "Python Code (Code)", "path": "flytech/python-codes-25k", "name_kw": None},
]

for spec in test_specs:
    print(f"Testing {spec['name']} ({spec['path']})...")
    t0 = time.time()
    try:
        kwargs = {"split": "train", "streaming": True}
        if spec["name_kw"]:
            kwargs["name"] = spec["name_kw"]
        ds = load_dataset(spec["path"], **kwargs)
        doc = next(iter(ds))
        keys = list(doc.keys())
        print(f"  SUCCESS ({time.time()-t0:.2f}s) - Keys: {keys}")
    except Exception as e:
        print(f"  FAILED ({time.time()-t0:.2f}s) - Error: {e}")
