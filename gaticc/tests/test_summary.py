import os, sys, json, gati, argparse

def run_test(models_dir):
    if not os.path.isdir(models_dir): raise ValueError(f"Invalid models dir: {models_dir}")
    gati.set_keep_quiet(True); ok, fail = [], []
    for f in os.listdir(models_dir):
        try: gati.summary(os.path.join(models_dir,f)); ok.append(f)
        except: fail.append(f)
    return {
        "passed": [{"model":f} for f in ok],
        "failed": [{"model":f} for f in fail],
        "summary": {"total":len(ok)+len(fail), "passed":len(ok), "failed":len(fail)}
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-m','--models',required=True)
    p.add_argument('-o','--output',default="test_summary.results.json")
    args = p.parse_args()

    res = run_test(args.models)
    with open(args.output,"w") as f:
        json.dump(res,f,indent=2)
        print(f"JSON saved to {args.output}")