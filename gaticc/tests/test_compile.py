import os, sys, json, gati, argparse

ARCHS = [
    {"ramsize":512,"sa-arch":"9,4,4","vasize":32,"accbuf-size":4096,"fcbuf-size":32768,"im2colbuf-size":1024},
    {"ramsize":512,"sa-arch":"9,8,8","vasize":32,"accbuf-size":4096,"fcbuf-size":32768,"im2colbuf-size":1024},
    {"ramsize":512,"sa-arch":"16,1,16","vasize":32,"accbuf-size":4096,"fcbuf-size":32768,"im2colbuf-size":1024}
]

def run_test(models):
    if len(models) == 1 and os.path.isdir(models[0]):
      models = [os.path.join(models[0], f) for f in os.listdir(models[0])]
    gati.set_keep_quiet(True); ok, fail = [], []
    for arch in ARCHS:
        gati.set_arch(config=arch)
        for f in models:
            try: gati.compile(f,"/tmp/model.gml"); ok.append((arch,f))
            except: fail.append((arch,f))
    return {
        "passed": [{"arch":a,"model":m} for a,m in ok],
        "failed": [{"arch":a,"model":m} for a,m in fail],
        "summary": {"total":len(ok)+len(fail), "passed":len(ok), "failed":len(fail)}
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-m','--models', nargs='+', required=True)
    p.add_argument('-o','--output',default="test_compile.results.json")
    args = p.parse_args()

    res = run_test(args.models)
    with open(args.output,"w") as f:
        json.dump(res,f,indent=2)
        print(f"JSON saved to {args.output}")
