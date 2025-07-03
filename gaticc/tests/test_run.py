import os, sys, json, subprocess, argparse

mut = [
  'mnist_6_28_int8.onnx', 'mnist_int8_pad2.onnx', 'mnist_int8_stride2.onnx',
  'mnist_int8_stride3.onnx', 'mnistpad1_6_28_int8.onnx', 'imagenet_vgg_16_224_int8.onnx',
  'cifar10_vgg11.onnx', 'cifar10_vgg16.onnx', 'cifar10_vgg19.onnx',
  'mnist_int8_k1x7_nomaxpool.onnx', 'mnist_int8_stride2_pad0.onnx',
  'mnist_int8_stride2_pad2.onnx', 'mnist_average_pool_int8.onnx', 'mnist_int8_pad4343.onnx'
]

def run_all(args):
    run_script=os.path.abspath(__file__)
    mdir=args.models[0] if os.path.isdir(args.models[0]) else os.path.dirname(args.models[0]) or '.'
    models=[f for f in os.listdir(mdir) if f in mut] if os.path.isdir(args.models[0]) else [os.path.basename(f) for f in args.models]
    failed,matches=[],[]
    for b in [os.path.join(args.bitstreams,f) for f in os.listdir(args.bitstreams) if f.endswith(".hex")]:
        print(f"\n=== Bitstream: {b} on {args.hostname} ===\n")
        for m in models:
            print(f"Running: {os.path.join(mdir,m)}")
            res_path=f"/tmp/{m}_{os.path.basename(b)}.res.json"
            try:
                subprocess.run([sys.executable,run_script,'--run-one','-m',os.path.join(mdir,m),'-a',args.arch,'-b',b,'-l',args.hostname,'--out',res_path],timeout=20,check=True)
                out=json.load(open(res_path))
                (matches if out.get("status")=="ok" else failed).append({"model":m,"bitstream":os.path.basename(b),"match":out.get("match",0),"error":out.get("error","unknown") if out.get("status")!="ok" else None})
            except subprocess.TimeoutExpired: print(f"Timeout: {m}"); failed.append({"model":m,"bitstream":os.path.basename(b),"match":0,"error":"timeout"})
            except KeyboardInterrupt: print(f"Skipped: {m}"); failed.append({"model":m,"bitstream":os.path.basename(b),"match":0,"error":"keyboard"})
            except Exception as e: print(f"Exception: {m}: {e}"); failed.append({"model":m,"bitstream":os.path.basename(b),"match":0,"error":str(e)})
    json.dump({"passed":matches,"failed":failed,"summary":{"passed":len(matches),"failed":len(failed),"total":len(matches)+len(failed)}},open("test_run.results.json","w"),indent=2)

def run_one(args):
    import numpy as np, gati
    gati.set_keep_quiet(True); gati.set_arch(config={"sa-arch":args.arch})
    if args.hostname!='localhost': gati.set_remote(args.hostname)
    onnx,gml,res=args.models[0],"/tmp/model.gml",{"status":"fail"}
    try:
        if gati.compile(onnx,gml)!=0: raise Exception("compile failed")
        gati.flash(args.bitstreams)
        data,lbl=(np.load("mnist_10.npy"),"mnist_10_labels.txt") if "mnist" in onnx else (np.load("imagenet_10.npy"),"imagenet_10_labels.txt") if "imagenet" in onnx else (np.load("cifar_10.npy"),"cifar_10_labels.txt") if "cifar" in onnx else (None,None)
        if data is None: raise Exception("unknown dataset")
        out=np.argmax(np.squeeze(np.stack([i[1] for i in gati.run(onnx,gml,data)]),1),-1)
        res={"status":"ok","match":gati.match(lbl,out)}
    except Exception as e: res["error"]=str(e)
    if args.out: json.dump(res,open(args.out,"w"))

def run_all_direct(models_dir,arch,bitstreams,hostname):
    args=argparse.Namespace(models=[models_dir],arch=arch,bitstreams=bitstreams,hostname=hostname,run_one=False,out=None)
    run_all(args); return json.load(open("test_run.results.json"))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument('-m','--models',nargs='+',required=True)
    p.add_argument('-a','--arch',default='9,4,4'); p.add_argument('-b','--bitstreams',required=True)
    p.add_argument('-l','--hostname',required=True); p.add_argument('--run-one',action='store_true'); p.add_argument('--out')
    args=p.parse_args(); (run_one if args.run_one else run_all)(args)
