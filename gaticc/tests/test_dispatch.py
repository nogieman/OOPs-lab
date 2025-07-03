# test_dispatch.py
import numpy as np, gati, os, json, sys, argparse

PAIRS = [("flatten_60_QuantizeLinear", ["vgg0_conv12_fwd_quant"]),
         ("vgg0_pool1_fwd_QuantizeLinear", ["vgg0_conv3_fwd_quant"])]
MODELS = ['imagenet_vgg_16_224_int8.onnx']

gen = lambda: np.expand_dims(np.load("imagenet_10.npy")[4],0)

def simulate(onnx, cpu): 
    print(f"[SIM] {cpu} on {onnx}"); gati.set_dispatch([cpu]); gati.sim(onnx,gen())

def runfpga(onnx, gml, fpga, arch): 
    print(f"[FPGA] {fpga} arch {arch}"); gati.set_dispatch(fpga); gati.set_arch(config={"sa-arch":arch})
    gati.compile(onnx,gml); gati.run(onnx,gml,gen())

def compare(fpga_npy, cpu_npy, verbose=False):
    if not os.path.exists(fpga_npy) or not os.path.exists(cpu_npy): return -1,"ERROR"
    a,b = np.load(fpga_npy).flatten().astype(np.int8), np.load(cpu_npy).flatten().astype(np.int8)
    if a.shape!=b.shape: return -1,"ERROR"
    d=np.abs(a-b); acc=100*np.sum(d<=3)/len(d); 
    if verbose and (bad:=np.sum(d>3))>0:
        print("Idx\tFPGA\tCPU\tDiff")
        [print(f"{i}\t{a[i]}\t{b[i]}\t{d[i]}") for i in np.where(d>3)[0]]
    return acc,("PASS" if acc>=98 else "FAIL")

def run_test(models_dir, arch, bitstream, hostname, verbose=False, keep=False):
    gml="model.gml"; gati.set_keep_quiet(True); gati.set_remote(hostname); gati.flash(bitstream)
    res={"bitstream":bitstream,"vaaman":hostname,"results":[]}
    for m in [f for f in os.listdir(models_dir) if f in MODELS]:
        onnx=os.path.join(models_dir,m)
        [simulate(onnx,cpu) for cpu,_ in PAIRS]
        [runfpga(onnx,gml,fpga,arch) for _,fpga in PAIRS]
        for cpu,fpga in PAIRS:
            cpu_f=f"{cpu}.tensor.npy"; fpga_f=f"fpga_{fpga[0]}.npy"
            acc,status = compare(fpga_f,cpu_f,verbose) if os.path.exists(cpu_f) and os.path.exists(fpga_f) else (-1,"ERROR")
            res["results"].append({"model":m,"cpu_layer":cpu,"fpga_layer":fpga[0],"match":f"{acc:.2f}%" if acc>=0 else "-","status":status})
            if not keep: [os.remove(f) for f in [cpu_f,fpga_f] if os.path.exists(f)]
    return res

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument('-m','--models',nargs='+',required=True)
    p.add_argument('-a','--arch',default='9,4,4'); p.add_argument('-b','--bitstream',required=True)
    p.add_argument('-l','--hostname',required=True); p.add_argument('-v','--verbose',action='store_true')
    p.add_argument('-k','--keep',action='store_true'); args=p.parse_args()
    d=args.models[0] if len(args.models)==1 and os.path.isdir(args.models[0]) else sys.exit("[ERROR] Provide models dir")
    r=run_test(d,args.arch,args.bitstream,args.hostname,args.verbose,args.keep)
    with open("test_dispatch.results.json","w") as f: json.dump(r,f,indent=2)
    print("[INFO] Dispatch test done.")
