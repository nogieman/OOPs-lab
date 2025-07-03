import os, json, shutil, subprocess, argparse
from pathlib import Path; from datetime import datetime
import test_sim, test_summary, test_compile, test_dispatch, test_run

P = argparse.ArgumentParser("GATICC Test Pipeline",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
P.add_argument("-r","--gaticc-root",required=True);P.add_argument("-v","--venv-path",required=True)
P.add_argument("-a","--archive-path",required=True);P.add_argument("-b","--bitstreams_folder",required=True)
P.add_argument("-m","--models_folder",required=True);P.add_argument("-l","--hostname",nargs='+',required=True)
P.add_argument("--build",action="store_true");P.add_argument("--run-tests",action="store_true")
P.add_argument("--render-html",action="store_true"); args=P.parse_args()

R,V,A,B,M,HN=Path(args.gaticc_root),Path(args.venv_path),Path(args.archive_path),Path(args.bitstreams),Path(args.models),args.hostname
PY,VF,HF,T=V/"bin/python",R/"VERSION.txt",R/"version_history.json",R/"tests"
H=lambda:json.load(open(HF)) if HF.exists() else [];Rv=lambda:VF.read_text().strip()
run=lambda c,cwd=None: print("[CMD]",' '.join(map(str,c))) or subprocess.run(c, cwd=cwd, check=True)
venv=lambda:run(["python3","-m","venv",str(V)]) if not V.exists() else None
update=lambda v:HF.write_text(json.dumps(H()+[{"version":v,"timestamp":str(datetime.now())}],indent=2))
new=lambda v:v not in {x["version"] for x in H()}

def build(): run(["bash",str(R/"scripts/build.sh"),str(V)],cwd=R)

def run_all(v):
    res={"sim":test_sim.run_test(str(M)),"summary":test_summary.run_test(str(M)),
         "compile":test_compile.run_test(str(M)),"dispatch":{},"run":{}}
    [res["dispatch"].setdefault(b.name,{}).update({vn:(test_dispatch.run_test(str(M),"9,4,4",str(b),vn,False,False))["results"]})
     for b in sorted(B.glob("*.hex")) for vn in HN]
    [res["run"].update({vn:test_run.run_all_direct(str(M),"9,4,4",str(B),vn)}) for vn in HN]
    (d:=A/v).mkdir(parents=True,exist_ok=True); (d/"metadata.json").write_text(json.dumps({"version":v,"results":res},indent=2))

def render_html():
    VERS,H_,W=H(),lambda: "".join(f"<option value='{v['version']}'>{v['version']}</option>" for v in VERS), lambda d,p="": (
        html.append("<p><b>Summary:</b> "+", ".join(f"{k}: {v}" for k,v in d["summary"].items())+"</p>") if "summary" in d else None,
        html.append(tbl(d)) if "passed" in d or "failed" in d else (
            [html.append(f"<h3>{p}/{k}</h3>") or W(v,p+"/"+k) for k,v in d.items()] if isinstance(d,dict) else (
                html.append(tbl_list(d)) if isinstance(d,list) else html.append(f"<p style='color:red'>[WARN] Unhandled data type: {type(d)}</p>")
            )
        )
    )
    N=[v["version"] for v in VERS]; L=sorted(N,key=lambda x:list(map(int,x.split('.'))))[-1] if N else ""
    if not N: print("[WARN] No versions found!"); return
    html=[f"""<html><head><style>body{{font-family:sans-serif;font-size:x-large;background:#f2f6ff;color:#222;padding:5px 20px}}table{{border-collapse:collapse;width:71%;font-family:monospace;margin-bottom:40px}}th,td{{border:1px solid #ccc;padding:8px;text-align:left;font-size:16px}}h1{{color:#1a237e}}.pass{{background:#e1ffe1}}.fail{{background:#ffe1e1}}select{{font-size:20px;padding:6px 12px;margin:10px 0 20px 10px;border-radius:6px;border:1px solid #ccc;background:#e8ecff;color:#000}}select:focus{{box-shadow:0 0 4px #90caf9;border-color:#64b5f6}}</style>
    <script>function showVersion(v){{document.querySelectorAll('.version-table').forEach(t=>t.style.display='none');let t=document.getElementById('ver_'+v);if(t)t.style.display='block';localStorage.setItem('selected_version',v);}}window.onload=function(){{let s=localStorage.getItem('selected_version')||'{L}';document.getElementById('version_select').value=s;showVersion(s);}}</script></head><body><marquee><h2>GATI Test Results Archive</h2></marquee>
    <label for='version_select'>Select Version:</label>
    <select id='version_select' onchange='showVersion(this.value)'>{H_()}</select><hr>"""]
    def tbl(d): k={k for cat in["passed","failed"] for e in d.get(cat,[]) for k in e}; k=["model"]+sorted(k-{"model"}) if "model" in k else sorted(k); return "<table><tr>"+"".join(f"<th>{c}</th>" for c in k)+"</tr>"+"".join(f"<tr class='{'pass' if cat=='passed' else 'fail'}'>"+"".join(f"<td>{e.get(c,'-')}</td>" for c in k)+"</tr>" for cat in["passed","failed"] for e in d.get(cat,[]))+"</table>"
    def tbl_list(lst): k={k for e in lst if isinstance(e,dict) for k in e}; k=["model"]+sorted(k-{"model"}) if "model" in k else sorted(k); return "<table><tr>"+"".join(f"<th>{c}</th>" for c in k)+"</tr>"+"".join(f"<tr class='{'pass' if e.get('status','').lower()=='pass' else 'fail'}'>"+"".join(f"<td>{e.get(c,'-')}</td>" for c in k)+"</tr>" if isinstance(e,dict) else "<tr><td colspan='99' style='color:red'>BAD ITEM: not a dict!</td></tr>" for e in lst)+"</table>"
    TN_={"sim":"CPU Simulation Accuracy","summary":"Model Summary (Parser Test)","compile":"gaticc Compilation Test","dispatch":"FPGA and CPU dispatch test","run":"FPGA Run Test (Vaaman)"}
    for v in N:
        f=A/v/"metadata.json"; html.append(f"<div id='ver_{v}' class='version-table' style='display:none'><h2>Version: {v}</h2>")
        if not f.exists(): html.append(f"<p style='color:red'>[ERROR] Missing metadata.json for version {v}</p></div>"); continue
        meta=json.load(open(f)); R_=meta.get("results",{})
        for tn,d in R_.items(): html.append(f"<h2>{TN_.get(tn,tn)}</h2>"); W(d)
        html.append("</div>")
    html.append("</body></html>")
    Path("index.html").write_text("\n".join(html)); print("[INFO] HTML written to index.html")

if __name__=="__main__":
    v=Rv(); print("="*60,f"[INFO] Testing version: {v}",sep="\n")
    venv()
    if args.build: build()
    if args.run_tests:
        if not new(v): print("[INFO] Already tested. Skipping."); exit()
        run_all(v); update(v)
    if args.render_html: render_html()
    print("="*60,f"[DONE] Pipeline completed for version {v}\nTo start html server use: python -m http.server",sep="\n")
