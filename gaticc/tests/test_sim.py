# test_sim.py
import os, sys, json, argparse, numpy as np, gati

FILES = [
  'cifar10_vgg11.onnx','cifar10_vgg16.onnx','cifar10_vgg19.onnx',
  'imagenet_mobilenetv2-int8-symmetric.onnx','imagenet_resnet50-int8-symmetric.onnx',
  'imagenet_vgg_16_224_int8.onnx','imagenet_vgg_16_224_uint8.onnx',
  'mnist_6_28_int8.onnx','mnist_average_pool_int8.onnx','mnist_depthwise_60acc.onnx',
  'mnist_depthwise_63acc.onnx','mnist_global_average_pool_int8.onnx','mnist_int8_2x2.onnx',
  'mnist_int8_k1x11.onnx','mnist_int8_k1x7.onnx','mnist_int8_maxpool_k3_s3.onnx',
  'mnist_int8_pad2.onnx','mnist_int8_stride2.onnx','mnist_int8_stride2_pad0.onnx',
  'mnist_int8_stride2_pad2.onnx','mnist_int8_stride3.onnx','mnistpad1_6_28_int8.onnx',
  'mnist_qlinearadd2.onnx','mnist_qlinearadd.onnx','mnist_uint8.onnx',
  'mnist_uint8_pad_0.onnx','mnist_uint8_tiny.onnx'
]

LABELS = {
  "mnist": ('mnist_10_labels.txt','mnist_10.npy'),
  "imagenet": ('imagenet_10_labels.txt','imagenet_10.npy'),
  "cifar": ('cifar_10_labels.txt','cifar_10.npy')
}

post = lambda arr: np.argmax(np.squeeze(np.stack([x[1] for x in arr]),1),-1)

def run_test(models_dir):
  if not os.path.isdir(models_dir): raise ValueError(f"Invalid models dir: {models_dir}")
  gati.set_keep_quiet(True); ok, fail = [], []
  for f in FILES:
    path, acc = os.path.join(models_dir,f), 0
    print(f"Sim {f}")
    try:
      key = next(k for k in LABELS if k in f)
      lbl, data = LABELS[key]
      acc = gati.match(lbl, post(gati.sim(path, np.load(data))))
    except: pass
    (ok if acc >= 50 else fail).append({"model":f,"accuracy":acc})
  return {
    "passed": ok, "failed": fail,
    "summary": {"total": len(ok)+len(fail), "passed": len(ok), "failed": len(fail)}
  }

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('-m','--models',required=True)
  p.add_argument('-o','--output',default="test_sim.results.json")
  args = p.parse_args()
  with open(args.output,"w") as f:
    json.dump(run_test(args.models),f,indent=2)
    print(f"Saved results to {args.output}")
