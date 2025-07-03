#include "options.h"
#include "executor.h"
#include "onnx_parser.h"
#include "rt.h"
#include "utils.h"
#include "optimization.h"
#include "instgen.h"

void dispatch_timeest(const Op::Parser &parser) {
  Op::Graph graph = parser.get_graph();
  time_estimate(graph);
}

void dispatch_info_ops() {
  std::string s = gbl_args["info"].as<std::string>();
  Op::Parser parser(s);
  if (gbl_args.has_option("summary")) {
    parser.bare_summary();
  }

  if (gbl_args.has_option("timeest")) {
    dispatch_timeest(parser);
  }
}

void dispatch_compile_ops() {
  std::string s = gbl_args["compile"].as<std::string>();
  Op::Parser parser(s);
  split_large_kernel(parser.get_graph());
  Pass::absorb(parser.get_graph());
  GmlGen gmlgen(GATI_INST_ORG);
  BinBlob binblob{gmlgen.generate_gml(parser)};

  if (gbl_args.has_option("output")) {
    auto filename = gbl_args["output"].as<std::string>();
    binblob.write(filename);
  }

  if (gbl_args.has_option("pretty-print-blob")) {
    binblob.pretty_print();
  }
}

void dispatch_sim_ops() {
  log_fatal("command line driver disabled, use the python interface\n");
  //Executor executor;
  //TensorPool ret = executor.run(onnx_path, arr);
}

void dispatch_run_ops() {
  log_fatal("command line driver disabled, use the python interface\n");
}

int dispatch() {
  if (gbl_args.has_option("help")) {
    gbl_args.print_usage();
    return 0;
  } else if (gbl_args.has_option("version")) {
    gbl_args.print_version();
    return 0;
  } else if (gbl_args.has_option("info")) {
    dispatch_info_ops();
  } else if (gbl_args.has_option("compile")) {
    dispatch_compile_ops();
  } else if (gbl_args.has_option("sim")) {
    dispatch_sim_ops();
  } else if (gbl_args.has_option("run")) {
    dispatch_run_ops();
  } else {
    log_fatal("Don't know what to do. See gaticc -h\n");
  }
  return 0;
}
