// C++ headers
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

// Project headers
#include "matrix-factorizer.h"

int ParseCommandArgs(const int argc, const char** argv,
                     std::unordered_map<std::string, std::string>& arg_map) {
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "-t", strlen(argv[i]))) {
      arg_map["-t"] = argv[++i];
    } else if (!strncmp(argv[i], "-e", strlen(argv[i]))) {
      arg_map["-e"] = argv[++i];
    } else if (!strncmp(argv[i], "-c", strlen(argv[i]))) {
      arg_map["-c"] = argv[++i];
    } else if (!strncmp(argv[i], "-d", strlen(argv[i]))) {
      arg_map["-d"] = argv[++i];
    } else {
      fprintf(stderr, "Unrecognized argument '%s'\n", argv[i]);
      exit(EXIT_FAILURE);
    }
  }
  if (arg_map.find("-t") == arg_map.end()) {
    fprintf(stderr, "Argument '-t' required!\n");
    exit(EXIT_FAILURE);
  }
  return 0;
}

int MatrixFactorCpu(std::unordered_map<std::string, std::string> arg_map) {
  MatrixFactorizer mf;
  mf.Setup(arg_map);
  mf.InitializeFeatures();
  mf.Preprocess();
  mf.TrainCpu();
  mf.DumpFeatures();
  return 0;
}

int MatrixFactorGpu(std::unordered_map<std::string, std::string> arg_map) {
  MatrixFactorizer mf;
  mf.Setup(arg_map);
  mf.InitializeFeatures();
  mf.Preprocess();
  mf.AllocateGpuMemory();
  mf.CopyToGpu();
  mf.TrainGpu();
  mf.DumpFeatures();
  return 0;
}

int PrintHelp(const char* exec) {
  fprintf(stderr, "\nUsage: %s -t train-file [-e test-file] [-c config-file] "
              "[-d gpu-id]\n\n", exec);
  fprintf(stderr, "-t train-file:\n");
  fprintf(stderr, "\tTrain data file in COO format.\n");
  fprintf(stderr, "\tEach line is a triplet (user-id, item-id, rating).\n");
  fprintf(stderr, "\tThe first line is a triplet "
          "(number-of-users, number-of-items, number-of-ratings).\n");
  fprintf(stderr, "\tComment lines start with '%%'s.\n");
  fprintf(stderr, "[-e test-file]:\n");
  fprintf(stderr, "\tTest data file in COO format.\n");
  fprintf(stderr, "\tEach line is a triplet (user-id, item-id, rating).\n");
  fprintf(stderr, "\tThe first line is a triplet "
          "(number-of-users, number-of-items, number-of-ratings).\n");
  fprintf(stderr, "\tComment lines start with '%%'s.\n");
  fprintf(stderr, "[-c config-file]:\n");
  fprintf(stderr, "\tConfiguration file.\n");
  fprintf(stderr, "\tComment lines start with '#'s.\n");
  fprintf(stderr, "[-d gpu-id]:\n");
  fprintf(stderr, "\tGPU #gpu-id used to accelerate computation.\n");
  return 0;
}

int main(const int argc, const char** argv) {
  if (argc >= 3) {
    std::unordered_map<std::string, std::string> arg_map;
    ParseCommandArgs(argc, argv, arg_map);
    if (arg_map.find("-d") == arg_map.end()) {
      MatrixFactorCpu(arg_map);
    } else {
      MatrixFactorGpu(arg_map);
    }
    return 0;
  }
  PrintHelp(argv[0]);
  return 0;
}

