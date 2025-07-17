// Automatically configured by CMake
// Author: Federico Sossai (fsossai), 2021

//#include <benchmark/benchmark.h>
//#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>
#include "TRandom.h"
#include "check_mem.h"

using namespace std;

bool testOutput = true;

//static void BM_ONNXRuntime_Inference(benchmark::State& state, string model_path)
static void BM_ONNXRuntime_Inference(int nevts , string model_path, int ne , int nh)
{
   check_mem("initial");
   Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark");

   Ort::SessionOptions session_options;
   session_options.SetIntraOpNumThreads(1);
   session_options.SetInterOpNumThreads(1);
   // ONNXRuntime optimization:
   // possible vlaues are:
   // GraphOptimizationLevel::ORT_DISABLE_ALL -> Disables all optimizations
   // GraphOptimizationLevel::ORT_ENABLE_BASIC -> Enables basic optimizations
   // GraphOptimizationLevel::ORT_ENABLE_EXTENDED -> Enables basic and extended optimizations
   // GraphOptimizationLevel::ORT_ENABLE_ALL -> Enables all available optimizations including layout
   
   //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
   session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

   //std::cout << "benchmarking model " << model_path << std::endl;
   Ort::Session session(env, model_path.c_str(), session_options);

   int nin = session.GetInputCount();
   int nout = session.GetOutputCount();

   vector<const char*> input_node_names(nin);
   vector<const char*> output_node_names(nout);
   vector<std::string> inputStrings(nin);
   vector<std::string> outputStrings(nout);

   Ort::AllocatorWithDefaultOptions allocator;

   std::cout << "ninputs: " << nin << "  " << nout << std::endl;


   for (int i = 0; i < nin; i++) {
#if ORT_API_VERSION > 12
      inputStrings[i] = session.GetInputNameAllocated(i, allocator).get();
#else
      inputStrings[i] = session.GetInputName(i, allocator);
#endif
      input_node_names[i] = inputStrings[i].c_str();
   }
   for (int i = 0; i < nout; i++) {
#if ORT_API_VERSION > 12
     outputStrings[i] = session.GetOutputNameAllocated(i, allocator).get();
#else
     outputStrings[i] = session.GetOutputName(i, allocator);
#endif
      output_node_names[i] = outputStrings[i].c_str();
   }
   // Getting the shapes
   vector<vector<int64_t>> input_node_dims(nin);
   vector<vector<int64_t>> output_node_dims(nout);

   for (int i = 0; i < nin; i++)
      input_node_dims[i] = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
   for (int i = 0; i < nout; i++)
      output_node_dims[i] = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();


   // fix negative shapes
   for (int i = 0; i < nin; i++) {
      std::cout << "input " << i << " : {";
      for (int j = 0; j < input_node_dims[i].size(); j++) {
         std::cout << input_node_dims[i][j] << "  ";
         if (input_node_dims[i][j] < 0) input_node_dims[i][j] = - input_node_dims[i][j];
      }
      std::cout << "}" << std::endl;
   }

   // Calculating the dimension of the input tensor

   int bsize = 1;//   input_node_dims[0][0]; // assume this
   //std::cout << "Using bsize = " << bsize << std::endl;
   int nbatches = nevts / bsize;

   // int n = 1;
   // int n_sv = 10;
   // int n_pf = 100;


   // for (int i = 0; i < nin; i++) {
   //    input_node_dims[i][0] = n;
   // }
   input_node_dims[0][0] = nh;
   input_node_dims[1][1] = ne;
   input_node_dims[2][0] = ne;


   std::vector<float> x(nh*12);
   std::vector<int64_t> eidx(ne*2);
   std::vector<float> ea(ne*6);

   //std::vector<float> pff(n*16*n_pf);

   //std::vector<std::vector<float>> inputData(nin);
   std::vector<size_t> inputSizes(nin);

   auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
   // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
   //    input_tensor_values.data(), input_tensor_size,
   //    input_node_dims.data(), input_node_dims.size());

   // Running the model
   float *floatarr = nullptr;

   std::vector<Ort::Value> input_tensors;

   size_t osize = 1;
   for (int d : output_node_dims[0])
   {
      if (d > 0)
         osize *= d; // first dim(batch size) can be -1
   }
   std::vector<float> yOut(osize);

   std::vector<size_t> input_offset(nin);

   // input sizes
   for (int i = 0; i < nin; i++) {
      size_t input_tensor_size = accumulate(input_node_dims[i].begin(), input_node_dims[i].end(), 1, multiplies<int>());
      inputSizes[i] = input_tensor_size;
   }

   // loop on events
   check_mem("before_looping");
   auto t1 = std::chrono::high_resolution_clock::now();
   double totDuration = 0;
   int ntimes = 0;


   int indebug = nevts/10;
   gRandom->SetSeed(111);
   for (int ievt = 0; ievt < nevts; ievt += bsize) {

      // generate inputs
       std::generate(x.begin(), x.end(), []{return gRandom->Gaus(0,3);});
      std::generate(eidx.begin(), eidx.end(), [&]{return gRandom->Integer(nh);});
      std::generate(ea.begin(), ea.end(), []{return gRandom->Gaus(0,5);});

      //inputData = { x, eidx, ea};



         /**
         auto &input_tensor_values = inputData[i];
         input_tensor_values.resize(input_tensor_size * nbatches);
         // std::cout << "input tensor size " << input_tensor_size << "  " << input_tensor_values.size() << std::endl;

         // Input tensor initialization

         if (testOutput)
            fill_n(input_tensor_values.begin(), input_tensor_values.size(), float(i) + 1.);
         else
         {
            static std::uniform_real_distribution<float> distribution(-1, 1);
            static std::default_random_engine generator;
            std::generate(input_tensor_values.begin(), input_tensor_values.end(), []()
                          { return distribution(generator); });
         }
         */



      // for (auto _ : state) {



      // if (input_offset > input_tensor_values.size()) {
      //    std::cout << "Error in input size " << i << "  " << nevts << "  " << model_path << std::endl;
      //    throw std::runtime_error("Bad input size ");
      // }

      for (int k = 0; k < nin; k++)
      {
         if (k == 0) 
            input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, x.data() /*+ input_offset[k] */,
                                                                    inputSizes[k], input_node_dims[k].data(), input_node_dims[k].size()));
         if (k == 1)
            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, eidx.data() /*+ input_offset[k] */,
                                                                    inputSizes[k], input_node_dims[k].data(), input_node_dims[k].size()));
         if (k == 2) 
            input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, ea.data() /*+ input_offset[k] */,
                                                                    inputSizes[k], input_node_dims[k].data(), input_node_dims[k].size()));
      }
      auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), nin,
                                        output_node_names.data(), nout);

      for (int k = 0; k < nout; k++) {
         float * floatarr = output_tensors[k].GetTensorMutableData<float>();
         if (ievt % indebug == 0) {
             std::cout << "input for ievt = " << ievt << " : ";
             std::cout << x[0] << "...,   " << eidx[0] << " .., " << ea[0] << " .... " << std::endl;
            std::cout << ievt << " output " << k << " : ";
            for (int j = 0; j < osize; j++)
               std::cout << floatarr[j] << "  ";
            std::cout << std::endl;
            check_mem("at current event");
         }
      }

      // for (int k = 0; k < nin; k++)
      // {
      //    input_offset[k] += inputSizes[k];
      // }
      // if (testOutput && i == 0)
      //    std::copy(floatarr, floatarr + osize, yOut.begin());
      input_tensors.clear();
   }
   check_mem("at the end");
   auto t2 = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
   totDuration += duration / 1.E6; // in seconds
   ntimes++;
   if (testOutput)
   {
      std::string filename = model_path + ".ort.out";
      // std::cout << "writing file" << filename << std::endl;
      ofstream f;
      f.open(filename);
      f << yOut.size();
      for (size_t i = 0; i < yOut.size(); i++)
      {
         if ((i % 10) == 0)
            f << "\n"; // add endline every 10
         f << yOut[i] << "  ";
      }
      f << std::endl;
      f.close();
   }
   std::cout << "total duration " << totDuration << std::endl;
   //}
   // for (int i = 0; i < 10; i++)
   //  printf("%f\t", i, floatarr[i]);
   // state.counters["time/evt(ms)"] = totDuration / double(ntimes * nevts);
}

// define main to pass some convenient command line parameters
int main(int argc, char **argv) {

   // Parse command line arguments
   //int ne = 300000;
  int ne = 300000;
  //int n_sv = 10;
  //int nh = 100000;
  int nh = 100000;
  int nevts = 10;
   for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "-v") {
         //std::cout << "---running in verbose mode" << std::endl;
         //verbose = true;
      } else if ((arg == "-d" || arg == "--dir") && argc > i+1) {
         std::string pathDir = argv[i+1];
         std::filesystem::path path(pathDir);
         std::filesystem::current_path(path);
         i++;
      } else {
	ne = std::atoi(argv[i]);
	nh = std::atoi(argv[i+1]);
	nevts = std::atoi(argv[i+2]);
	break;
      }
   }

   std::cout << "running benchmark from current directory " << std::filesystem::current_path()  << std::endl;
   std::cout << "using " << ne << "  " << nh << "   " << nevts << std::endl; 

//   ::benchmark::Initialize(&argc, argv);
//   ::benchmark::RunSpecifiedBenchmarks();

   //BM_ONNXRuntime_Inference(nevts, "atlas/tracking_gnn_fromBenjamin.onnx", ne, nh);
   BM_ONNXRuntime_Inference(nevts, "atlas/gnn_large.onnx", ne, nh);

   return 0;
}
