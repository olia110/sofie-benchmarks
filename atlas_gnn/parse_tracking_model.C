#include <string>
#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/RModel.hxx"
#include <iostream>
// number of edges must be <= number of hits
void Parse(const std::string & filename, int nedges = 0, int nhits = 0) {


   TMVA::Experimental::SOFIE::RModelParser_ONNX p;
   auto m = p.Parse(filename,true);
   // do static parsing
   if (nedges > 0 && nhits > 0) { 
      std::map<std::string, size_t> mp;
      mp["num_edges"]=nedges;
      mp["num_spacepoints"] = nhits;

     //m.PrintInitializedTensors(); 
      m.Initialize(mp, true);
   }

   m.Generate(TMVA::Experimental::SOFIE::Options::kProfile, -1, 0, true);

   m.OutputGenerated();
   m.PrintGenerated();

}

int parse_tracking_model(int argc = 0, char **argv = nullptr) {


   std::string filename = "gnn_small.onnx";
   int ne = 300000;
   int nh = 100000;
   if (argc > 3) {
      ne = std::atoi(argv[1]);
      nh = std::atoi(argv[2]);
   }


   std::cout << "parsing model from " << filename << " with ne = " << ne << " nh = " << nh << std::endl;
   
   Parse(filename, ne, nh);
   return 0;
}

int main(int argc, char **argv) {

   return parse_tracking_model(argc,argv);

}
