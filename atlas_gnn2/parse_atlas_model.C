#include <string>
#include "TMVA/RModelParser_ONNX.hxx"
#include "TMVA/RModel.hxx"
#include <iostream>

void Parse(const std::string & filename) {

   int ntracks = 100;
   int nflows = 3; // for gn3
   TMVA::Experimental::SOFIE::RModelParser_ONNX p;
   auto m = p.Parse(filename,true);
   std::map<std::string, size_t> mp;
   mp["n_tracks"]=ntracks;
   // case of gn3 we need also n_flow
   if (filename.find("gn3") != std::string::npos)
      mp["n_flow"]=nflows;

   m.PrintInitializedTensors();
   m.Initialize(mp, true);

// changed from kDefaulf to kProfile 
   m.Generate(TMVA::Experimental::SOFIE::Options::kProfile, -1, 0, true);

   m.OutputGenerated();
   m.PrintGenerated();

}

int parse_atlas_model() {
   Parse("network_gn2.onnx");
}

int main(int argc, char **argv) {

   std::string filename = "network_gn2.onnx";
//   if (argc > 1) {
//      std::string  modelName = argv[1];
//      filename = "atlas/" + modelName + ".onnx";
//      std::cout << "parsing model from " << filename << std::endl;
//   }

   
   Parse(filename);
   return 0;
}
