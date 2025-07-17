{
   
TMVA::Experimental::SOFIE::RModelParser_ONNX p;
auto m = p.Parse("cd.onnx",true);
m.Generate(TMVA::Experimental::SOFIE::Options::kProfile, 1, 0, true);
m.OutputGenerated();
m.PrintGenerated();

}
