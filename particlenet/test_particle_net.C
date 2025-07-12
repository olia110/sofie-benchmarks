void test_particle_net(bool verbose = 0, int n = 1, int n_sv = 10, int n_pf = 100) {

   TMVA::Experimental::SOFIE::RModelParser_ONNX p;
   auto m = p.Parse("particle-net.onnx", verbose);
   // define initial tensor shapes
   std::map<std::string, size_t> mp;
   mp["N"]=n; mp["n_sv"] = n_sv; mp["n_pf"] = n_pf;
   m.PrintInitializedTensors();
   m.Initialize(mp, verbose);
   TMVA::Experimental::SOFIE::Options opt = TMVA::Experimental::SOFIE::Options::kProfile;

   m.Generate(opt, -1, 0, verbose);
   m.OutputGenerated();

   gSystem->Exec(TString::Format("cp particle_net.hxx particle_net_%d_%d_%d.hxx",n,n_sv,n_pf));
   gSystem->Exec(TString::Format("cp particle_net.dat particle_net_%d_%d_%d.dat",n,n_sv,n_pf));


}
