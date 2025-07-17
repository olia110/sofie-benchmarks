//#define DO_DEBUG
//#define RANDOM
//#define RUN_TIMER_MODE

//#define USE_DYNAMIC


#ifndef NEVTS
#define NEVTS 10
#endif

#ifdef DO_DEBUG
int nevt = 3;
#include "SOFIE_debug.hxx"

#include "gnn_small.hxx"
#else

//int nevt = 1000;

#ifdef USE_DYNAMIC
#include "gnn_small_dynamic.hxx"
#else
#include "gnn_small.hxx"
#endif

#define RANDOM
#endif

#include "TRandom.h"
#include "TStopwatch.h"
#include "check_mem.h"
#include "TH2.h"
#include "TCanvas.h"


// number of edges needs to be less or equal than number of hits
void test_model(int nevts = 1000, int ne = 0, int nh= 20) {
   TStopwatch tw;

   int nprint = nevts/10;
#ifdef DO_DEBUG
   nprint = 1;
#endif

   // int n = 1;
   // int n_sv = 100;
   // int n_pf = 1000;

   std::cout << "creating session..." << std::endl;
   check_mem("initial");

   //std::string fileName = "/Users/moneta/cernbox/root/tests/tmva/sofie/particle_net_" + std::to_string(n) + "_" +  std::to_string(n_sv) +  "_" + std::to_string(n_pf) +	".dat";
#ifdef USE_DYNAMIC
   TMVA_SOFIE_gnn_small::Session s("gnn_small_dynamic.dat",nh, ne);
#else
   TMVA_SOFIE_gnn_small::Session s("gnn_small.dat");
#endif

   std::vector<float> x(nh*12);
   std::vector<int64_t> eidx(ne*2);
   std::vector<float> ea(ne*6);

   auto h2 = new TH2D("h2","Result",10,0,10,100,0,1);

   // auto h1 = new TH1D("h1","h1",50,-1,0);
   // auto h2 = new TH1D("h2","h2",50,-1,0);
   // auto h3 = new TH1D("h3","h3",50,-1,0);
   check_mem("before looping");
   tw.Start();
#ifdef RANDOM
   gRandom->SetSeed(111);
   std::cout << "using random inputs" << std::endl;
#endif

   for (int i = 0; i < nevts; i++) {


#ifdef RANDOM
      std::generate(x.begin(), x.end(), []{return gRandom->Gaus(0,3);});
      std::generate(eidx.begin(), eidx.end(), [&]{return gRandom->Integer(nh);});
      std::generate(ea.begin(), ea.end(), []{return gRandom->Gaus(0,5);});

#else
      //std::fill(pff.begin(), pff.end(),0.1*(j+1));
   
#endif
      if (i % nprint == 0) {
         std::cout << "inputr for i = " << i << " : ";
         std::cout << x[0] << "   " << eidx[0] << "   " << ea[0] << "....." << std::endl;
      }

      //std::cout << "running inference..." << std::endl;
#ifdef USE_DYNAMIC
      auto result = s.infer(nh, x.data(), ne, eidx.data(), ea.data());
#else      
      auto result = s.infer(x.data(), eidx.data(), ea.data());
#endif
 

      if (i % nprint == 0) {
         std::cout << "ouput for i = " << i << " : ";
//         for (auto & e : result)
//            std::cout << e << " ";
         std::cout << " result: "  << result[0] << std::endl;
         //std::cout << "\t" << "inputs " << pfp[0] << "  " << pff[0] << std::endl;
         std::cout  << std::endl;
         check_mem("at current event");
      }

      float ires = 0.5;
      for (auto & r : result) {
         h2->Fill(ires,r);
         ires++;
      }
   }

   tw.Print();
   check_mem("memory at the end");
   auto c1 = new TCanvas();
   c1->Divide(2,5);
   for (int i = 0; i < 10; i++) {
      c1->cd(i+1);
      std::string pname = std::string("py_") + std::to_string(i); 
      auto p = h2->ProjectionY(pname.c_str(),i+1,i+1);
      p->Draw();
   }
   c1->SaveAs("model_tracking.pdf");
}

int main(int argc, char **argv) {
   
   int ne = 300000;
   int nh = 100000;
   int nevts = NEVTS;
   if (argc > 2) {
      ne = std::atoi(argv[1]);
      nh = std::atoi(argv[2]);
   }
   if (argc > 3) 
      nevts = std::atoi(argv[3]);
   
   std::cout << "testing model with nedges: " << ne << " nhits:  " << nh << std::endl;
      test_model(nevts, ne, nh);
}

