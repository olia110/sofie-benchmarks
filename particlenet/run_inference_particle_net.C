//#define DO_DEBUG
//#define RANDOM
//#define RUN_TIMER_MODE

#ifndef NEVTS
#define NEVTS 100
#endif

#ifdef DO_DEBUG
int nevt = 3;
#include "SOFIE_debug.hxx"
#include "particle_net_1_5_10.hxx"
#else

//int nevt = 1000;

//#include "particle_net_1_50_500.hxx"
#include "particle_net_1_10_100.hxx"

#define RANDOM
#endif

#include "TRandom.h"
#include "TStopwatch.h"
#include "check_mem.h"
#include "TH2.h"
#include "TCanvas.h"

#include <omp.h>

void test_particle_net(int nevts = 1000, int n = 1, int n_sv = 10, int n_pf= 100) {
   TStopwatch tw;

   int nprint = nevts/10;
#ifdef DO_DEBUG
   nprint = 1;
#endif


   std::cout << "creating session..." << std::endl;
   check_mem("initial");

   std::string fileName = "particle_net_" + std::to_string(n) + "_" +  std::to_string(n_sv) +  "_" + std::to_string(n_pf) +	".dat";
   TMVA_SOFIE_particle_net::Session s(fileName);

   std::vector<float> pfp(n*2*n_pf);
   std::vector<float> pff(n*20*n_pf);
   std::vector<float> pfm(n*n_pf);

   std::vector<float> svp(n*2*n_sv);
   std::vector<float> svf(n*11*n_sv);
   std::vector<float> svm(n*n_sv);

   auto h2 = new TH2D("h2","Result",8,0,8,100,0,1);

   check_mem("before looping");
   tw.Start();
#ifdef RANDOM
   gRandom->SetSeed(111);
   std::cout << "using random inputs" << std::endl;
#endif

   for (int i = 0; i < nevts; i++) {


#ifdef RANDOM
      std::generate(pfp.begin(), pfp.end(), []{return gRandom->Integer(9)+1;});
      std::generate(pff.begin(), pff.end(), []{return gRandom->Gaus(0,1);});
      std::generate(pfm.begin(), pfm.end(), []{return gRandom->Integer(2);});

      std::generate(svp.begin(), svp.end(), []{return gRandom->Integer(9)+1;});
      std::generate(svf.begin(), svf.end(), []{return gRandom->Gaus(0,1);});
      std::generate(svm.begin(), svm.end(), []{return gRandom->Integer(2);});
#else
      int j = std::min(i,9);
      
      std::fill(pfp.begin(), pfp.end(),int(2+j/2));
      std::fill(pff.begin(), pff.end(),0.1*(j+1));
      std::fill(pfm.begin(), pfm.end(),1);

      std::fill(svp.begin(), svp.end(),int(3+j/2));
      std::fill(svf.begin(), svf.end(),0.2*(j+1));
      std::fill(svm.begin(), svm.end(),1);
   
#endif
      if (i % nprint == 0) {
         std::cout << "inputr for i = " << i << " : ";
         std::cout << pfp[0] << "   " << pff[0] << "  " << pfm[0] << "  " << svp[0] << "  " << svf[0] << "  " << svm[0] << std::endl;
      }

      //std::cout << "running inference..." << std::endl;
      auto result = s.infer(pfp.data(), pff.data(), pfm.data(), svp.data(), svf.data(), svm.data());

      if (i % nprint == 0) {
         std::cout << "ouput for i = " << i << " : ";
         for (auto & e : result)
            std::cout << e << " ";
         std::cout << "\t" << "inputs " << pfp[0] << "  " << pff[0] << std::endl;
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
   c1->Divide(4,2);
   for (int i = 0; i < 8; i++) {
      c1->cd(i+1);
      std::string pname = std::string("py_") + std::to_string(i); 
      auto p = h2->ProjectionY(pname.c_str(),i+1,i+1);
      p->Draw();
   }
   c1->SaveAs("particle_net.pdf");
}

int main() {
   test_particle_net(NEVTS, 1, 10, 100);
}
