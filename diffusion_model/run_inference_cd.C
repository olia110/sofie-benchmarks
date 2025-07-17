//#define DO_DEBUG
//#define RANDOM
//#define RUN_TIMER_MODE

#ifndef NEVTS
#define NEVTS 100
#endif
//#define DO_DEBUG
#ifdef DO_DEBUG
int nevt = 3;
#include "SOFIE_debug.hxx"
#include "cd_debug.hxx"
#else

//int nevt = 1000;

#include "cd.hxx"
#define RANDOM
#endif

#include "TRandom.h"
#include "TStopwatch.h"
#include "check_mem.h"
#include "TProfile.h"
#include "TCanvas.h"



void test_cd(int nevts = 100) {
   TStopwatch tw;

   int nprint = nevts;///10;
#ifdef DO_DEBUG
   nprint = 10;
#endif

   // int n = 1;
   // int n_sv = 100;
   // int n_pf = 1000;

   std::cout << "creating session..." << std::endl;
   check_mem("initial");

   //std::string fileName = "particle_net_" + std::to_string(n) + "_" +  std::to_string(n_sv) +  "_" + std::to_string(n_pf) +	".dat";
   TMVA_SOFIE_cd::Session s; //(fileName);

   auto h1 = new TH1D("h1","Total Energy",100,110,100);
   auto p1 = new TProfile("p1","Longitudinal profile",45,0,45);

   std::vector<float> input(8);
   input[0] = 100.; // energy in Gev
   input[1] = 0.0; // phi
   input[2] = 1.57; // theta
   input[3] = 1.0; // flag for geometry
   check_mem("before looping");
   tw.Start();
//#ifdef RANDOM
//   gRandom->SetSeed(111);
//   std::cout << "using random inputs" << std::endl;
//#endif

   for (int i = 0; i < nevts; i++) {



      std::cout << "running inference...  " << i << std::endl;
      auto result = s.infer(input.data());

      if (i % nprint == 0) {
         std::cout << "ouput for i = " << i << " : ";
         for (auto & e : result)
            std::cout << e << " ";
//         std::cout << "\t" << "inputs " << pfp[0] << "  " << pff[0] << std::endl;
         std::cout  << std::endl;
         check_mem("at current event");
      }

       // output shpe is 1 x 9 x 16 x 45   (x x y x z)
      assert(result.size() == 9*16*45);
      double sum = 0;
      std::vector<double> sumy(45, 0.);
      for (size_t ix = 0; ix < 9 ; ix++) {
         for (size_t iy = 0; iy < 16 ; iy++) {
            for (size_t iz = 0; iz < 45; iz++) {
               double y = result[ 45*16*ix + 45*iy + iz];
               sumy[iz] += y;
               sum += y;
            }
         }
      }
      std::cout << "total energy " << sum << std::endl;

      h1->Fill(sum);
      for (int i = 0; i < 45; i++)
         p1->Fill(double(i)+0.5,sumy[i]);



      // for (auto & r : result) {
      //    h1->Fill(r[0]);
      //    h2->Fill(r[1]);
      //    h3->Fill(r[2]);
      // }
   }

   tw.Print();
   check_mem("memory at the end");

//#ifdef PLOT
   auto c1 = new TCanvas();
   c1->Divide(1,2);
   c1->cd(1);
   h1->Draw();
   c1->cd(2);
   p1->Draw();
   c1->SaveAs("c1.pdf");
//#endif
}

int main() {
   test_cd(NEVTS);
}
