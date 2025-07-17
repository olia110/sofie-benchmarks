#include <cmath>
#define NTRK 100
#include "network_gn2.hxx"


#ifndef NEVTS
#define NEVTS 1000
#endif

#include <chrono>
#include "check_mem.h"

#include "TRandom.h"
#include "TStopwatch.h"
#include "TCanvas.h"
#include "TH1.h"


void test_inference(int n = 10000) {
   TStopwatch tw;

   int ntrack = NTRK;

   std::cout << "runnning SOFIE inference for ATLAS GN2 model with ntrk = " << ntrack << std::endl;

   check_mem("initial - before session creation");

   TMVA_SOFIE_network_gn2::Session s;   // model has no weight file

   std::vector<float> tf(ntrack*14,1.);
   std::vector<float> jf(2,2.);

   auto h1 = new TH1D("h1","h1",50,-1,0);
   auto h2 = new TH1D("h2","h2",50,-1,0);
   auto h3 = new TH1D("h3","h3",50,-1,0);
   check_mem("memory before looping");
   tw.Start();
   auto t1 = std::chrono::high_resolution_clock::now();
   for (int i = 0; i < n; i++) {

      for (int j = 0; j< tf.size(); j++)
         tf[j] = gRandom->Uniform(-10,10);
      for (int j = 0; j< jf.size(); j++)
         jf[j] = gRandom->Uniform(-10,10);

      auto result = s.infer(jf.data(), tf.data());

      if (i % (n/10) == 0) {
         std::cout << "Result " << i << " : " << result[0][0] << "  " << result[1][0] << "  " << result[2][0] << std::endl;
         check_mem("at current event");
      }
      for (auto & r : result) {
          h1->Fill(r[0]);
          h2->Fill(r[1]);
          h3->Fill(r[2]);
      }
   }
   auto t2 = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
   std::cout << "total duration " << duration/1.E6 << " (sec)" << std::endl;
   tw.Print();
   auto c1 = new TCanvas();
   c1->Divide(1,3);
   c1->cd(1);
   h1->Draw();
   c1->cd(2);
   h2->Draw();
   c1->cd(3);
   h3->Draw();
   c1->SaveAs("atlas_gn2.pdf");

   check_mem("memory at the end");
}

int main() {
   test_inference(NEVTS);
}
