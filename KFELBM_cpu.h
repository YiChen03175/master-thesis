/*
@ file KFELBM_cpu.h
*/

double SUM(double *p, int length);
double maximum(double *p, int length);
double minimum(double *p, int length);
void FuzzyEnergyForce(double *F, double *U, double *Img, double m, double sigma, double *c1, double *c2);
void LatticeBoltzmannMethod(double *U,double *fparticle,double *F, double lambda, double time);
double kernel(double x, double v, double sigma);
void NeumannBoundCond(double *u);
void OutputImgContour(double *Img, double *background);
