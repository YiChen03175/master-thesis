/*
@ file KFEAC_CPU.h
*/

double SUM(double *p, int length);
double maximum(double *p, int length);
double minimum(double *p, int length);
void FuzzyEnergyForce(double *F, double *U, double *Img, double m, double sigma, double *c1, double *c2);
void DifferentialMethod(double *F, double*u, double timestep, double mu, double epsilon);
double* Curvature_central(double *u);
double* gradientx(double *f, int height, int width);
double* gradienty(double *f, int height, int width);
double kernel(double x, double v, double sigma);
void NeumannBoundCond(double *u);
void OutputImgContour(double *Img, double *background);
