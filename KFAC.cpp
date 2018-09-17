/*
@file FEAC_CPU.cpp
@brief 
    Demo of Kerenl Fuzzy Energy Active Contour
@test_img 
    test_1(50,50): simple faded out circle
    test_2(128,152): CT images of heart
    test_3(100,160): Natural plane picture
@output_img
    IContour(8-bit): inital contour on image (255 inside and 0 outside)
    FContour(24-bit RGB): final contour on image (red line for contour)
    MF(8-bit): final membership function     
@ref 
    This algorithm comes from the paper, 
    Y. Wu, W. Ma, M. Gong, H. Li, and L. Jiao,
    "Novel fuzzy active contour model with kernel metric for image segmentation",
    Appl. Soft. Comput., vol. 34, pp. 301?311, Sep. 2015.

    URL: https://www.sciencedirect.com/science/article/pii/S1568494615002951
*/

#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <time.h>

#include "KFAC.h"

// Set the right height and width respect to different images
#define H 100
#define W 160

using namespace std;

// Set the right path to different test images
string PATH("./image/");

int main(int argc, const char * argv[]) {
    
    // time variable
    clock_t start, end; 

    /* Read input image in raw data */
    double *Img, *ImgN;
    unsigned char ch;
    Img = new double [H*W];
    ImgN = new double [H*W];
    fstream myfile;
    
    myfile.open((PATH+"img.raw").c_str(), ios::in|ios::binary);
    
    if(!myfile)
    {
        cout << "Something wrong with the image!" << endl;
        return 0;
    }else{
        cout << "Loading image ..." << endl;
        for(int i=0; i<H; i++){
            for(int j=0; j<W; j++){
                myfile.read((char*)&ch, sizeof(char));
                Img[j+W*i] = (double)ch;
            }
        }
        myfile.close();
        cout << "Loading Finish!" << endl;
    }
    
    /* Normalize the image data */
    double Img_min = minimum(Img, H*W);
    double Img_max = maximum(Img, H*W);
    for (int i=0; i<H*W; i++){
        ImgN[i] = (Img[i]-Img_min)/(Img_max-Img_min);
    }
    
    /* 
    Initialize the level set function
    (a, b) are the center of circle
    */
    double a=50, b=75, r=0.45;
    double *U, *IContour;
    U = new double [H*W];
    IContour = new double [H*W];
    
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            U[j+i*W] = pow((pow((i-a),2)+pow((j-b),2)),0.5);
        }
    }

    /* make sure value not exceed */
    double U_min = minimum(U, H*W);
    double U_max = maximum(U, H*W);
    for(int i=0; i<H*W; i++){
        U[i] = (U[i]-U_min)/(U_max-U_min) + r;
        if(U[i]>1)
            U[i]=1;
        else if(U[i]<0)
            U[i]=0;
    }
    
    /* Output initial contour */
    for(int i=0;i<H*W;i++){
        if (U[i]<=0.5){
            IContour[i]=255;
        }else{
            IContour[i]=0;
        }
    }

    fstream ofile;

    ofile.open((PATH+"output/IContour.raw").c_str(), fstream::out | fstream::binary);
    if( !ofile.good())
        cout << "ofile讀檔失敗" << endl;
    
    for(int i = 0 ; i < H ; i++){
        for(int j = 0 ; j < W ; j++){
                    ofile << (char)IContour[j+i*W];
            }}
    ofile.close();
    
    /* Set the iteration parameter */
    int N=0, iterNum=300;
    double mu=0.4, sigma=3, m=2, timestep=0.1, epsilon=1.0;
    double *F;
    F = new double [H*W];

    /* Initialize c1, c2 for the first loop of kernel */
    double *Nu1, *De1, *Nu2, *De2, *c1, *c2;
    Nu1 = new double [H*W];
    Nu2 = new double [H*W];
    De1 = new double [H*W];
    De2 = new double [H*W];
    c1 = new double;
    c2 = new double;

    for(int i=0; i<H*W; i++){
        Nu1[i] = (U[i]>0.5) * ImgN[i];
        Nu2[i] = (U[i]<=0.5) * ImgN[i];
        De1[i] = (U[i]>0.5) * 1;
        De2[i] = (U[i]<=0.5) * 1;
    }

    *c1 = SUM(Nu1, H*W) / SUM(De1, H*W);
    *c2 = SUM(Nu2, H*W) / SUM(De2, H*W);

    delete[] Nu1;
    delete[] Nu2;
    delete[] De1;
    delete[] De2;

    start = clock(); // time start

    /* Iteration Start */
    while(N<iterNum){

        // Calculate new force
        FuzzyEnergyForce(F, U, ImgN, m, sigma, c1, c2);
        
        // Update membership function U
        DifferentialMethod(F, U, timestep, mu, epsilon);
        
        // Boundary Condition
        NeumannBoundCond(U);

        N++;
    }

    end = clock(); // time end
    
    cout << double(end - start) / CLOCKS_PER_SEC <<endl;
    
    /* Output the final image */
    OutputImgContour(U, Img);
    ofile.open((PATH+"output/MemFuction.raw").c_str(), fstream::out | fstream::binary);
    if( !ofile.good())
        cout << "ofile讀檔失敗" << endl;
    
    for(int i = 0 ; i < H ; i++){
        for(int j = 0 ; j < W ; j++){
            ofile << (char)(U[j+i*W]*255);
        }}
    ofile.close();
    
    delete c1, c2;
    delete[] IContour;
    delete[] F;
    delete[] Img;
    delete[] ImgN;
    delete[] U;
    return 0;
}

/*------------------------------------------------
 Calculate the sum of elements in array
 
 @var *p: array
 @var length: length of array
 -------------------------------------------------*/
double SUM(double *p, int length){
    double sum = 0;
    
    for(int i=0; i<H*W; i++){
        sum += p[i];
    }
    
    return sum;
}

/*------------------------------------------------
 Calculate the maximum of elements in array
 
 @var *p: array
 @var length: length of array
 -------------------------------------------------*/
double maximum(double *p, int length){
    double maximum = p[0];
    
    for (int i=0;i<length;i++){
        if(p[i]>maximum)
            maximum = p[i];
    }
    
    return maximum;
}

/*------------------------------------------------
 Calculate the minimum of elements in array
 
 @var *p: array
 @var length: length of array
 -------------------------------------------------*/
double minimum(double *p, int length){
    double minimum = p[0];
    
    for (int i=0;i<length;i++){
        if(p[i]<minimum)
            minimum = p[i];
    }
    
    return minimum;
}

/*------------------------------------------------
 Calculate additional force 
 
 @var *F: external force calculate from function FuzzyEnergyForce
 @var *U: membership function
 @var *Img: input image
 @var m: parameter control fuzziness
 -------------------------------------------------*/
void FuzzyEnergyForce(double *F, double *U, double *Img, double m, double sigma, double *c1, double *c2){
    double *Nu1, *De1, *Nu2, *De2;
    Nu1 = new double [H*W];
    Nu2 = new double [H*W];
    De1 = new double [H*W];
    De2 = new double [H*W];
    
    for(int i=0; i<H*W; i++){
        Nu1[i] = pow(U[i], m) * Img[i] * kernel(Img[i], *c1, sigma);
        Nu2[i] = pow(1-U[i], m) * Img[i] * kernel(Img[i], *c2, sigma);
        De1[i] = pow(U[i], m) * kernel(Img[i], *c1, sigma);
        De2[i] = pow(1-U[i], m) * kernel(Img[i], *c2, sigma);
    }
    
    /* Update average c1&c2 */
    *c1 = SUM(Nu1, H*W) / SUM(De1, H*W);
    *c2 = SUM(Nu2, H*W) / SUM(De2, H*W);
    
    /* Calculate the externel force */
    for(int i=0; i<H*W; i++)
        F[i] = m*pow(1-U[i], m-1)*(1-kernel(Img[i], *c2, sigma)) - m*pow(U[i], m-1)*(1-kernel(Img[i], *c1, sigma));
    
    delete[] Nu1;
    delete[] Nu2;
    delete[] De1;
    delete[] De2;
}

/*------------------------------------------------
 Calculate PDE using gradient descent
 
 @var *U: membership function
 @var *F: external force calculate from function FuzzyEnergyForce
 @var timestep: parameter control the step of gradient descent
 @var mu: parameter control the smoothness of contour
 @var epsilon: constant parameter
 -------------------------------------------------*/
void DifferentialMethod(double *F, double*u, double timestep, double mu, double epsilon) {
    double *K, *DrcU, *Lengthterm;
    DrcU = new double [H*W];
    Lengthterm = new double [H*W];
    
    // Curvature term
    K = Curvature_central(u);
    
    // delta function
    for(int i=0;i<H*W;i++)
        DrcU[i] = (epsilon/M_PI)/(pow(epsilon,2)+pow(u[i],2));

    
    for(int i=0;i<H*W;i++)
        Lengthterm[i]=mu*DrcU[i]*K[i];
    
    for(int i=0;i<H*W;i++){
        u[i] = u[i] + timestep*(F[i]+Lengthterm[i]);
        if (u[i]>1)
            u[i]=1;
        else if (u[i]<0)
            u[i]=0;
    }
    
    delete [] K;
    delete [] DrcU;
    delete [] Lengthterm;
}
/*-------------------------------------------------
 Calculate kernel value
 
 -------------------------------------------------*/
double kernel(double x, double v, double sigma){

    return exp((-1*(x-v)*(x-v))/sigma);
}
/*-------------------------------------------------
 Calculate curvature term
 
 -------------------------------------------------*/
double* Curvature_central(double *u){
    double *ux, *uy, *Nx, *Ny, *NormDu, *nxx, *nyy, *k;
    NormDu = new double [H*W];
    Nx = new double [H*W];
    Ny = new double [H*W];
    k = new double [H*W];
    
    // Calculate curvature
    ux = gradientx(u, H, W);
    uy = gradienty(u, H, W);
    
    for(int i=0;i<H*W;i++){
        NormDu[i] = sqrt(pow(ux[i],2)+pow(uy[i],2)+1e-8);
        Nx[i] = ux[i]/NormDu[i];
        Ny[i] = uy[i]/NormDu[i];
    }
    
    nxx = gradientx(Nx, H, W);
    nyy = gradienty(Ny, H, W);
    
    for(int i=0;i<H*W;i++)
        k[i]=nxx[i]+nyy[i];
    
    delete [] ux;
    delete [] uy;
    delete [] Nx;
    delete [] Ny;
    delete [] NormDu;
    delete [] nxx;
    delete [] nyy;
    
    return k;
}

/*-------------------------------------------------
 Calculate gradient in x direction
 
 -------------------------------------------------*/
double* gradientx(double *f, int height, int width){
    double *fx;
    fx = new double [H*W];
    
    // center part
    for(int i=0;i<H;i++){
        for(int j=1;j<W-1;j++){
            fx[j+i*W] = 0.5*(f[(j+1)+i*W]-f[(j-1)+i*W]);
        }
    }
    
    // side part
    for(int i=0;i<H;i++){
        fx[0+i*W]=f[1+i*W]-f[0+i*W]; // left
        fx[(W-1)+i*W]=f[(W-1)+i*W]-f[(W-2)+i*W]; // right
    }
    
    return fx;
}

/*-------------------------------------------------
 Calculate gradient in y direction

 -------------------------------------------------*/
double* gradienty(double *f, int height, int width){
    double *fy;
    fy = new double [H*W];
    
    // center part
    for(int i=1;i<H-1;i++){
        for(int j=0;j<W;j++){
            fy[j+i*W] = 0.5*(f[j+(i+1)*W]-f[j+(i-1)*W]);
        }
    }
    
    // side part
    for(int i=0;i<W;i++){
        fy[i+0*W]=f[i+1*W]-f[i+0*W]; // left
        fy[i+(W-1)*W]=f[i+(W-1)*W]-f[i+(W-2)*W]; // right
    }
    
    return fy;
}
/*------------------------------------------------
 Boundary Condition
 
 @var *u: membership function
 @var *g: membership function after boundary condition
 -------------------------------------------------*/

void NeumannBoundCond(double *u){
    
    double *g;
    g = new double [H*W];
    
    // Copy
    for(int i=0; i<H*W; i++)
        g[i] = u[i];
    
    // Four corner points
    g[0+W*0] = u[2+W*2];
    g[0+W*(H-1)] = u[2+W*(H-3)];//bug
    g[(W-1)+W*0] = u[(W-3)+W*2];
    g[(W-1)+W*(H-1)] = u[(W-3)+W*(H-3)];

    // Four edge around image u
    for(int i=1; i<W-1; i++){
        g[i+W*0] = u[i+W*2];
        g[i+W*(H-1)] = u[i+W*(H-3)];
    }
    for(int i=1; i<H-1; i++){
        g[0+W*i] = g[2+W*i];
        g[(W-1)+W*i] = g[(W-3)+W*i];
    }

    // Copy Back
    for(int i=0; i<H*W; i++)
        u[i] = g[i];
    
    delete[] g;
}

/*------------------------------------------------
Output image contour in RBG(red line)

@var *Img: membership function used to find contour
@var *background: origin figure which will turn into RGB 
 -------------------------------------------------*/
void OutputImgContour(double *Img, double *background){
    fstream oput;
    oput.open((PATH+"output/FContour.raw").c_str(), fstream::out | fstream::binary);
    if( !oput.good())
        cout << "ofile讀檔失敗" << endl;
    
    for(int i = 0 ; i < H ; i++){
        for(int j = 0 ; j < W ; j++){
            if(Img[j+i*W]<=0.5 && (Img[(j+1)+i*W]>0.5||Img[(j-1)+i*W]>0.5||Img[j+(i-1)*W]>0.5||Img[j+(i+1)*W]>0.5||Img[(j+1)+(i+1)*W]>0.5||Img[(j-1)+(i+1)*W]>0.5||Img[(j+1)+(i-1)*W]>0.5||Img[(j-1)+(i-1)*W]>0.5)){
                oput << (char)255;
                oput << (char)0;
                oput << (char)0;
            }else{
                oput << (char)background[j+i*W];
                oput << (char)background[j+i*W];
                oput << (char)background[j+i*W];
            }
        }}
    oput.close();
    
}
