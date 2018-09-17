/*
@file 
    KFELBM_cpu.cpp
@brief 
    Demo of Fuzzy Energy-based Lattice Boltzmann Method
@test_img 
    test_1(50,50): simple faded out circle
    test_2(128,152): CT images of heart
    test_3(100,160): Natural plane picture
@output_img
    img(8-bit): original image
    IContour(8-bit): inital contour on image (255 inside and 0 outside)
    FContour(24-bit RGB): final contour on image (red line for contour)
    MF(8-bit): final membership function     
@ref 
    my github: https://github.com/YiChen03175
    my thesis: http://etd.lib.nctu.edu.tw/cgi-bin/gs32/hugsweb.cgi?o=dnthucdr&s=id=%22G021040125380%22.&searchmode=basic
*/

#include <iostream>
#include <fstream>
#include <math.h>
#include <string>

#include "KFELBM_cpu.h"

// Set the right height and width respect to different images
#define H 100
#define W 160

using namespace std;

// Set the right path to different test images
string PATH("./image/");

int main(int argc, const char * argv[]) {
    
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
    
    /* Initialize the fparticle */
    double *fparticle;
    fparticle = new double [H*W*9];
    for(int i=0, j=0; j<H*W; i+=9, j++){
        fparticle[i+0] = (4.0/9)*U[j];
        fparticle[i+1] = (1.0/9)*U[j];
        fparticle[i+2] = (1.0/9)*U[j];
        fparticle[i+3] = (1.0/9)*U[j];
        fparticle[i+4] = (1.0/9)*U[j];
        fparticle[i+5] = (1.0/36)*U[j];
        fparticle[i+6] = (1.0/36)*U[j];
        fparticle[i+7] = (1.0/36)*U[j];
        fparticle[i+8] = (1.0/36)*U[j];
    }    
    
    /* Set the iteration parameter */
    int N=0, iterNum=200;
    double lambda = 0.9, m = 2, time = 1, sigma=0.6;
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
    
    /* Iteration Start */
    while(N<iterNum){

        // Calculate new force
        FuzzyEnergyForce(F, U, ImgN, m, sigma, c1, c2);
        
        // Update membership function U
        LatticeBoltzmannMethod(U, fparticle, F, lambda, time);
        
        // Boundary Condition
        NeumannBoundCond(U);
        
        N++;
    }
    
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
 Calculate PDE using Lattice Boltamann Method.
 
 @var *U: membership function
 @var *fparticle: particle 
 @var *F: external force calculate from function FuzzyEnergyForce 
 @var lambda: parameter control effect of fuzzy energy force
 @var time: relaxation time
 -------------------------------------------------*/
void LatticeBoltzmannMethod(double *U,double *fparticle,double *F, double lambda, double time) {
    double *tmp, *feq, con1=4.0/9, con2=1.0/9, con3=1.0/36, con4=1.0/3, con5=1.0/16.97;
    
    tmp = new double [H*W*9];
    feq = new double [H*W*9];
    
    for(int i=0; i<H*W; i++)
        F[i] = F[i] * lambda;
    
    for(int i=0, j=0; j<H*W; i+=9, j++){
        feq[i+0] = U[j] * con1;
        feq[i+1] = U[j] * con2;
        feq[i+2] = U[j] * con2;
        feq[i+3] = U[j] * con2;
        feq[i+4] = U[j] * con2;
        feq[i+5] = U[j] * con3;
        feq[i+6] = U[j] * con3;
        feq[i+7] = U[j] * con3;
        feq[i+8] = U[j] * con3;
    }

    for(int i=0;i<H*W*9; i+=9){
        tmp[i+0] = fparticle[i+0] - (1/time)*(fparticle[i+0]-feq[i+0]);
        tmp[i+1] = fparticle[i+1] - (1/time)*(fparticle[i+1]-feq[i+1]);
        tmp[i+2] = fparticle[i+2] - (1/time)*(fparticle[i+2]-feq[i+2]);
        tmp[i+3] = fparticle[i+3] - (1/time)*(fparticle[i+3]-feq[i+3]);
        tmp[i+4] = fparticle[i+4] - (1/time)*(fparticle[i+4]-feq[i+4]);
        tmp[i+5] = fparticle[i+5] - (1/time)*(fparticle[i+5]-feq[i+5]);
        tmp[i+6] = fparticle[i+6] - (1/time)*(fparticle[i+6]-feq[i+6]);
        tmp[i+7] = fparticle[i+7] - (1/time)*(fparticle[i+7]-feq[i+7]);
        tmp[i+8] = fparticle[i+8] - (1/time)*(fparticle[i+8]-feq[i+8]);
    }
    
    for(int i=0, j=0; j<H*W; i+=9, j++){
        tmp[i+1] = tmp[i+1] + ((2*time-1)/(2*time))*con4*F[j];
        tmp[i+2] = tmp[i+2] + ((2*time-1)/(2*time))*con4*F[j];
        tmp[i+3] = tmp[i+3] + ((2*time-1)/(2*time))*con4*F[j];
        tmp[i+4] = tmp[i+4] + ((2*time-1)/(2*time))*con4*F[j];
        tmp[i+5] = tmp[i+5] + ((2*time-1)/(2*time))*con5*F[j];
        tmp[i+6] = tmp[i+6] + ((2*time-1)/(2*time))*con5*F[j];
        tmp[i+7] = tmp[i+7] + ((2*time-1)/(2*time))*con5*F[j];
        tmp[i+8] = tmp[i+8] + ((2*time-1)/(2*time))*con5*F[j];
    }
    
    for(int i=1; i<H-1; i++){
        for(int j=9; j<(W-1)*9; j+=9){
            fparticle[j+i*W*9+0] = tmp[j+i*W*9+0];
            fparticle[j+i*W*9+1] = tmp[j+(i-1)*W*9+1];
            fparticle[j+i*W*9+2] = tmp[(j+9)+i*W*9+2];
            fparticle[j+i*W*9+3] = tmp[j+(i+1)*W*9+3];
            fparticle[j+i*W*9+4] = tmp[(j-9)+i*W*9+4];
            fparticle[j+i*W*9+5] = tmp[(j-9)+(i-1)*W*9+5];
            fparticle[j+i*W*9+6] = tmp[(j+9)+(i-1)*W*9+6];
            fparticle[j+i*W*9+7] = tmp[(j+9)+(i+1)*W*9+7];
            fparticle[j+i*W*9+8] = tmp[(j-9)+(i+1)*W*9+8];
        }
    }
    
    /* Sum all neighborhood lattice to get new level set funciton */
    for(int i=0, j=0; j<H*W; i+=9, j++){
        U[j] = fparticle[i+0]+fparticle[i+1]+fparticle[i+2]+fparticle[i+3]+fparticle[i+4]+ \
                              fparticle[i+5]+fparticle[i+6]+fparticle[i+7]+fparticle[i+8];
        if (U[j]>1)
            U[j] = 1;
        else if (U[j]<0)
            U[j] = 0;
    }

    delete[] tmp;
    delete[] feq;
}

/*-------------------------------------------------
 Calculate kernel value
 
 -------------------------------------------------*/
double kernel(double x, double v, double sigma){

    return exp((-1*(x-v)*(x-v))/sigma);
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
