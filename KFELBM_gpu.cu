/*
@file 
    KFELBM_gpu.cu
@brief [GPU version]
	Demo of Kernel Fuzzy Energy Active Contour using Lattice Boltzmann
    Method implementing on CPU 
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
#include <time.h>

#include "KFELBM_gpu.h"

//Set the right height and width respect to different images
#define H 100
#define W 160

using namespace std;

// Set the right path to different test images
string PATH("./image/");

/*------------------------------------------------
 Cuda kernel run on GPU
 
 @var *U: membership function
 @var *F: external force calculate from function FuzzyEnergyForce
 @var *fparticle: particle distribution
 @var *tmp: temporary variable for calculating particle distribution
 @var lambda: parameter control effect of fuzzy energy force
 @var time: relaxation time
 -------------------------------------------------*/
__global__
void LBM(double* U, double* fparticle, double* F, double* tmp,double lambda, double time){
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double con1=4.0/9, con2=1.0/9, con3=1.0/36, con4=1.0/3, con5=1.0/16.97;

	while (tid<H*W){

		// Bhatnagar, Gross, and Krook collision model
		tmp[tid*9+0] = fparticle[tid*9+0] - (1/time)*( fparticle[tid*9+0]-(U[tid]*con1) );
        tmp[tid*9+1] = fparticle[tid*9+1] - (1/time)*( fparticle[tid*9+1]-(U[tid]*con2) );
        tmp[tid*9+2] = fparticle[tid*9+2] - (1/time)*( fparticle[tid*9+2]-(U[tid]*con2) );
        tmp[tid*9+3] = fparticle[tid*9+3] - (1/time)*( fparticle[tid*9+3]-(U[tid]*con2) );
        tmp[tid*9+4] = fparticle[tid*9+4] - (1/time)*( fparticle[tid*9+4]-(U[tid]*con2) );
        tmp[tid*9+5] = fparticle[tid*9+5] - (1/time)*( fparticle[tid*9+5]-(U[tid]*con3) );
        tmp[tid*9+6] = fparticle[tid*9+6] - (1/time)*( fparticle[tid*9+6]-(U[tid]*con3) );
        tmp[tid*9+7] = fparticle[tid*9+7] - (1/time)*( fparticle[tid*9+7]-(U[tid]*con3) );
        tmp[tid*9+8] = fparticle[tid*9+8] - (1/time)*( fparticle[tid*9+8]-(U[tid]*con3) );

        // Including external force
        tmp[tid*9+1] += ((2*time-1)/(2*time))*con4*lambda*F[tid];
        tmp[tid*9+2] += ((2*time-1)/(2*time))*con4*lambda*F[tid];
        tmp[tid*9+3] += ((2*time-1)/(2*time))*con4*lambda*F[tid];
        tmp[tid*9+4] += ((2*time-1)/(2*time))*con4*lambda*F[tid];
        tmp[tid*9+5] += ((2*time-1)/(2*time))*con5*lambda*F[tid];
        tmp[tid*9+6] += ((2*time-1)/(2*time))*con5*lambda*F[tid];
        tmp[tid*9+7] += ((2*time-1)/(2*time))*con5*lambda*F[tid];
        tmp[tid*9+8] += ((2*time-1)/(2*time))*con5*lambda*F[tid];
		
		// Distribution diffusion
		if(tid>2*W && tid<H*(W-2) && tid%W!=0 && (tid+1)%W!=0){
			fparticle[tid*9+0] = tmp[tid*9+0];
        	fparticle[tid*9+1] = tmp[(tid-W)*9+1];
        	fparticle[tid*9+2] = tmp[(tid+1)*9+2];
        	fparticle[tid*9+3] = tmp[(tid+W)*9+3];
	        fparticle[tid*9+4] = tmp[(tid-1)*9+4];
	        fparticle[tid*9+5] = tmp[(tid-W-1)*9+5];
	        fparticle[tid*9+6] = tmp[(tid-W+1)*9+6];
	        fparticle[tid*9+7] = tmp[(tid+W+1)*9+7];
	        fparticle[tid*9+8] = tmp[(tid+W-1)*9+8];
		}

		// Accumulate distribution at each grid point
        U[tid] = fparticle[tid*9+0]+fparticle[tid*9+1]+fparticle[tid*9+2]+ \
        		 fparticle[tid*9+3]+fparticle[tid*9+4]+fparticle[tid*9+5]+ \
        		 fparticle[tid*9+6]+fparticle[tid*9+7]+fparticle[tid*9+8];

       	// Make sure no excced value
        if (U[tid]>1)
        	U[tid] = 1;
        else if (U[tid]<0)
        	U[tid] = 0;

		tid += blockDim.x * gridDim.x;
	}

}

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
    int N=0, iterNum=0;
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
    
    /* Cuda initialization */
    double *dev_fparticle, *dev_F, *dev_U, *dev_tmp;

    cudaMalloc( (void**)&dev_fparticle, H*W*9*sizeof(double) );
    cudaMalloc( (void**)&dev_tmp, H*W*9*sizeof(double) );
	cudaMalloc( (void**)&dev_F, H*W*sizeof(double) );
	cudaMalloc( (void**)&dev_U, H*W*sizeof(double) );
	
	cudaMemcpy( dev_fparticle, fparticle, H*W*9*sizeof(double),cudaMemcpyHostToDevice );
	cudaMemcpy( dev_U, U, H*W*sizeof(double),cudaMemcpyHostToDevice );

	start = clock(); // time start

    /* Iteration Start */
    while(N<iterNum){

        // Calculate new force
        FuzzyEnergyForce(F, U, ImgN, m, sigma, c1, c2);

		cudaMemcpy( dev_F, F, H*W*sizeof(double),cudaMemcpyHostToDevice );
        
        LBM<<<128, 128>>>(dev_U, dev_fparticle, dev_F, dev_tmp, lambda, time);
        
        cudaMemcpy( U, dev_U, H*W*sizeof(double),cudaMemcpyDeviceToHost );

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

    cudaFree(dev_fparticle);
	cudaFree(dev_U);
	cudaFree(dev_F);
	cudaFree(dev_tmp);
    
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
