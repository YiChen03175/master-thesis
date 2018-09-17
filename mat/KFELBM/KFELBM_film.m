%
% This matlab file creates a film recording membership function varying 
% with iterations in KFELBM algorithm.
%
% Execution version: MATLAB R2016a
%
% There are more details in my thesis, 
% "Kernel Fuzzy Energy Active Contour Using Lattice Boltzmann Method In Image Segmentation"
%
% Author: Yi Chen
% Advisor: Professor Ching-Han Hsu
% Email: cy477622@gmail.com
% URL: http://etd.lib.nctu.edu.tw/cgi-bin/gs32/hugsweb.cgi/ccd=dbBQ0c/record?r1=1&h1=2
%
% NOTE1: The parameters for initial curve has been fixed to make sure the
%        intial setting as same as KFAC.
%
% NOTE2: This KFELBM algorithm is a valid CPU version for visualization
%        that run the kernel LBM on CPU instead of GPU.
%        You can get GPU version on my github:
%        https://github.com/YiChen03175
%        

clc;clear all;

Img = imread('../../image/img.bmp');
Img = double(Img(:,:,1));

% normailize image pixels value 
ImgN = (Img-min(Img(:))) ./ (max(Img(:))-min(Img(:)));

[row ,col] = size(Img);
initialU =zeros(row,col);

% set parameters for initial curve
a = 70;
b = 20;
r = 0.45;

% set intial function to be a signed distance function (SDF)
for i=1:row
    for j=1:col
       initialU(i,j) = (((i-a).^2+(j-b).^2).^0.5);
   end
end

% normalize membership function value
U = initialU;
U = (U-min(U(:))) ./ (max(U(:))-min(U(:))) + r;

% make sure there are no exceed value
U = U.*(U<=1)+(U>1);
U = U.*(U>=0);

% reverse the membership function just for visualization
U = 1-U;

% set parameters for KFELBM
sigma = 0.6;
lambda = 0.9;
m = 2;
time = 1;

% initialize particle distrbution as figure below
%
%      1/36  1/9  1/36
%          \  |  /
%           \ | /
%     1/9 -- 4/9 -- 1/9
%           / | \
%          /  |  \
%      1/36  1/9  1/36
%
fparticle = zeros(row, col, 9);
fparticle(:,:,1) = (4/9)*U;
fparticle(:,:,2) = (1/9)*U;
fparticle(:,:,3) = (1/9)*U;
fparticle(:,:,4) = (1/9)*U;
fparticle(:,:,5) = (1/9)*U;
fparticle(:,:,6) = (1/36)*U;
fparticle(:,:,7) = (1/36)*U;
fparticle(:,:,8) = (1/36)*U;
fparticle(:,:,9) = (1/36)*U;

% intitialize c1, c2
nu1 = (U>0.5).*ImgN;
nu2 = (U<=0.5).*ImgN;
de1 = (U>0.5);
de2 = (U<=0.5);

c1 = sum(nu1(:))/sum(de1(:));
c2 = sum(nu2(:))/sum(de2(:));

% set film parameters 
writerObj = VideoWriter('KFELBM_film.avi');
writerObj.FrameRate = 10;
open(writerObj);

for n=1:50 
    
    % update external force from fuzzy energy
    [F ,c1 ,c2] = Force(U, ImgN, c1, c2, m, sigma);
    
    % update memebership function using Lattice Boltzmann Method
    [U ,fparticle] =  LBM(U, fparticle,F,lambda,time);
    
    % set parameters for film
    mesh(U); 
    title([num2str(n), ' iterations']); 
    view(-11,63); 
    colormap('jet'); 
    zlabel('memebership function'); 
    zlim([0,1]); 
    
    % get each frame in each iteration
    frame = getframe(gcf);
    writeVideo(writerObj, frame);
    
end

close(writerObj);