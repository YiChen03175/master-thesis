%
% This matlab file creates a film recording membership function varying 
% with iterations in KFAC algorithm.
%
% Execution version: MATLAB R2016a
%
% This algorithm comes from the paper, 
% Y. Wu, W. Ma, M. Gong, H. Li, and L. Jiao,
% "Novel fuzzy active contour model with kernel metric for image segmentation",
% Appl. Soft. Comput., vol. 34, pp. 301?311, Sep. 2015.
%
% URL: https://www.sciencedirect.com/science/article/pii/S1568494615002951
%
% NOTE1: The parameters for initial curve has been fixed to make sure the
%        intial setting as same as KFELBM.
%
    

clc;clear all;

Img = imread('../../image/img.bmp');
Img = double(Img(:,:,1));
ImgN = (Img-min(Img(:))) ./ (max(Img(:))-min(Img(:)));

[row ,col] = size(Img);
initialU =zeros(row,col);

%initial parameter for initial curve
a = 70;
b = 20;
r = 0.45;

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
U = (1-U);

% set parameters for KFAC
sigma = 0.6;
mu = 0.4;
m = 2;
epsilon =1.0;
timestep = 0.1;

% intitialize c1, c2
nu1 = (U>0.5).*ImgN;
nu2 = (U<=0.5).*ImgN;
de1 = (U>0.5);
de2 = (U<=0.5);

c1 = sum(nu1(:))/sum(de1(:));
c2 = sum(nu2(:))/sum(de2(:));

% set film parameters 
writerObj = VideoWriter('KFAC_film.avi');
writerObj.FrameRate = 50;
open(writerObj);

for n=1:300 
    
    % update external force from fuzzy energy
    [U, c1, c2] = KFEAC(U, ImgN, c1, c2, mu, m, epsilon, timestep, sigma);
    
    % set parameters for film
    mesh(U);
    title([num2str(n), ' iterations']);
    view(-11,63);
    zlabel('memebership function');
    colormap('jet');
    zlim([0,1]);
    
    % get each frame in each iteration
    frame = getframe(gcf);
    writeVideo(writerObj, frame);

end

close(writerObj);