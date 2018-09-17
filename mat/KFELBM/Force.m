%
% This matlab file provides function for updating external force from 
% kernel fuzzy energy in KFELBM algorithm. 
%
% Execution version: MATLAB R2016a
%

function [F ,c1 ,c2] = Force(U0, Img, c1, c2, m, sigma)

U=U0;

% calculate centroid in fuzzy clustering
[c1 ,c2] = UpdateC(U, Img, c1, c2, sigma, m);    

% calculate external force by kernel fuzzy energy
F = m.*((1-U).^(m-1)).*(1-KERNEL(Img,c2,sigma))-m.*(U.^(m-1)).*(1-KERNEL(Img,c1,sigma)); 

    
function [c1 ,c2] = UpdateC(U, Img,v1, v2, sigma, m)

% calculate numerator
Tmp1 = (U.^m).*Img.*KERNEL(Img,v1,sigma);
Tmp2 = ((1-U).^m).*Img.*KERNEL(Img,v2,sigma);

% calculate denominator
De1 = (U.^m).*KERNEL(Img,v1,sigma);
De2 = (1-U).^m.*KERNEL(Img,v2,sigma);

c1 = sum(Tmp1(:))/sum(De1(:));
c2 = sum(Tmp2(:))/sum(De2(:));

function value = KERNEL(x,v,sigma)

value = exp(-1.*((x-v).^2)/sigma);