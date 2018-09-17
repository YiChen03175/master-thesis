function [U, c1, c2] = KFEAC(U0, Img, c1, c2, mu, m, epsilon, timestep, sigma)

U=U0;
    
K=curvature_central(U);
  
DrcU=(epsilon/pi)./(epsilon^2.+(U).^2);               

[c1 ,c2] = UpdateC(U, Img, c1, c2, sigma, m);    
    
EnergyTerm = m.*((1-U).^(m-1)).*(1-KERNEL(Img,c2,sigma))-m.*(U.^(m-1)).*(1-KERNEL(Img,c1,sigma));
        
LengthTerm = mu.*DrcU.*K;
    
U = U + timestep*(EnergyTerm + LengthTerm);
    
U = U.*(U<=1)+(U>1);
U = U.*(U>=0);

U=NeumannBoundCond(U);

function [c1 ,c2] = UpdateC(U, Img, v1, v2, sigma, m)

Tmp1 = (U.^m).*Img.*KERNEL(Img,v1,sigma);
Tmp2 = ((1-U).^m).*Img.*KERNEL(Img,v2,sigma);

De1 = (U.^m).*KERNEL(Img,v1,sigma);
De2 = ((1-U).^m).*KERNEL(Img,v2,sigma);

c1 = sum(Tmp1(:))/sum(De1(:));
c2 = sum(Tmp2(:))/sum(De2(:));

function g = NeumannBoundCond(f)
% Neumann boundary condition
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  

function k = curvature_central(u)                       
% compute curvature
[ux,uy] = gradient(u);                                  
normDu = sqrt(ux.^2+uy.^2+1e-8);                       % the norm of the gradient plus a small possitive number 
                                                        % to avoid division by zero in the following computation.
Nx = ux./normDu;                                       
Ny = uy./normDu;
[nxx,~] = gradient(Nx);                              
[~,nyy] = gradient(Ny);                              
k = nxx+nyy;     

function value = KERNEL(x,v,sigma)

value = exp(-1.*((x-v).^2)/sigma);