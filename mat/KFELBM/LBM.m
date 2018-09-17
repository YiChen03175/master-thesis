%
% This matlab file provides function for updating membership function
% by using Lattice Boltzmann Method (LBM) in CPU version KFELBM.
%
% Execution version: MATLAB R2016a
%
% NOTE: This function works as same as the kernel on GPU version.

function [U, fparticle] = LBM(U0, fparticle, F, lambda, time) 

[row, col] = size(F);
tmp = zeros(size(fparticle));
feq = zeros(size(fparticle));

% calculate balanced lattices
feq(:,:,1) = U0 .* (4/9);
feq(:,:,2) = U0 .* (1/9);
feq(:,:,3) = U0 .* (1/9);
feq(:,:,4) = U0 .* (1/9);
feq(:,:,5) = U0 .* (1/9);
feq(:,:,6) = U0 .* (1/36);
feq(:,:,7) = U0 .* (1/36);
feq(:,:,8) = U0 .* (1/36);
feq(:,:,9) = U0 .* (1/36);

% calculate distribution by BGK model
tmp(:,:,1) = fparticle(:,:,1) - (1/time).*(fparticle(:,:,1)-feq(:,:,1));
tmp(:,:,2) = fparticle(:,:,2) - (1/time).*(fparticle(:,:,2)-feq(:,:,2));
tmp(:,:,3) = fparticle(:,:,3) - (1/time).*(fparticle(:,:,3)-feq(:,:,3));
tmp(:,:,4) = fparticle(:,:,4) - (1/time).*(fparticle(:,:,4)-feq(:,:,4));
tmp(:,:,5) = fparticle(:,:,5) - (1/time).*(fparticle(:,:,5)-feq(:,:,5));
tmp(:,:,6) = fparticle(:,:,6) - (1/time).*(fparticle(:,:,6)-feq(:,:,6));
tmp(:,:,7) = fparticle(:,:,7) - (1/time).*(fparticle(:,:,7)-feq(:,:,7));
tmp(:,:,8) = fparticle(:,:,8) - (1/time).*(fparticle(:,:,8)-feq(:,:,8));
tmp(:,:,9) = fparticle(:,:,9) - (1/time).*(fparticle(:,:,9)-feq(:,:,9));

% add external force to particle distribution
tmp(:,:,2) = tmp(:,:,2) + ((2*time-1)/(2*time))*(1/3)*lambda.*F;
tmp(:,:,3) = tmp(:,:,3) + ((2*time-1)/(2*time))*(1/3)*lambda.*F;
tmp(:,:,4) = tmp(:,:,4) + ((2*time-1)/(2*time))*(1/3)*lambda.*F;
tmp(:,:,5) = tmp(:,:,5) + ((2*time-1)/(2*time))*(1/3)*lambda.*F;
tmp(:,:,6) = tmp(:,:,6) + ((2*time-1)/(2*time))*(1/16.9706)*lambda.*F;
tmp(:,:,7) = tmp(:,:,7) + ((2*time-1)/(2*time))*(1/16.9706)*lambda.*F;
tmp(:,:,8) = tmp(:,:,8) + ((2*time-1)/(2*time))*(1/16.9706)*lambda.*F;
tmp(:,:,9) = tmp(:,:,9) + ((2*time-1)/(2*time))*(1/16.9706)*lambda.*F;

% distribution diffusion
for i=2:row-1
    for j=2:col-1
        fparticle(i,j,1) = tmp(i,j,1);
        fparticle(i,j,2) = tmp(i-1,j,2);
        fparticle(i,j,3) = tmp(i,j+1,3);
        fparticle(i,j,4) = tmp(i+1,j,4);
        fparticle(i,j,5) = tmp(i,j-1,5);
        fparticle(i,j,6) = tmp(i-1,j-1,6);
        fparticle(i,j,7) = tmp(i-1,j+1,7);
        fparticle(i,j,8) = tmp(i+1,j+1,8);
        fparticle(i,j,9) = tmp(i+1,j-1,9);
    end
end

% update membership function by summing nine lattices 
U = sum(fparticle,3);

% make sure no excced value
U = U.*(U<=1)+(U>1);
U = U.*(U>=0);

% Neumann boundary condition
U = NeumannBoundCond(U);

function g = NeumannBoundCond(f)

[nrow,ncol] = size(f);
g = f;

g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  
    
