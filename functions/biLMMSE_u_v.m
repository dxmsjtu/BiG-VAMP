function [MEAN, VAR, mu_ext,gamma_ext] = biLMMSE_u_v(A,B,mu_gauss, gamma_gauss)
% [1]vector approximate message passing with arbitrary i.i.d. noise priors
% [2]Bilinear Generalized Vector Approximate Message Passing
% [3] Vector Approximate Message Passing
   K = size(mu_gauss,1);
   % posterior message
   VAR=inv(diag(gamma_gauss).*eye(size(A))+A);
   MEAN=(B+mu_gauss.*repmat(gamma_gauss, K, 1))*VAR;
   
   % extrinsic message
   eta = (1./diag(VAR))';
   gamma_ext = (eta-gamma_gauss);% line (17) of Alg. 1 of [1] or line 14 of Alg3. of [3]
   gamma_ext =max(gamma_ext,1e-5);
   mu_ext = (repmat(eta,K,1) .* MEAN - repmat(gamma_gauss,K,1) .* mu_gauss)./ repmat(gamma_ext,K,1);
   % line (16) of Alg. 2 of [2] or  line (19) of Alg. 1 of [1] or line 15 of Alg3. of [3]
end
% function [mu_ext,gamma_ext,MEAN] = LMMSE_z(u, u_var, v, v_var, mu_gauss, gamma_gauss, beta)
% % [1]vector approximate message passing with arbitrary i.i.d. noise priors
% % [2]Bilinear Generalized Vector Approximate Message Passing
%    [m, n]= size(mu_gauss);   
%    % compute the extrinsic variables
%    MEAN = u * v'+trace((1/beta) * u_var * v_var')*mu_gauss* gamma_gauss;
%    VAR = trace((1/beta) * u_var * v_var'*m*n + u_var * (v'*v)*m + v_var * (u'*u) * n )/m/n;   
%    % compute the extrinsic variables
%    eta = (1./VAR)+gamma_gauss;
%    gamma_ext = eta-gamma_gauss; % line (17) of Alg. 1 of [1]
%    gamma_ext=max(gamma_ext,1e-11);   
%    mu_ext = (MEAN*eta-gamma_gauss*mu_gauss)/gamma_ext; % line (19) of Alg. 1 of [1]
% end



