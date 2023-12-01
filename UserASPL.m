function RecoveredImage = UserASPL(Y, Phi, param, num_levels,Index,Psi)

  % msgbox('userASPL called begin','信息对话框','help');
  
   % 1) index unscrambing process
   Y = UnIndexScramble(Index, Y);
    
    %% Phi秘钥敏感性测试begin
    T=param.width*param.high;
    x=zeros(1,T);
    y=zeros(1,T);
    z=zeros(1,T);
   
    %x(1)=0.311111111111111;
    x(1)=0.3;
    y(1)=0.4;
    z(1)=0.5;
    a=10;
    w=3;
    % 2) 三维混沌映射
    for i=2:T
     x(i)=y(i-1)-z(i-1);
     y(i)=sin(pi*x(i-1)-a*y(i-1));
     z(i)=cos(w*acos(z(i-1))+y(i-1));
    end
    z_matrix = reshape(z,param.width,param.high);
    Phi = orth(z_matrix)'; 
  
    Phi = Phi(1:param.M, :);
    %% 秘钥敏感性测试end
    
    % 2）Remove R from Y before decryption 
    % 1) 初始值
    T=param.width*param.high;
    xxx=zeros(1,T);
    yyy=zeros(1,T);
    zzz=zeros(1,T);
 
    xxx(1)=0.6;
    yyy(1)=0.2;
    zzz(1)=0.5;
    %zzz(1)=0.511111111111111;
    a=10;
    w=3;
    % 2) 三维混沌映射
    for i=2:T
      xxx(i)=yyy(i-1)-zzz(i-1);
      yyy(i)=sin(pi*xxx(i-1)-a*yyy(i-1));
      zzz(i)=cos(w*acos(zzz(i-1))+yyy(i-1));
    end
    z_sample = zzz;
    R = reshape(z_sample,param.high,param.width);
    Y = Y*pinv(R);

    % 3） execute decryption
    % 3.1）参数初始化
    lambda = 0.8;               %  convergence-control factor
    max_iterations = 200;       %  maximum iteration numbers
    TOL = 0.01;               %  error tolerance
    D_prev = 0;                             
    
    % 3.2） start decryption
    X = Phi'*Y;  % Initial value
    
    % added by hxf begin 2023 6 10
    beta = 0.8;                       %  sigma的衰减因子
    sigma_x = X(:);
    sigma = 2*max(abs( sigma_x ));    %  光滑函数的初始参数
    sigma_min = 0.01;                 %  算法终止的界限
    %sigma_min = 0.0005;
    param.beta = beta;
    param.sigma= sigma;
    param.sigma_min = sigma_min;
    % added by hxf end  2023 6 10
    
    
    for i = 1:max_iterations
        [X, D_cur] = SPLIterationUserA(Y, X, Phi, param, lambda, num_levels,Psi);
        
        % if the error smaller than the given error tolerance, break.
        if ((D_prev ~= 0) && (abs(D_cur - D_prev) < TOL))     
            break;
        end
        D_prev = D_cur; % update D_prev
        % added by hxf begin 6 12 
        if  param.sigma > param.sigma_min
            param.sigma = param.sigma*beta;
        end
        % addd by hxf end  6 12
    end
    
    [X, D_cur] = SPLIterationUserA(Y, X, Phi, param, lambda, num_levels,Psi);
    RecoveredImage = X;

    %msgbox('userASPL called end','信息对话框','help');
end