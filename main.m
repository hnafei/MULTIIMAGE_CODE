% 
% 2DCS scheme: 2D Compressive Sensing for multilevel image encryption.
% 

clear all;
clc;
close all;

addpath('C:\Users\39302\Desktop\multi-image-master\2DCS_multi_image\pic'); 

addpath('C:\Users\39302\Desktop\multi-image-master\2DCS_multi_image\WaveletSoftware');      

addpath('C:\Users\39302\Desktop\multi-image-master\2DCS_multi_image\work');              

%% (A) obtain the information of the test image


execflag = 2;
fprintf('start cs sampling and encryption:\n');
tic
if execflag == 1
% 1) test images from pic folder  
 
%  filename = 'face.bmp';  %lena very good  35.01  group1
%  filename1 =  'peppers256.bmp'; % very good 33.6779 group1
%  filename2 =  'Tiffany256.bmp'; %  very good 34.4710 group 1
 
 filename =  'Masuda256.bmp'; %  very good 38.0335 group2
 filename1 =  'girl256.bmp'; %  very good 34.8184  group2
 filename2 =  'oldman256.bmp'; %   good 36.0030  group2

 % 2） read image
 original_image = imread(filename);
 original_image1 = imread(filename1);  
 original_image2 = imread(filename2);

 % 3) change RGB image to gray image
 original_image = rgb2gray(original_image);
 original_image1 = rgb2gray(original_image1);
 original_image2 = rgb2gray(original_image2);

 original_image = double(original_image);
 original_image1 = double(original_image1);
 original_image2 = double(original_image2);

 % 4) obtain the  rows and  columns of the image,所有图像大小都一样目前都是256x256
  [high, width] = size(original_image);

 % 5) Parameters used for compression sampling and reconstructiong
  CR = 0.5;                  %  Compression ratio（CR）or called sampling ratio（SR)
  quantizer_bitdepth = 6;
  num_levels = 3;            %  Wavelet decomposition level    
 
 %% (B) privacy-sensitive region process
 % group 1
% 1.1) peppers256.bmp 敏感区域的位置信息
%   Loc_a.x1 = 106;
%   Loc_a.y1 = 78;
%   Loc_a.x2 = 211;
%   Loc_a.y2 = 183;
  Loc_a.x1 = 100;
  Loc_a.y1 = 105;
  Loc_a.x2 = 185;
  Loc_a.y2 = 190;
  
  % 1.2) Tiffany256 敏感区域的位置信息
%   Loc_b.x1 = 102;
%   Loc_b.y1 = 55;
%   Loc_b.x2 = 220;
%   Loc_b.y2 = 183;
  Loc_b.x1 = 106;
  Loc_b.y1 = 78;
  Loc_b.x2 = 211;
  Loc_b.y2 = 183;
 
  %  1.3 d.256 敏感区域的位置信息  
  Loc_d.x1 = 85;
  Loc_d.y1 = 55;
  Loc_d.x2 = 185;
  Loc_d.y2 = 155;
  
  % group 2
  % 1.1) Masuda256.bmp 敏感区域的位置信息
  Loc_e.x1 = 86;
  Loc_e.y1 = 84;
  Loc_e.x2 = 187;
  Loc_e.y2 = 185;
  
  % 1.2 girl256.bmp 敏感区域的位置信息
  Loc_c.x1 = 105;
  Loc_c.y1 = 88;
  Loc_c.x2 = 205;
  Loc_c.y2 = 198;
 
  
  % 1.3 oldman256.bmp 敏感区域的位置信息
%   Loc_f.x1 = 86;
%   Loc_f.y1 = 84;
%   Loc_f.x2 = 187;
%   Loc_f.y2 = 185;
  Loc_f.x1 = 86;
  Loc_f.y1 = 53;
  Loc_f.x2 = 176;
  Loc_f.y2 = 143;
 
   % 2) 不同图像隐私区域加密
   original_image_first = original_image;
   original_image1_first = original_image1;
   original_image2_first = original_image2;
   % 2.1) 处理filename condition
   if strcmp(filename, 'girl256.bmp') == 1
      for mm =  Loc_c.y1:Loc_c.y2
          for nn = Loc_c.x1:Loc_c.x2
           original_image_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename, 'peppers256.bmp') == 1
      for mm =  Loc_a.y1:Loc_a.y2
          for nn = Loc_a.x1:Loc_a.x2
           original_image_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename, 'face.bmp') == 1
      for mm =  Loc_d.y1:Loc_d.y2
          for nn = Loc_d.x1:Loc_d.x2
           original_image_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename, 'Tiffany256.bmp') == 1
      for mm =  Loc_b.y1:Loc_b.y2
          for nn = Loc_b.x1:Loc_b.x2
           original_image_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename, 'Masuda256.bmp') == 1
      for mm =  Loc_e.y1:Loc_e.y2
          for nn = Loc_e.x1:Loc_e.x2
           original_image_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename, 'oldman256.bmp') == 1
        for mm =  Loc_f.y1:Loc_f.y2
          for nn = Loc_f.x1:Loc_f.x2
           original_image_first(mm,nn)  = 0; 
          end
       end
   end
   % 2.2) 处理filename1 condition
   if strcmp(filename1, 'girl256.bmp') == 1
      for mm =  Loc_c.y1:Loc_c.y2
          for nn = Loc_c.x1:Loc_c.x2
           original_image1_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename1, 'peppers256.bmp') == 1
      for mm =  Loc_a.y1:Loc_a.y2
          for nn = Loc_a.x1:Loc_a.x2
           original_image1_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename1, 'face.bmp') == 1
      for mm =  Loc_d.y1:Loc_d.y2
          for nn = Loc_d.x1:Loc_d.x2
           original_image1_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename1, 'Tiffany256.bmp') == 1
      for mm =  Loc_b.y1:Loc_b.y2
          for nn = Loc_b.x1:Loc_b.x2
           original_image1_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename1, 'Masuda256.bmp') == 1
      for mm =  Loc_e.y1:Loc_e.y2
          for nn = Loc_e.x1:Loc_e.x2
           original_image1_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename1, 'oldman256.bmp') == 1
        for mm =  Loc_f.y1:Loc_f.y2
          for nn = Loc_f.x1:Loc_f.x2
           original_image1_first(mm,nn)  = 0; 
          end
       end
   end
   % 2.3) 处理filename2 condition
   if strcmp(filename2, 'girl256.bmp') == 1
      for mm =  Loc_c.y1:Loc_c.y2
          for nn = Loc_c.x1:Loc_c.x2
           original_image2_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename2, 'peppers256.bmp') == 1
      for mm =  Loc_a.y1:Loc_a.y2
          for nn = Loc_a.x1:Loc_a.x2
           original_image2_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename2, 'face.bmp') == 1
      for mm =  Loc_d.y1:Loc_d.y2
          for nn = Loc_d.x1:Loc_d.x2
           original_image2_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename2, 'Tiffany256.bmp') == 1
      for mm =  Loc_b.y1:Loc_b.y2
          for nn = Loc_b.x1:Loc_b.x2
           original_image2_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename2, 'Masuda256.bmp') == 1
      for mm =  Loc_e.y1:Loc_e.y2
          for nn = Loc_e.x1:Loc_e.x2
           original_image2_first(mm,nn)  = 0; 
          end
      end
   elseif strcmp(filename2, 'oldman256.bmp') == 1
        for mm =  Loc_f.y1:Loc_f.y2
          for nn = Loc_f.x1:Loc_f.x2
           original_image2_first(mm,nn)  = 0; 
          end
       end
   end
   
   % 3) generate invertible elementary matrix 
   % original_image_first = original_image*P1;
   % original_image1_first = original_image1*P2;
   % original_image2_first = original_image2*P3;
   
   %%%%%%%%%%%% multipe image test
%    original_image_first = imread('e256.bmp');
%    original_image_first = rgb2gray(original_image_first);
%    original_image_first = double(original_image_first);
   %%%%%%%%%%%
   
   P1 = pinv(original_image'*original_image)*original_image'*original_image_first;
   %P1 = original_image_first*original_image'*pinv(original_image*original_image');
   P2 = pinv(original_image1'*original_image1)*original_image1'*original_image1_first;
   P3 = pinv(original_image2'*original_image2)*original_image2'*original_image2_first;
elseif execflag == 2
    
     % added by hxf 2023 6 8 begin used for entropy test
%       filename = 'face.bmp';  %lena very good  35.01  group1
%       filename1 =  'peppers256.bmp'; % very good 33.6779 group1
%       filename2 =  'Tiffany256.bmp'; %  very good 34.4710 group 1
%       filename3 =  'Masuda256.bmp'; %  very good 38.0335 group2
%       filename4 =  'girl256.bmp'; %  very good 34.8184  group2
%       filename5 =  'oldman256.bmp'; %   good 36.0030  group2
     % added by hxf 2023 6 8 end
    
     filename =  'zelda256.bmp'; %   very good 34.4786 group3
     filename1 =  'house256.bmp'; %  very good 35.9586 group3
     filename2 =  'tmp256.bmp'; %   very good 33.1100  group3
     filename3 =  'd256-gray.bmp'; %  very good 33.8300 gray group3 lena
     filename4 ='anothergirl256.bmp'; % very good 36.7283 group3
     filename5 ='flowers256.bmp'; % very good 33.3780 group3
    %%filename1 =  'david256.bmp'; %  very good 32.6572 group3 tmp
    
    % added by hxf 2023 6 13 512x512 begin
%      filename =  'Zelda512.bmp'; %   very good 34.4786 group3
%      filename1 =  'lena512.bmp'; %  very good 35.9586 group3
%      filename2 =  'masuda512.bmp'; %   very good 33.1100  group3
%      filename3 =  'pepper512.bmp'; %  very good 33.8300 gray group3 lena
%      filename4 ='tiffany512.bmp'; % very good 36.7283 group3
%      filename5 ='flower512.bmp'; % very good 33.3780 group3
    % added by hxf 2023 6 13 512x512 end

  % 2） read image
  original_image = imread(filename);
  original_image1 = imread(filename1);  
  original_image2 = imread(filename2);
  original_image3 = imread(filename3);
  original_image4 = imread(filename4);  
  original_image5 = imread(filename5);

  % 3) change RGB image to gray image
  original_image = rgb2gray(original_image);
  original_image1 = rgb2gray(original_image1);
  original_image2 = rgb2gray(original_image2);
  original_image3 = rgb2gray(original_image3);
  original_image4 = rgb2gray(original_image4);
  original_image5 = rgb2gray(original_image5);
 
  original_image = double(original_image);
  original_image1 = double(original_image1);
  original_image2 = double(original_image2);
  original_image3 = double(original_image3);
  original_image4 = double(original_image4);
  original_image5 = double(original_image5);
 
  % 4) obtain the  rows and  columns of the image,所有图像大小都一样目前都是256x256
  [high, width] = size(original_image);

  % 5) Parameters used for compression sampling and reconstructiong
  CR = 0.5;                  %  Compression ratio（CR）or called sampling ratio（SR)
  quantizer_bitdepth = 6;
  num_levels = 3;            %  Wavelet decomposition level    
  
  
  % 6) 不同图像之间建立关系
  original_image_first = original_image3;
  original_image1_first = original_image4;
  original_image2_first = original_image5;
  
   P1 = pinv(original_image'*original_image)*original_image'*original_image_first;
   P2 = pinv(original_image1'*original_image1)*original_image1'*original_image1_first;
   P3 = pinv(original_image2'*original_image2)*original_image2'*original_image2_first;
   
   % used for entroy test begin
%   original_image_first = original_image;
%   original_image1_first = original_image1;
%   original_image2_first = original_image2;
%    P1 = pinv(original_image3'*original_image3)*original_image3'*original_image_first;
%    P2 = pinv(original_image4'*original_image4)*original_image4'*original_image1_first;
%    P3 = pinv(original_image5'*original_image5)*original_image5'*original_image2_first;
   % used for entroy test end
end
 %% (C) 2D compress and sampling 
   
   % added by hxf begin
   % 1) 初始值
   T=width*high;
   x=zeros(1,T);
   y=zeros(1,T);
   z=zeros(1,T);
   
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
    z_matrix = reshape(z,width,high);
    Phi = orth(z_matrix)'; 
   % added by hxf end
   
   % c.1 构成测量矩阵Phi
  % Phi = orth(randn(width, high))';    

   M = round(CR*high);
   Phi = Phi(1:M, :);
   Psi = DwtTransform(width);

   Y1 =  Phi * original_image_first; 
   Y2 =  Phi * original_image1_first; 
   Y3 =  Phi * original_image2_first; 

   % ste visual security test begin
   % Y1 =  Phi * original_image; 
   % ste visual security test end
   
   
   % c.3 为了安全性，对测量值进行二次加密
   % added by hxf begin for another method
   % 1) 初始值
   T=width*high;
   xxx=zeros(1,T);
   yyy=zeros(1,T);
   zzz=zeros(1,T);
   
   xxx(1)=0.6;
   yyy(1)=0.2;
   zzz(1)=0.5;
   a=10;
   w=3;
   % 2) 三维混沌映射
   for i=2:T
    xxx(i)=yyy(i-1)-zzz(i-1);
    yyy(i)=sin(pi*xxx(i-1)-a*yyy(i-1));
    zzz(i)=cos(w*acos(zzz(i-1))+yyy(i-1));
   end
   R3 = reshape(zzz,width,width);
   % added by hxf end  for another method

   % c3.2）进行加密
    Y1= Y1*R3;
    Y2 = Y2*R3;
    Y3 = Y3*R3;
    
    % c.4 index scrambling begin
    [m_high, m_width] = size(Y1);
    Index = randperm (m_high * m_width);

    Y1 = IndexScramble(Index, Y1);
    Y2 = IndexScramble(Index, Y2);
    Y3 = IndexScramble(Index, Y3);
    % c.4 index scrambling end  
    
toc
fprintf('end compression and encryption:\n');

   %% %%%%%%%%%%%%(D) 2D-DCT processing %%%%%%%%%%%%%%%%
fprintf('start embedding :\n');  
tic

   %matlab中， 函数dct2()和函数idct2()分布进行二维DCT变换和二维DCT反变换
   % 1) exec dct2 operation
   Y1D = dct2(Y1);
   Y2D = dct2(Y2);
   Y3D = dct2(Y3);

   %% (E) exec 3DDCT and embedding processing
    % 2) exec reading carrier image
   % added by hxf for 512x512 test begin
   %carrier_filename='yachtcarrier512.bmp' ;    % 512x512
   % added by hxf for 512x512 test end
   
   % carrier_filename='Sailboatcarrier.bmp'; % group3
   carrier_filename='fruitscarrier.bmp';   % group3 tmp
  % carrier_filename='yachtcarrier.bmp' ;    % group2
  % carrier_filename='monarchcarrier.bmp';   % group1
   
  %carrier_filename='house256.bmp'    % group2
   carrier_image = imread(carrier_filename);
   carrier_image = double(carrier_image);
   carrier_image_old = carrier_image; %第一组载波图像
   
   nrow=size(carrier_image,1);
   ncol=size(carrier_image,2);
   
   % 3) exec 3d dct operation
   RR = carrier_image(:,:,1);
   GG = carrier_image(:,:,2);
   BB = carrier_image(:,:,3);
   
   RR = dct2(RR);
   GG = dct2(GG);
   BB = dct2(BB);
   
   image3DDCT_tmp = cat(3, RR, GG, BB);

  
%    image3DDCT_tmp = zeros(nrow,ncol,3);
%    % divide video 8x8x3 block and implement dct
%    for i=0:8:nrow-1
%        % fprintf('exe 3DDCT in send side line: %g \n',i);
%        for j=0:8:ncol-1
%            tmp = carrier_image(i+1:i+8,j+1:j+8,:);
%             X= DCT3D(tmp);
%             image3DDCT_tmp(i+1:i+8,j+1:j+8,:)=X;
%        end
%    end

   DCTR = image3DDCT_tmp(:,:,1); %image3DDCT(:,:,1);
   DCTG = image3DDCT_tmp(:,:,2); %image3DDCT(:,:,2);
   DCTB = image3DDCT_tmp(:,:,3); %image3DDCT(:,:,3);
   % E.2 embedding processing
   tmpR = image3DDCT_tmp(:,:,1);
   tmpG = image3DDCT_tmp(:,:,2);
   tmpB = image3DDCT_tmp(:,:,3);
   N = high;

    %% test begin
     % \lamba的值为0.0001--0.003都可以测试的

   %  0.0001->0.0009-->0.001-->0.002-->0.0025-->0.003
    DCTR(1:M,1:N) = DCTR(1:M,1:N)- Y1D*0.001;
    DCTG(1:M,1:N) = DCTG(1:M,1:N)- Y2D*0.001;
    DCTB(1:M,1:N) = DCTB(1:M,1:N)- Y3D*0.001;

    %% test end
    
    image3DDCT_tmp(:,:,1)=DCTR;
    image3DDCT_tmp(:,:,2)=DCTG;
    image3DDCT_tmp(:,:,3)=DCTB;

  %% (F) i3DCT processing
% %   % number of element saved for each 8x8x8 block
% %   nV=zeros(nrow,ncol,3);
% %   for i=0:8:nrow-1
% %      % fprintf('exec i3DDCT in send side for line: %g \n',i);
% %       for j=0:8:ncol-1
% %        tmp = image3DDCT_tmp(i+1:i+8,j+1:j+8,:);
% %         yx=IDCT3D(tmp);
% %         nV(i+1:i+8,j+1:j+8,:)=yx;
% %     end
% % end
% % 
% %    R =  nV(:,:,1);
% %    G =  nV(:,:,2);
% %    B =  nV(:,:,3);
% %    
% %    carrier_image_tmp = cat(3, R, G, B);  %带有隐藏信息的载波图像
% %    carrier_image_new = nV; 
   
   R = idct2( image3DDCT_tmp(:,:,1));
   G = idct2( image3DDCT_tmp(:,:,2));
   B = idct2( image3DDCT_tmp(:,:,3));
toc
fprintf('end embedding :\n'); 
  carrier_image_tmp = cat(3, R, G, B);  %带有隐藏信息的载波图像
  carrier_image_new = carrier_image_tmp;

  
   %% (G) Transformation through network
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
   %% (G) Transformation through network
   
   %% (H) %%%%%%%%%Decoding process%%%%%%%%%%%%
   
   % step2 exec 3D-DCT process
% %    row=size(carrier_image_new,1);
% %    col=size(carrier_image_new,2);
% % 
% %    image3DDCT_new = zeros(row,col,3);
% %  
% %    % divide video 8x8x3 block and implement dct
% %    for i=0:8:row-1
% %        % fprintf('exe 3DDCT in received side for line: %g \n',i);
% %        for j=0:8:col-1
% %            tmp = carrier_image_new(i+1:i+8,j+1:j+8,:);
% %             XX=DCT3D(tmp);
% %             image3DDCT_new(i+1:i+8,j+1:j+8,:)=XX;
% %        end
% %    end
   
  %% added by hxf begin 613
fprintf('start extracting :\n');
 tic

  RRR = carrier_image_new(:,:,1);
  GGG = carrier_image_new(:,:,2);
  BBB = carrier_image_new(:,:,3);
  
  RRR = dct2(RRR);
  GGG = dct2(GGG);
  BBB = dct2(BBB);
  image3DDCT_new = cat(3, RRR,GGG,BBB);
  %% added by hxf end 613

    % step2 extracting Y1D,Y2D,Y3D information
    DCTR_new = image3DDCT_new(:,:,1);
    DCTG_new = image3DDCT_new(:,:,2);
    DCTB_new = image3DDCT_new(:,:,3);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%    
    Y1D_new = ( DCTR_new(1:M,1:N) - tmpR(1:M,1:N) );
    Y2D_new = ( DCTG_new(1:M,1:N) - tmpG(1:M,1:N) );
    Y3D_new = ( DCTB_new(1:M,1:N) - tmpB(1:M,1:N) );
    
    %% test begin
    
    Y1D_new = Y1D_new*(-1)*(1000);
    Y2D_new = Y2D_new*(-1)*(1000);
    Y3D_new = Y3D_new*(-1)*(1000);
    %% test end
    
    % 2) exec idct2 operation
    Y1_n = idct2(Y1D_new);
    Y2_n = idct2(Y2D_new);
    Y3_n = idct2(Y3D_new);
toc
fprintf('end extracting :\n'); 

%% %%%%(E) Reconstruct images for users with different authorites%%%%%
% initial operation
 param.width = width;
 param.high = high;
 param.num_levels = num_levels;
 %param.distance = distance;
 %param.k = k;
 %param.zz = zz(1);
 param.M = M;
%  param.n1 = n1;
%  param.aa = aa;
%  param.bb = bb;

 fprintf('start description :\n'); 
 tic 
%% userA only with keys Phi, Index tent
% Y1->original_image3, Y2->original_image4, Y3->original_image5
% %   recovered_image = UserASPL(Y1_n, Phi, param, num_levels,Index,Psi);
% % toc;
% % fprintf('end description :\n'); 
% %   imshow(uint8(recovered_image));
 
%% userB with keys Phi, Indexi, tent Pi
 recovered_image = UserBSPL(Y1_n, Phi, param, num_levels,Index,P1,Psi);
toc;
fprintf('end description :\n');  
 imshow(uint8(recovered_image));
 errorx=sum(sum(abs(recovered_image - original_image).^2));   %MSE误差
 psnr=10*log10(255*255/(errorx/high/width))   %  PSNR

 K = [0.01 0.03];
 winsize = 11;
 sigma = 1.5;
 window = fspecial('gaussian', winsize, sigma);
 level = 5;
 weight = [0.0448 0.2856 0.3001 0.2363 0.1333];
 method = 'product';
  %mssim=mssim(carrier_image_old(:,:,1),carrier_image_new(:,:,1),K, window, level, weight, 'product')
mssim = mssim(original_image,recovered_image,K, window, level, weight, 'product')



