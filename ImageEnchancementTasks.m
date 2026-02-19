clc;
close all;
clear;
% Image
Img_original=rgb2gray(imread('peppers.png'));
LowExposureImg=Img_original-200;
Img=LowExposureImg;
imshow(Img);
title('Original Image');
figure;
imshow(LowExposureImg);
title('Low Exposure Image');

% Histogram
[h_k,k]=imhist(Img);
k_I=[1:256];
[L,indx]=max(Img(:));  % maximum Intensity in Image
figure;
plot(k,h_k);
exposure=sum(h_k(1:L+1).*k(1:L+1))/sum(h_k(1:L+1));

Xa=L*(1-exposure);  %check exposure;
Tc=(1/L)*sum(h_k(1:L+1));
hc_k=h_k;
hc_k(h_k>=Tc)=Tc;
Xal=L*[(Xa/L)-sum(h_k(1:Xa).k(1:Xa))/h_k(1:Xa)];
Xau=L*[1+(Xa/L)-sum(h_k(1:L+1).k(1:Xa))/(L*h_k(Xa:L+1))];

NLL=sum(h_k(1:Xa));
NLU=sum(h_k(Xal:Xa));
NUL=sum(h_k(Xa:Xau));
NUU=sum(h_k(Xau:L+1));

PLL=h_c(1:Xal)/NLL;
PLU=h_c(Xal:Xa)/NLU;
PUL=h_c(xa:xau)/NUL;
PUU=h_c(Xau:L+1)/NUU;

CLl=sum(PLL(1:Xal));
CLu=sum(PLU(Xal:Xa));
CUl=sum(PUL(Xa:Xau));
CUu=sum(PUU(Xau:L+1));

%% Recursive Exposure Sub image Histogram equalization(RESIHE)
clc;
close all;
clear;
% Image
Img_original=rgb2gray(imread('peppers.png'));
imshow(Img_original);
title('Original Image');
LowExposureImg=Img_original-50;
Img=LowExposureImg;
% Img=Img_original;
figure;
imshow(LowExposureImg);
title('Low Exposure Image');



T=0.01;
Exposure_difference=1;
prev_exposure=0;
Fs=[];
while Exposure_difference>T
    Img=Img_original-randi([50,51],[1,1]);

    [h_k,k]=imhist(Img);
    L=256;
    % figure;
    % plot(k,h_k);
    exposure=sum(h_k(1:L).*k(1:L))/(L*sum(h_k(1:L)));
    Xa=floor(L*(1-exposure));
    Tc=(1/L)*sum(h_k(1:L));
    hc_k=h_k;
    hc_k(h_k>=Tc)=Tc;

    %k=0:Xa-1
    NL=sum(h_k(1:Xa));
    NU=sum(h_k(Xa+1:end));
    PL_k=h_k(1:Xa)/NL;
    PU_k=h_k(Xa+1:end)/NU;
    CL_k=cumsum(PL_k);
    CU_k=cumsum(PU_k);
    WL=double(Img).*(Img<Xa); %Low exposure image
    WU=double(Img).*(Img>=Xa);%High exposure image
    WU=WU-Xa;
    temp=length(CU_k);
    CU_k(end+1)=temp+1;
    WU(WU<0)=temp;

    FL=Xa*CL_k(WL+1);
    FU=(Xa+1)+(L-Xa+1)*CU_k(WU+1);
    remover=(Xa+1)+(L-Xa+1)*(temp+1);
    FU(FU==remover)=0;
    F=FL+FU;
    Fs(:,:,end+1)=F;
    % figure;imshow(uint8(F));
    % surfl(Img);
    % figure;surfl(uint8(F))
    Exposure_difference=abs(exposure-prev_exposure);
    prev_exposure=exposure;
end

figure;imshow(uint8(max(Fs,[],3)));

%% Recursively Separate exposure based sub Image histogram equalization(RS-ESIHE)
clc;
close all;
clear;
% Image
Img_original=rgb2gray(imread('peppers.png'));
imshow(Img_original);
title('Original Image');
LowExposureImg=Img_original-20;
Img=LowExposureImg;
% Img=Img_original;
figure;
imshow(LowExposureImg);
title('Low Exposure Image');

r=2;
[h_k,k]=imhist(Img);
L=256;
% figure;
% plot(k,h_k);
exposure=sum(h_k(1:L).*k(1:L))/(L*sum(h_k(1:L)));
Xa=floor(L*(1-exposure));
Tc=(1/L)*sum(h_k(1:L));
hc_k=h_k;
hc_k(h_k>=Tc)=Tc;


%k=0:Xa-1
Xal=round(L*(  (Xa/L) -(sum( h_k(1:Xa).*k(1:Xa) )/(L*sum(h_k(1:Xa)))) )); %Reculating expsoure threshold in under exposure image
Xau=round(L*(1 + (Xa/L)-(sum( h_k(Xa+1:L).*k(Xa+1:L) )/(L*sum( h_k(Xa+1:L)))))); %Reculating expsoure threshold in Over exposure image
% Count of pixels in each sub image
NLl=sum(h_k(1:Xal));
NLu=sum(h_k(Xal+1:Xa));
NUl=sum(h_k(Xa+1:Xau));
NUu=sum(h_k(Xau+1:L));
% PDF of each sub image
PLl_k=hc_k(1:Xal)/NLl;
PLu_k=hc_k(Xal+1:Xa)/NLu;
PUl_k=hc_k(Xa+1:Xau)/NUl;
PUu_k=hc_k(Xau+1:L)/NUu;
% CDF od each sub image
CLl_k=cumsum(PLl_k);
CLu_k=cumsum(PLu_k);
CUl_k=cumsum(PUl_k);
CUu_k=cumsum(PUu_k);
%Sub images
W=zeros(size(Img));
% 1st subimage
WLl=double(Img).*(Img<Xal); % 0 to Xal-1
W=W+WLl;
subplot(5,2,1);imshow(uint8(WLl))
title('WLl');
% 2nd Subimage
WLu=double(Img).*(Img>Xal & Img<Xa); % xal to Xa-1
W=W+WLu;
subplot(5,2,3);imshow(uint8(WLu))
title('WLu');
WLu=WLu-Xal;
temp_Lu=length(CLu_k);
CLu_k(end+1)=temp_Lu+1;
WLu(WLu<0)=temp_Lu;
% 3rd Subimage
WUl=double(Img).*(Img>Xa & Img<Xau);
W=W+WUl;
subplot(5,2,5);imshow(uint8(WUl))
title('WUl');
WUl=WUl-Xa;
temp_Ul=length(CUl_k);
CUl_k(end+1)=temp_Ul+1;
WUl(WUl<0)=temp_Ul;
% 4th Subimage
WUu=double(Img).*(Img>Xau & Img<L);
W=W+WUu;
subplot(5,2,7);imshow(uint8(WUu))
title('WUu');
WUu=WUu-Xau;
temp_Uu=length(CUu_k);
CUu_k(end+1)=temp_Uu+1;
WUu(WUu<0)=temp_Uu;
subplot(5,2,9);imshow(uint8(W));title('W');

% Histogram Equalized Images
F=zeros(size(Img));
% HE of 1st subimage
FLl=Xal*CLl_k(WLl+1);
% HE of 2nd subimage
FLu=(Xal+1)+(Xa-Xal+1)*CLu_k(WLu+1);
remover=(Xal+1)+(Xa-Xal+1)*(temp_Lu+1);
FLu(FLu==remover)=0;
% HE of 3rd subimage
FUl=(Xa+1)+(Xau-Xa+1)*CUl_k(WUl+1);
remover=(Xa+1)+(Xau-Xa+1)*(temp_Ul+1);
FUl(FUl==remover)=0;
% HE of 4th subimage
FUu=(Xau+1)+(L-Xau+1)*CUu_k(WUu+1);
remover=(Xau+1)+(L-Xau+1)*(temp_Uu+1);
FUu(FUu==remover)=0;
% Viusaling Images
subplot(5,2,2);imshow(uint8(FLl))
title('FLl');
subplot(5,2,4);imshow(uint8(FLu))
title('FLu');
subplot(5,2,6);imshow(uint8(FUl))
title('FUl');
subplot(5,2,8);imshow(uint8(FUu))
title('FUu');
F=FLl+FLu+FUl+FUu;
subplot(5,2,10);imshow(uint8(F));title('F');

%%  RS-ESIHE(changes in threshold) 
clc;
close all;
clear;
% Image
Img_original=rgb2gray(imread('peppers.png'));
imshow(Img_original);
title('Original Image');
LowExposureImg=Img_original-20;
Img=LowExposureImg;
% Img=Img_original;
figure;
imshow(LowExposureImg);
title('Low Exposure Image');

r=2;

[h_k,k]=imhist(Img);
L=256;
% figure;
% plot(k,h_k);
exposure=sum(h_k(1:L).*k(1:L))/(L*sum(h_k(1:L)));
Xa=floor(L*(1-exposure));

%k=0:Xa-1
Xal=round(L*(  (Xa/L) -(sum( h_k(1:Xa).*k(1:Xa) )/(L*sum(h_k(1:Xa)))) )); %Reculating expsoure threshold in under exposure image
Xau=round(L*(1 + (Xa/L)-(sum( h_k(Xa+1:L).*k(Xa+1:L) )/(L*sum( h_k(Xa+1:L)))))); %Reculating expsoure threshold in Over exposure image
NL=Xa;
NU=L-Xa;
Tc=(1/L)*sum(h_k(1:L));
TL=(1/NL)*sum(h_k(1:Xa));
TU=(1/NU)*sum(h_k(Xa+1:L));
hc_k=h_k;
hc_k(h_k>=TL & h_k<Tc)=TL;
%hc_k(h_k>=Tc & h_k<TU) %leave it same
hc_k(hc_k>=TU)=TU;
% Count of pixels in each sub image
NLl=sum(h_k(1:Xal));
NLu=sum(h_k(Xal+1:Xa));
NUl=sum(h_k(Xa+1:Xau));
NUu=sum(h_k(Xau+1:L));
% PDF of each sub image
PLl_k=hc_k(1:Xal)/NLl;
PLu_k=hc_k(Xal+1:Xa)/NLu;
PUl_k=hc_k(Xa+1:Xau)/NUl;
PUu_k=hc_k(Xau+1:L)/NUu;
% CDF od each sub image
CLl_k=cumsum(PLl_k);
CLu_k=cumsum(PLu_k);
CUl_k=cumsum(PUl_k);
CUu_k=cumsum(PUu_k);
%Sub images
W=zeros(size(Img));
% 1st subimage
WLl=double(Img).*(Img<Xal); % 0 to Xal-1
W=W+WLl;
subplot(5,2,1);imshow(uint8(WLl))
title('WLl');
% 2nd Subimage
WLu=double(Img).*(Img>Xal & Img<Xa); % xal to Xa-1
W=W+WLu;
subplot(5,2,3);imshow(uint8(WLu))
title('WLu');
WLu=WLu-Xal;
temp_Lu=length(CLu_k);
CLu_k(end+1)=temp_Lu+1;
WLu(WLu<0)=temp_Lu;
% 3rd Subimage
WUl=double(Img).*(Img>Xa & Img<Xau);
W=W+WUl;
subplot(5,2,5);imshow(uint8(WUl))
title('WUl');
WUl=WUl-Xa;
temp_Ul=length(CUl_k);
CUl_k(end+1)=temp_Ul+1;
WUl(WUl<0)=temp_Ul;
% 4th Subimage
WUu=double(Img).*(Img>Xau & Img<L);
W=W+WUu;
subplot(5,2,7);imshow(uint8(WUu))
title('WUu');
WUu=WUu-Xau;
temp_Uu=length(CUu_k);
CUu_k(end+1)=temp_Uu+1;
WUu(WUu<0)=temp_Uu;
subplot(5,2,9);imshow(uint8(W));title('W');

% Histogram Equalized Images
F=zeros(size(Img));
% HE of 1st subimage
FLl=Xal*CLl_k(WLl+1);
% HE of 2nd subimage
FLu=(Xal+1)+(Xa-Xal+1)*CLu_k(WLu+1);
remover=(Xal+1)+(Xa-Xal+1)*(temp_Lu+1);
FLu(FLu==remover)=0;
% HE of 3rd subimage
FUl=(Xa+1)+(Xau-Xa+1)*CUl_k(WUl+1);
remover=(Xa+1)+(Xau-Xa+1)*(temp_Ul+1);
FUl(FUl==remover)=0;
% HE of 4th subimage
FUu=(Xau+1)+(L-Xau+1)*CUu_k(WUu+1);
remover=(Xau+1)+(L-Xau+1)*(temp_Uu+1);
FUu(FUu==remover)=0;
% Viusaling Images
subplot(5,2,2);imshow(uint8(FLl))
title('FLl');
subplot(5,2,4);imshow(uint8(FLu))
title('FLu');
subplot(5,2,6);imshow(uint8(FUl))
title('FUl');
subplot(5,2,8);imshow(uint8(FUu))
title('FUu');
F=FLl+FLu+FUl+FUu;
subplot(5,2,10);imshow(uint8(F));title('F');

%% Recursive Exposure Sub image Histogram equalization(RESIHE) Working
clc;
close all;
clear;
% Image
Img_original=rgb2gray(imread('LowLight4.png'));
imshow(Img_original);
title('Original Image');
Img=Img_original;

T=0.01;
Exposure_difference=1;
prev_exposure=0;
Fs=[];
exposures=[];
while Exposure_difference>T
[h_k,k]=imhist(Img);
L=256;
max_img=double(max(Img(:)));
% L=max_img;
figure;
plot(k,h_k);
exposure=sum(h_k(1:L).*k(1:L))/(L*sum(h_k(1:L)));
exposures=[exposures,exposure];
Xa=floor(max_img*(1-exposure));
Tc=(1/max_img)*sum(h_k(1:L));
hc_k=h_k;
hc_k(h_k>=Tc)=Tc;

k=0:Xa-1;
NL=sum(hc_k(1:Xa));
NU=sum(hc_k(Xa+1:L));
PL_k=hc_k(1:Xa)/NL;
PU_k=hc_k(Xa+1:L)/NU;
CL_k=cumsum(PL_k);
CU_k=cumsum(PU_k);
WL=double(Img).*(Img<Xa); %Low exposure image
WU=double(Img).*(Img>=Xa);%High exposure image
WU=WU-Xa;
temp=length(CU_k);
CU_k(end+1)=temp+1;
WU(WU<0)=temp;

FL=Xa*CL_k(WL+1);
FU=(Xa+1)+(L-Xa+1)*CU_k(WU+1);
remover=(Xa+1)+(L-Xa+1)*(temp+1);
FU(FU==remover)=0;
F=FL+FU;
% figure;
% imshow(uint8(F));
Img=uint8(F);
Fs(:,:,end+1)=F;
    % figure;imshow(uint8(F));
    % surfl(Img);
    % figure;surfl(uint8(F))
    Exposure_difference=abs(exposure-prev_exposure);
    prev_exposure=exposure;
end

figure;
imshow(uint8(F));
%% Recursively Separate exposure based sub Image histogram equalization(RS-ESIHE)
clc;
close all;
clear;
% Image
Img=rgb2gray(imread('LowLight4.png'));
imshow(Img);
title('Low Exposure Image');


T=0.01;
Exposure_difference=1;
prev_exposure=0;
Fs=[];
exposures=[];
% while Exposure_difference>T
for i=1:1
r=2;
[h_k,k]=imhist(Img);
L=double(max(Img(:)));
exposure=sum(h_k(1:L).*k(1:L))/(L*sum(h_k(1:L)));
exposures=[exposures,exposure];
Xa=floor(L*(1-exposure));
Tc=(1/L)*sum(h_k(1:L));
hc_k=h_k;
hc_k(h_k>=Tc)=Tc;


%k=0:Xa-1
Xal=round(L*(  (Xa/L) -(sum( hc_k(1:Xa).*k(1:Xa) )/(L*sum(hc_k(1:Xa)))) )); %Reculating expsoure threshold in under exposure image
Xau=round(L*(1 + (Xa/L)-(sum( hc_k(Xa+1:L).*k(Xa+1:L) )/(L*sum( hc_k(Xa+1:L)))))); %Reculating expsoure threshold in Over exposure image
% Count of pixels in each sub image
NLl=sum(hc_k(1:Xal));
NLu=sum(hc_k(Xal+1:Xa));
NUl=sum(hc_k(Xa+1:Xau));
NUu=sum(hc_k(Xau+1:L));
% PDF of each sub image
PLl_k=hc_k(1:Xal)/NLl;
PLu_k=hc_k(Xal+1:Xa)/NLu;
PUl_k=hc_k(Xa+1:Xau)/NUl;
PUu_k=hc_k(Xau+1:L)/NUu;
% CDF od each sub image
CLl_k=cumsum(PLl_k);
CLu_k=cumsum(PLu_k);
CUl_k=cumsum(PUl_k);
CUu_k=cumsum(PUu_k);
%Sub images
W=zeros(size(Img));
% 1st subimage
WLl=double(Img).*(Img<Xal); % 0 to Xal-1
W=W+WLl;
% subplot(5,2,1);imshow(uint8(WLl))
% title('WLl');
% 2nd Subimage
WLu=double(Img).*(Img>Xal & Img<Xa); % xal to Xa-1
W=W+WLu;
% subplot(5,2,3);imshow(uint8(WLu))
% title('WLu');
WLu=WLu-Xal;
temp_Lu=length(CLu_k);
CLu_k(end+1)=temp_Lu+1;
WLu(WLu<0)=temp_Lu;
% 3rd Subimage
WUl=double(Img).*(Img>Xa & Img<Xau);
W=W+WUl;
% subplot(5,2,5);imshow(uint8(WUl))
% title('WUl');
WUl=WUl-Xa;
temp_Ul=length(CUl_k);
CUl_k(end+1)=temp_Ul+1;
WUl(WUl<0)=temp_Ul;
% 4th Subimage
WUu=double(Img).*(Img>Xau & Img<L);
W=W+WUu;
% subplot(5,2,7);imshow(uint8(WUu))
% title('WUu');
WUu=WUu-Xau;
temp_Uu=length(CUu_k);
CUu_k(end+1)=temp_Uu+1;
WUu(WUu<0)=temp_Uu;
% subplot(5,2,9);imshow(uint8(W));title('W');

% Histogram Equalized Images
F=zeros(size(Img));
% HE of 1st subimage
FLl=Xal*CLl_k(WLl+1);
% HE of 2nd subimage
FLu=(Xal+1)+(Xa-Xal+1)*CLu_k(WLu+1);
remover=(Xal+1)+(Xa-Xal+1)*(temp_Lu+1);
FLu(FLu==remover)=0;
% HE of 3rd subimage
FUl=(Xa+1)+(Xau-Xa+1)*CUl_k(WUl+1);
remover=(Xa+1)+(Xau-Xa+1)*(temp_Ul+1);
FUl(FUl==remover)=0;
% HE of 4th subimage
FUu=(Xau+1)+(L-Xau+1)*CUu_k(WUu+1);
remover=(Xau+1)+(L-Xau+1)*(temp_Uu+1);
FUu(FUu==remover)=0;
% Viusaling Images
% subplot(5,2,2);imshow(uint8(FLl))
% title('FLl');
% subplot(5,2,4);imshow(uint8(FLu))
% title('FLu');
% subplot(5,2,6);imshow(uint8(FUl))
% title('FUl');
% subplot(5,2,8);imshow(uint8(FUu))
% title('FUu');
F=FLl+FLu+FUl+FUu;
% subplot(5,2,10);imshow(uint8(F));title('F');
% Img=uint8(F);
Fs(:,:,end+1)=F;
% figure;imshow(uint8(F));
% surfl(Img);
% figure;surfl(uint8(F))
Exposure_difference=abs(exposure-prev_exposure);
prev_exposure=exposure;
Img=uint8(F);
end
figure;
imshow(uint8(F));







