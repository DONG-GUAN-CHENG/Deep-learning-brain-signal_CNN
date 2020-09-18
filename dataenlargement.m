%% rawdata先切時間
clc;clear
load('thb perfect data')
total=0;
result={};
timeCh=[6000];
targetIndex={};
for currTime=timeCh 
    for combos=22:37 %rawdata要幾個人的
        total=total+1;
        temp=zeros(currTime,52); %currTime,52,3
        Hb=[];
        HbO2=[];
        tHb=[];
        
      % Hb=Whole1(combos).Hb;
       % HbO2=Whole2(combos).HbO2;
         tHb=Whole3(combos).tHb;  %tHb=Test.Whole(combos).tHb;
        
      % temp1(:,:,1)=temp(:,:,1)+Hb(1:currTime,:);
      %temp2(:,:,1)=temp(:,:,1)+HbO2(1:currTime,:);
       temp3(:,:,1)=temp(:,:,1)+tHb(1:currTime,:);
       
      %  result1{total}=temp1;
      % result2{total}=temp2;
        result3{total}=temp3;
       
        plot(temp3)
        hold on
            
      aa1 = combntns(result3,2) %C幾取幾
      %  aa2 = combntns(result2,2)
   %aa3 = combntns(result2,3)
       %aa4 = combntns(result2,4)
%         aa5 = combntns(result,5)
%         aa6 = combntns(result,6)
    end
end

% size(result2)
% Whole2=Whole2'
% aa4=combntns(male_train,1)
% aa4=combntns(female_test,3)
 %aa4=combntns(Whole2,4)
% aa4=combntns(female_test,5)
%%
% % -------------------------------------%
% %    %%Band pass filter (butterworth)
% % --------------------------------------%

total2=0;

   for pw=1:240
        fs=10; % 取樣頻率 Fs = 10 Hz
        Fn=fs/2; %奈奎斯特頻率（Hz）
        
        CF=0.07;  % 截止頻率 0.07 Hz
        Wn=CF/Fn; %  cutoff/nyquistFrequency
        [num,den]=butter(4,Wn,'low'); % Lowpass filter %[b,a] = besself(order,cutoff/nyquistFrequency)
        [h,w]=freqz(num,den,0:0.01:pi);
        Test.Female_FL.Hb=filter(num,den,aa1{pw},[],1);   %對一個時間40個人
        
        
        figure(2),subplot(211),plot(Test.Female_FL.Hb);
        
        CF2=0.02; % 截止頻率 0.02 Hz
        Wn2=CF2/Fn;
        [num1,den1]=butter(4,Wn2,'high'); % Highpass filter
        
        
        Test.Female_FLH.Hb=filter(num1,den1,Test.Female_FL.Hb,[],1);
        
        total2=total2+1;
        temp2=zeros(timeCh,52);
        %temp2{pw}=Test.Female_FLH.Hb;
        temp2(:,:)=temp2(:,:)+Test.Female_FLH.Hb(1:timeCh,:);
        
        result4{total2}=temp2;
        
        figure(2),subplot(212),plot(Test.Female_FLH.Hb)
    end
    
%% 堆疊矩陣2:歸一化
%baresult=reshape(result2,703,2) %兩行兩列重新排列要依據取進來的資料量
% total3=0To RESHAPE the number of elements must not change.
% for pw=1:38
% 
%     S=sum(result2{1,pw})
% %     [min_a,index]=min(result2{1,pw}); %找出每行最小值
% %     [max_a,index]=max(result2{1,pw}); %找出每行最大值
% %     minA=repmat(min_a,3000,1);%複製列成1800*52
% %     maxB=repmat(max_a,3000,1);%複製列成1800*52
% %     CUTA=result2{1,pw}-minA %X-MinValue
% %     CUTB=maxB-minA %(MaxValue-MinValue)
% %     normalize=CUTA./CUTB;  %歸一化y=(x-MinValue)/(MaxValue-MinValue)
%     total3=total3+1;
%     temp3{pw}=S;
%     result3{total3}=temp3{pw};
% end
% 
total4=0
for pw=1:2310
   % SA=result3{1,pw}/6000
    %repmean=repmat(SA,6000,1) %sumX去平均幾個6000個點得MeanX
    [min_a,index]=min(result4{1,pw}); %找出每行最小值
    [max_a,index]=max(result4{1,pw}); %找出每行最大值
    minA=repmat(min_a,6000,1);%複製列成6000*52 1指的是列
    maxB=repmat(max_a,6000,1);%複製列成6000*52 1指的是列
    k=ones(6000,52)
    CUTA=result4{1,pw}-minA % 歸一化 [-1~1] Y=[2(x-x_min)/(x_max-x_min)]-1
    CUTB=2.*CUTA
    CUTC=maxB-minA
    YN=CUTB./CUTC
    re=YN-k
    total4=total4+1;
    temp4{pw}=re;
    result5{total4}=temp4{pw};
    
    figure(3),plot(re)
end
%% gender train 
%pe=
norresult=reshape(aa1,120,2)
A1=norresult(:,1)
A2=norresult(:,2)
% A3=norresult(:,3)
% A4=norresult(:,4)
% A5=norresult(:,5)
% A1=aa1(:,1)
% A2=aa1(:,2)
% A3=aa1(:,3)
% A4=aa1(:,4)
% A5=aa1(:,5)
% A6=norresult(:,6)
% A7=norresult(:,7)
% A8=norresult(:,8)
% A9=norresult(:,9)
for i=1:120
    for j=1:1
        sumAAA{i,j}=(A1{i,j}+A2{i,j})./2; %+A3{i,j}+A4{i,j}+A5{i,j}+A6{i,j}+A7{i,j}+A8{i,j}+A9{i,j}相加獲得擴增的7770位的data+A3{i,j}+A4{i,j}+A5{i,j}
    end 
end 
%first取3050
% A=cell2mat(sumAA1);
% B=A./3 %真正獲得擴增的7770位的data
% % C=mat2cell(B,600*ones(1,3050),52*ones(1,1));
% result66={}
% sumAA=sumAA'
% zz=zeros(6000-timeCh,52)
% total66=0
% for i=1:220
%     total66=total66+1
%     temp66=zeros(6000,52)
%     ccc=cat(1,sumAA{1,i},zz)
%     temp77(:,:,1)=temp66(:,:,1)+ccc(1:6000,:);
%     result66{total66}=temp77;
% end
    %% Pearson's correlations 
R=zeros(52,52);
P=zeros(52,52);

% 計算 Pearson's correlations
%total9=0
%for cc=1:37
    for m=1:52
        for n=1:52
            [r p]=corrcoef(result3{1,6}(:,m),result3{1,6}(:,n));
            R(m,n)=r(2,1);
            P(m,n)=p(2,1);
%             total9=total9+1;
%             temp9{cc}=R;
%             result9{total9}=temp9{cc}
        end
    end
%end

figure(5)
grid on
grid minor
%imagesc(R);colormap(gray);colorbar;axis image; % 全部 CH 兩兩相關係數
 imagesc(R,[-0.2 1]);colormap(jet);colorbar;axis image;set(gca,'fontsize',16,'fontweight','bold');%colormap(gray);
%title('ΔHbO2','fontsize',22,'fontweight','bold')
%title('ΔHb','fontsize',22,'fontweight','bold')
%title('ΔtHb','fontsize',22,'fontweight','bold')
% TT=R-diag(diag(R))
% finalR=reshape(R,2704,1)
%% gender test
norresult=reshape(result4,20,3)
A1=norresult(:,1)
A2=norresult(:,2)
A3=norresult(:,3)
% A4=norresult(:,4)
% A5=norresult(:,5)
% A6=norresult(:,6)
% A7=norresult(:,7)
for i=1:20
    for j=1:1
        sumAA{i,j}=(A1{i,j}+A2{i,j}+A3{i,j})./3; %相加獲得擴增的7770位的data+A3{i,j}+A4{i,j}+A5{i,j}
    end 
end 

result66={}
zz=zeros(6000-timeCh,52)
total66=0
for i=1:37
    total66=total66+1
    temp66=zeros(6000,52)
    ccc=cat(1,result2{1,i},zz)
    temp77(:,:,1)=temp66(:,:,1)+ccc(1:6000,:);
    result66{total66}=temp77;
end
%%
% %--------------------------------------%
% %              fft _train
% %--------------------------------------%

figure(6)
total4=0;
total9=0
total12=0
L=6000; % Length of signal 10min
Fs=10; % 取樣頻率 Fs = 10 Hz
T=1/Fs


for pw=1:16
% t = (1800:2999)*T;
NFFT = 2^nextpow2(L);
f = Fs/2*linspace(0,1,NFFT/2+1);
Y = fft(result4{pw},NFFT)/L;
YY=2*abs(Y(1:NFFT/2+1,1:52));
tY=YY(1:128,1:52);
total9=total9+1;
temp6=zeros(128,52); %要先跑看Y多少資料點
temp6(:,:)=temp6(:,:)+tY;
result9{total9}=temp6;

f1=f(:,1:128);
f1=f1';

% % tY=YY(1:82);
% % xindex=f1<0.1;
% total10=total10+1;
% temp8=zeros(4097,52); %要先跑看YY跑多少資料點
% temp8(:,:)=temp8(:,:)+YY;
% result10{total10}=temp8; %輸出ok 2049*52
% %tY=result10{1,1:3}(1:512)
% tY=YY(1:512,1:52);


% hold on

% xindex=f1<0.1;
% newf1=f1(xindex);
% newtY=tY(xindex);
 
% total4=total4+1;
% temp6=zeros(512,52); %要先跑看他多少資料點
% temp6(:,:)=temp6(:,:)+tY;
% result3{total4}=temp6;

%plot(f,2*abs(Y(1:NFFT/2+1))) 
%plot(f1,result9{1,9}) 
plot(f1,temp6) 
%[a.b]=findpeaks(tY);
%hold on
%plot(x(b),a,'ro')
%fftsult=reshape(result3,703,2) %兩行兩列重新排列要跑完才能用
title('frequency domain')
xlabel('Frequency (Hz)')
ylabel('signal amplitude') 
%plot(f1(tY==max(tY)),max(tY),'bo')
%text(f1(tY==max(tY)),max(tY),['(',num2str(f1(tY==max(tY))),',',num2str(max(tY)),')'])
end
%%
result9=result9'
total12=0
for rc=1:462
DATA=reshape(result9{rc,1},6656,1)
total12=total12+1
temp10{rc}=DATA
result10{total12}=temp10{rc}
end
result10=result10'
rcnn=cell2mat(result10)
%%
b=zeros(1,1)
train_data=cat(1,b,rcnn)
final=cat(1,train_data,rcnn)
csvwrite('x_test2min.csv',final); %寫入.csv檔
%fftsult=reshape(result9,1332,1)
%%
final=cat(1,C10_5_femalefirst,C10_4_female,C10_3_female,C10_2_female,C10_1_female,C10_5_male,C10_4_male,C10_3_male,C10_2_male,C10_1_male)
final=cat(1,C10_1traindata_m,C11_2_femaletest,C11_3_femaletest,C11_4_femaletest,C11_5_femaletest,C6_1_maletest,C6_2_maletest,C6_3_maletest,C6_4_maletest,C6_5_maletest)
final=cat(1,b,female10,female45,female120,female210,male10,male45,male120,male210)
final=cat(1,b,female11_2test,male6_2test,male6_3test)
%%
%%
% %--------------------------------------%
% %              fft _test 
% %--------------------------------------%

figure(6)
total4=0;
total9=0
total12=0
L=6000; % Length of signal 10min
Fs=10; % 取樣頻率 Fs = 10 Hz
T=1/Fs

for pw=1:10
% t = (1800:2999)*T;
NFFT = 2^nextpow2(L);
f = Fs/2*linspace(0,1,NFFT/2+1);
Y = fft(result4{pw},NFFT)/L;
YY=2*abs(Y(1:NFFT/2+1,1:52));
tY=YY(1:128,1:52);
total9=total9+1;
temp6=zeros(128,52); %要先跑看Y多少資料點
temp6(:,:)=temp6(:,:)+tY;
result9{total9}=temp6;

f1=f(:,1:128);
f1=f1';

% % tY=YY(1:82);
% % xindex=f1<0.1;
% total10=total10+1;
% temp8=zeros(4097,52); %要先跑看YY跑多少資料點
% temp8(:,:)=temp8(:,:)+YY;
% result10{total10}=temp8; %輸出ok 2049*52
% %tY=result10{1,1:3}(1:512)
% tY=YY(1:512,1:52);


% hold on

% xindex=f1<0.1;
% newf1=f1(xindex);
% newtY=tY(xindex);
 
% total4=total4+1;
% temp6=zeros(512,52); %要先跑看他多少資料點
% temp6(:,:)=temp6(:,:)+tY;
% result3{total4}=temp6;

%plot(f,2*abs(Y(1:NFFT/2+1))) 
plot(f1,tY) 
%[a.b]=findpeaks(tY);
%hold on
%plot(x(b),a,'ro')
%fftsult=reshape(result3,703,2) %兩行兩列重新排列要跑完才能用
title('frequency domain')
xlabel('Frequency (Hz)')
ylabel('signal amplitude') 
%plot(f1(tY==max(tY)),max(tY),'bo')
%text(f1(tY==max(tY)),max(tY),['(',num2str(f1(tY==max(tY))),',',num2str(max(tY)),')'])
end
%%
%取出PFC特定顯著差異之34個通道
x=[3 4 5 6 7 8 13 14 15 16 17 18 19 23 24 25 26 27 28 29 30 34 35 36 37 38 39 40 45 46 47 48 49 50]
total5=0
total12=0
for pw=1:16 %幾個人+
     total5=total5+1
    for j=x
        dominantfre=result9{1,pw}(:,x)
       temp5=zeros(128,34);
        temp5(:,:)=temp5(:,:)+dominantfre;
    end
    result5{total5}=temp5;
end
result5=result5'
for rc=1:16
DATA=reshape(result5{rc,1},4352,1)
total12=total12+1
temp10{rc}=DATA
result10{total12}=temp10{rc}
end
result10=result10'
rcnn=cell2mat(result10)
%%
%取出PFC特定顯著差異11個通道
x=[7 8 13 19 23 29 30 34 38 39 45]
total5=0
total12=0
people=55
for pw=1:people %幾個人+
     total5=total5+1
    for j=x
        dominantfre=c11_2_result9{1,pw}(:,x)
       temp5=zeros(128,11);
        temp5(:,:)=temp5(:,:)+dominantfre;
    end
    result5{total5}=temp5;
end
result5=result5'
for rc=1:people
DATA=reshape(result5{rc,1},1408,1)
total12=total12+1
temp10{rc}=DATA
result10{total12}=temp10{rc}
end
result10=result10'
rcnn=cell2mat(result10)
%%
b=zeros(1,1)
train_data=cat(1,b,rcnn)
final_fe_ma=cat(1,b,rcnnfemale,rcnnmaleraw6P)
final=cat(1,b,rcnn1fe,rcnn2fe,rcnn1,rcnn2,rcnn3,rcnn4,rcnn5)
csvwrite('xraw_train 34 channel.csv',train_data); %寫入.csv檔
%fftsult=reshape(result9,1332,1)
final=cat(1,b,C10_1traindata,C10_2traindata,C10_3traindata,C10_4traindata,C10_5traindata,C10_6traindata,C10_7traindata,C10_8traindata,C10_9traindata)
final=cat(1,C10_1traindata_m,C10_2traindata_m,C10_3traindata_m,C10_4traindata_m,C10_5traindata_m,C10_6traindata_m,C10_7traindata_m,C10_8traindata_m,C10_9traindata_m)
final_fe_ma=cat(1,b,finalc11_1_2,finalc615)
final=cat(1,b,female_C10_2_34ch,male_C6_2_34ch,male_C6_3_34ch)
%% 取出seed-based 和特定channel 38,11,21
%csvwrite('CNN_traindata.csv', train_data);
clc;clear
load("filhbo2")
initial=0
k=0
temp=zeros(6000,52)
currTime=6000
finaltrain=zeros(6000,3050)
for i=1:3050
    initial=initial+1
    k=k+1
    result=C
%     tHb=Whole2(i).tHb  %取出rawdata
%     temp2(:,:,1)=temp(:,:,1)+tHb(1:currTime,:);233
%     result{initial}=temp2
    ds1=result{1,i}(:,11) %取出特定channel
    ds2=result{1,i}(:,21)
    ds3=result{1,i}(:,38)
    finaltrain2=finaltrain(6000,k)+ds1
    finaltrain3=finaltrain(6000,k)+ds2
    finaltrain4=finaltrain(6000,k)+ds3
    result2{k}=finaltrain2 %存成個別cell矩陣
    result3{k}=finaltrain3
    result4{k}=finaltrain4
end
IDCNN_b1=cell2mat(result2) %將cell矩陣轉為一般
IDCNN_b2=cell2mat(result3)
IDCNN_s1=cell2mat(result4)
N1DCNN_b1=reshape(IDCNN_b1,18300000,1) %重新排列矩陣成要的資料數列
N1DCNN_b2=reshape(IDCNN_b2,18300000,1)
N1DCNN_s1=reshape(IDCNN_s1,18300000,1)
b=zeros(1,1)
final_b1=cat(1,b,N1DCNN_b1) %因python起始是從0，故新增一個數在最前面
train_data=cat(1,final_b1,N1DCNN_b2,N1DCNN_s1)
csvwrite('CNN_traindata.csv', train_data); %寫入.csv檔
