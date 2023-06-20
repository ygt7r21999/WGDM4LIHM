% 1.Amp_gt
Int_o_gt = im2double(imread("I0.bmp"));
Amp_o_gt = sqrt(Int_o_gt);

% 2.Pha_gt
Pha_o_gt = im2double(imread("P0.bmp"));
Pha_o_gt = rgb2gray(Pha_o_gt);
% LPha = 0; HPha = pi;
% Pha_o_gt = (HPha-LPha)*...
%            (Pha_o_gt-min(Pha_o_gt(:)))/...
%            (max(Pha_o_gt(:))-min(Pha_o_gt(:)));

objTypeList = ["Amp", "Pha", "Combined"];
objType = objTypeList(2);

Uo_gt = ObjectFieldGenerator(Amp_o_gt,Pha_o_gt,objType);

figure;imagesc(abs(Uo_gt));colormap("gray");axis square;axis off;
title("Amp_o^{gt}");
figure;imagesc(angle(Uo_gt));colormap("gray");axis square;axis off;
title("Pha_o^{gt}");

psize = 1.34e-3;% unit:mm
dist = 1.21;
lamda = 0.534e-3;

% 前向传播
Ui_gt = ASM_Prop(Uo_gt,psize,dist,lamda);

% ------------------------------------------------
Amp_i_gt = abs(Ui_gt); % amplitude mesurement
figure;imagesc(Amp_i_gt);colormap("gray");axis square;axis off;

Max_iter = 100;
%{ 
Wirtinger Gradient Descent
1.alphalist(*):
*=1 → alpha=0.1  for defalut
*=2 → alpha=0.5  for Alternating Projection
*=3 → alpha=0.01 for long iteration
*=4 → alpha=5    for random gradient descent
%}

alphaList = [0.1, 0.5, 0.01, 5];
alpha = alphaList(1);

Uo_pred = zeros(size(Amp_i_gt)).*exp(1j*zeros(size(Amp_i_gt)));
% Uo_pred = randn(size(Amp_i_gt)).*exp(1j*randn(size(Amp_i_gt)));
X_list = linspace(1,Max_iter,Max_iter);
Y_abs_lossList = zeros([1,Max_iter]);
Y_ang_lossList = zeros([1,Max_iter]);
Y_Propabs_lossList = zeros([1,Max_iter]);

for iter = 1:Max_iter
    Uo_pred = Uo_pred - alpha*WirtingerGradient(Uo_pred,Amp_i_gt,"Amp");
    Y_abs_lossList(iter) = norm(abs(Uo_pred)-Amp_o_gt)^2;
    Y_ang_lossList(iter) = norm(angle(Uo_pred)-Pha_o_gt)^2;
    Y_Propabs_lossList(iter) = norm(abs(ASM_Prop(Uo_pred,psize,dist,lamda))-Amp_i_gt)^2;
end

figure;imagesc(abs(Uo_pred));colormap("gray");axis square;axis off;
title("Amp_o^{pred}");
figure;imagesc(angle(Uo_pred));colormap("gray");axis square;axis off;
title("Pha_o^{pred}");

figure;plot(X_list,Y_abs_lossList);
title("{L2-loss}_{gt}^{abs}");
fprintf("the L2 loss of gt-abs is %f\n\n\n", Y_abs_lossList(end));

figure;plot(X_list,Y_ang_lossList);
title("{L2-loss}_{gt}^{pha}");
fprintf("the L2 loss of gt-pha is %f\n\n\n", Y_ang_lossList(end));

figure;plot(X_list,Y_Propabs_lossList);
title("{L2-loss}_{Prop}^{abs}");
fprintf("the L2 loss of prop-abs is %f\n\n\n", Y_Propabs_lossList(end));

%{ 
utilsFunc list:
1.WirtingerGradient

2.ObjectFieldGenerator
%}

function grad_Uin = WirtingerGradient(U_in,ref,Type)
    psize = 1.34e-3;% unit:mm
    dist = 1.21;
    lamda = 0.534e-3;    
    
    if Type=="Amp"
        Ui_fromProp = ASM_Prop(U_in,psize,dist,lamda);
        Ui_AmpC = ref.*exp(1j*angle(Ui_fromProp));
        grad_Uin = (U_in - ASM_Prop(Ui_AmpC,psize,-dist,lamda));
    elseif Type=="Int"
%         grad_Uin = 
    end
end

function U_out = ObjectFieldGenerator(Amp_in,Pha_in,Type)        
    if Type=="Amp"
        Pha_in = zeros(size(Pha_in));
    elseif Type=="Pha"
        Amp_in = ones(size(Amp_in));
    elseif Type=="Combined"
    end
    U_out = Amp_in.*exp(1j*Pha_in);
end

function OptimInit
end
