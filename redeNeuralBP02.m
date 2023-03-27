function [resp] = redeNeuralBP02( )
clear 
clc
% Entrada
x = linspace(-10,10,250);
y = linspace(-10,10,250);

% Calcular função
[X,Y]=meshgrid(x,y);
r= sqrt(X.^2+Y.^2);
z=sin(r)./r;
z(round(length(z)/2),round(length(z)/2))=0;

% Gerar o gráfico
mesh(X,Y,z)
entrada=[x y];
Z=z;

% Mecanismo para fazer o reshape das matrizes
% Guarda linha e coluna de x
[lX,cX]=size(X);
nX=reshape(X,1,lX*cX);
% Guarda linha e coluna de y
[lY,cY]=size(Y);
nY=reshape(Y,1,lY*cY);
% Guarda linha e coluna de z
[lZ,cZ]=size(z);
nZ=reshape(z,1,lZ*cZ);
entrada=[nX;nY];
Z=nZ;

% Criar a rede
%net = newff ( minmax(entrada),[26 1],{'tansig', 'purelin'},'traingd');
%net = newff ( minmax(entrada),[26 1],{'tansig', 'purelin'},'traingdm');
net = newff ( minmax(entrada),[26 1],{'tansig', 'purelin'},'trainlm');

% Definir erro desejado
net.trainParam.goal<(10^-4);

% Treinar a rede
net = train(net,entrada,Z);

%-----------------------------------------------------------
% Simular a rede com menos entradas (41 Pontos)
xx=-10:.5:10;
yy=-10:.5:10;

% Calcular função
[X,Y]=meshgrid(xx,yy);
r= sqrt(X.^2+Y.^2);
z=sin(r)./r;
z(round(length(z)/2),round(length(z)/2))=0;

% Mecanismo para fazer o reshape das matrizes
% Guarda linha e coluna de x
[lX,cX]=size(X);
nX=reshape(X,1,lX*cX);
% Guarda linha e coluna de y
[lY,cY]=size(Y);
nY=reshape(Y,1,lY*cY);
% Guarda linha e coluna de z
[lZ,cZ]=size(z);
nZ=reshape(z,1,lZ*cZ);
entrada2=[nX;nY];
Z=nZ;

% simular a rede com menos entrada
S= sim(net,entrada2);

% Gerar o gráfico com a saída da rede neural
hold on
nS=reshape(S,lZ,cZ);
plot3(X,Y,nS,'ro')
norm(z-nS)
