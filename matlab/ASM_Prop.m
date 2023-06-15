function U_out = ASM_Prop(U_in,psize,dist,lamda)
% coordinate definition
[Y,X] = size(U_in);
Lx = X*psize;
Ly = Y*psize;
[nx,ny] = meshgrid(-X/2+1:X/2, -Y/2+1:Y/2);
fx = nx/Lx; fy = ny/Ly;

k = 2*pi/lamda;
inside_sqrt = 1-(lamda*fx).^2-(lamda*fy).^2;
inside_sqrt(inside_sqrt<0)=0;
Transfer_Matrix = exp(1j*k*dist*sqrt(inside_sqrt));
Transfer_Matrix = fftshift(Transfer_Matrix);

FU_in = fft2(U_in);
FU_out = FU_in.*Transfer_Matrix;
U_out = ifft2(FU_out);
% figure;imagesc(abs(U_out));colormap("gray");axis square;axis off;
end