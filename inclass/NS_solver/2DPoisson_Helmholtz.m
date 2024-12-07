% ---------------------------------------------------------------------- %
% 2D Poisson Equation - Numerical Solution by Gauss-Siedel Method        %
% ---------------------------------------------------------------------- %
%                                 Kiran Bhaganagar
% ---------------------------------------------------------------------- %

% Clear memory and workspaces
clear;
clc;

%------------------%
% Initialization
%------------------%

% Define domain dimensions
x_max = 2 * pi; 
y_max = 4 * pi; 

% Define grid sizes
dx = 0.01;  % Grid size in x-direction
dy = 0.01;  % Grid size in y-direction

rho=0.005; % Conductivity coefficient 

% Convergence Criteria 
err_limit = 1d-5;     % User specified error tolerance limit
err_calc_gauss = 1;  
    
% Populate the x and y grid
x=0:dx:x_max;
y=0:dy:y_max;

% Number of indices in the domain
x_num = numel(x);  
y_num = numel(y);  

% Initializing the matrices required for the calculations
U_init_gauss(y_num,x_num)=zeros;
U_exact(y_num,x_num)=zeros;
f_gauss(y_num,x_num)=zeros;
U_fin_gauss(y_num,x_num)=zeros;
U_residue(y_num,x_num)=zeros;

% Calculating exact solution and source function  
for i=1:y_num
    for j=1:x_num
        U_exact(i,j)= 2 * ( ( 1 + cos( 4 * x[i] ) ) * ( 1 + cos( 10 * y[i] ) ) ) - 2 * sin( 4 * x[i] ) * sin( 10 * y[i] ) % sin(pi * x(j)) * cos(2 * pi * y(i));
        f_gauss(i,j)= (1+(rho * 5 * pi * pi)) * U_exact(i,j);
    end
end

% Applying boundary Conditions
U_init_gauss(1,:) = U_exact(1,:);
U_init_gauss(:,1) = U_exact(:,1);
U_init_gauss(:,x_num) = U_exact(:,x_num);
U_init_gauss(y_num,:) = U_exact(y_num,:);

U_fin_gauss = U_init_gauss;

% Variable used to store number of iterations 
iter_counter_gauss = 0;

%-------------------%
% Calculations
%-------------------%
cc = 1 + ((2 * rho * (dx^2 + dy^2)) / (dx^2 * dy^2));

% Gauss-Siedel Method
while err_calc_gauss > err_limit 
    
    % Display the current iteration number
%     Iteration_Gauss = iter_counter_gauss
    
    % Loop to find the values using Gauss-Siedel iteration method
    for i=2:y_num-1
        for j=2:x_num-1
            
         U_fin_gauss(i,j) = (1/cc) *...
                            (((rho/(dx^2)) * (U_init_gauss(i+1,j) + U_fin_gauss(i-1,j))) + ...
                            ((rho/(dy^2)) * (U_init_gauss(i,j+1) + U_fin_gauss(i,j-1))) + ...
                            (f_gauss(i,j)));
        end
    end
    
    % Find the error matrix as the difference between current itarations
    % value and previous iterations value
    Err_gauss=U_fin_gauss-U_init_gauss;
    
    % Update the initial matrix with the recently calculated values 
    U_init_gauss=U_fin_gauss;

    % Calculate the Root Mean Square of the error matrix obtained 
    err_calc_gauss=sqrt(mean(Err_gauss(:).^2));
       
    % Store the RMS value of the error in an error vector 
    error_gauss(iter_counter_gauss+1) = err_calc_gauss;
    
    % Update the interation count
    iter_counter_gauss=iter_counter_gauss+1;
    err_calc_gauss
end

U_error=U_exact-U_init_gauss;

for i=2:y_num-1
    for j=2:x_num-1
        U_residue(i,j)= U_error(i,j)-(rho*((1/dx^2)*(U_error(i+1,j)-(2*U_error(i,j))+U_error(i-1,j)) + ...
                        (1/dy^2)*(U_error(i,j+1)-(2*U_error(i,j))+U_error(i,j-1))));
    end
end

%--------------------%
% Plotting results
%--------------------%

% Generate Iterations grid for various methods
iter_gauss=1:iter_counter_gauss;

figure(1);
subplot(2,3,1);
    surf(x,y,U_init_gauss);
    axis([0 2*pi 0 4*pi]);
    ch_t1_jac=sprintf('Numerical Solution - Gauss-Siedel Method');
    ch_t2_jac=sprintf('%d iterations to converge',iter_counter_gauss);
    s1_t_jac=char(ch_t1_jac,ch_t2_jac);
    title(s1_t_jac,'fontsize',13);

subplot(2,3,2);
    surf(x,y,U_exact);
    axis([0 2*pi 0 4*pi]);
    title('Exact Solution','fontsize',13);

subplot(2,3,3);
    surf(x,y,f_gauss);
    axis([0 2*pi 0 4*pi]);
    title('Source function','fontsize',13);

subplot(2,3,4);
    surf(x,y,U_residue);
    axis([0 2*pi 0 4*pi]);
    ch_t2=sprintf('Residue mapping for dx=%.2f , dy=%.2f', dx,dy);
    title(ch_t2,'fontsize',13);
    
subplot(2,3,5);
    surf(x,y,U_error);
    axis([0 2*pi 0 4*pi]);
    ch_t3=sprintf('Error mapping for dx=%.2f , dy=%.2f', dx,dy);
    title(ch_t3,'fontsize',13);

for i = 1:5
    subplot(2,3,i);
    shading interp;
    view(2);
    c=colorbar;
    ylabel(c,'Velocity m/sec','fontweight','bold','fontsize',14);
    xlabel('X (0 \rightarrow 2pi)','fontsize',12);
    ylabel('Y (0 \rightarrow 4pi)','fontsize',12);
end

subplot(2,3,6);
    semilogy(iter_gauss,error_gauss);
    grid on;
    grid minor;
    xl2=sprintf('Iterations');
    yl2=sprintf('Error');
    ch_t3=sprintf('Convergence rate');
    title(ch_t3,'fontweight','bold','fontsize',13);
    xlabel(xl2,'fontsize',12);
    ylabel(yl2,'fontsize',12);

% The End %

