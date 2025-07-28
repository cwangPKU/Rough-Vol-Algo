% MAIN script for discretization scheme implementation--Lifted model
% Markovian approximation for rough Heston model
clear; clc;
rng(42);
addpath("helpers/");

function calcData(H0, rho0)

    %  Rough Heston model:  (8.1.3)-(8.1.4) in (Abi Jaber, 2018)
    H = H0;
    theta = 0.05;
    rho = rho0;
    lambda = 0.1;
    nu = 0.3;
    r = 0; d = 0;
    V0 = 0.05;
    eps = 0.0001;
    
    %  Lifted Heston model: (8.2.1)-(8.2.3) in (Abi Jaber, 2018)
    nf = 400; % number of factors 
    alpha = H + 0.5;
    rn = 1 + 10 * nf^(-0.9);
    cs = ones(1,nf) .* ((rn^(1-alpha) - 1)*rn^((alpha-1)*(1+nf/2))) ./ (gamma(alpha) * gamma(2-alpha));
    xs = ones(1,nf) .* ((1-alpha)/(2-alpha)*(rn^(2-alpha)-1)/(rn^(1-alpha)-1));
    for i = 1:nf
        cs(i) = cs(i) * rn^((1-alpha)*i);
        xs(i) = xs(i) * rn^(i-1-nf/2);
    end
    
    % time grid definition
    dt_maturity = 1/252;  % gap between maturities in IV surface
    %num_maturity = 20;    % number of maturities for IV surface
    num_maturity = 10;
    num_moneyness = 200;
    step_maturity = 50;  % steps in-fill (100 for reasonable results for 1/10 day to maturity)
    num_steps_mc = num_maturity * step_maturity; % number of steps in total
    num_path_mc = 3e5;    % Monte Carlo paths
    dt = dt_maturity / step_maturity; 
    J = 2; % hybrid scheme #(Gaussian noise)
    
    filename = sprintf('output/result_H_%.2f_rho_%.2f.mat', H, rho);
    if isfile(filename)
        disp(filename);
        return;
    end
    %% Kernel coefficients and random matrices
    % reference: Sec~3.1, Bennedsen et al. (2017)
    noise_mat_3d_mc = get_noise_mat(J, H, num_path_mc, num_steps_mc);
    
    %% Implicit-explicit scheme lifted Heston (Markov Approximation)
    u_current = zeros(1,nf);
    num_steps_mc = num_maturity * step_maturity;
    forward_curve = lambda * theta * sum(cs./xs .* (1-exp(-(1:num_steps_mc)'*dt*xs)),2)'; % func g_0(t)
    results_1 = markov_approx(V0, num_maturity, dt_maturity,...
        num_moneyness, num_path_mc, step_maturity, u_current, forward_curve, cs, xs, ...,
        theta, rho, lambda, nu, H, noise_mat_3d_mc, r, eps, 'iv');
    t1 = results_1.t;
    moneyness_mat_1 = results_1.moneyness_mat;
    iv_mat_1 = results_1.iv_mat;
    
    %% Hybrid scheme rough Heston
    forward_curve_hyb = lambda * theta * ((1:num_steps_mc)*dt).^(H+1/2) / (H+1/2) / gamma(H+1/2);
    results_2 = standard_hybrid(V0, num_maturity, dt_maturity,...
        num_moneyness, num_path_mc, step_maturity, forward_curve_hyb,...
        theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps, 'iv');
    t2 = results_2.t;
    moneyness_mat_2 = results_2.moneyness_mat;
    iv_mat_2 = results_2.iv_mat;
    
    
    %% Inverse transform 
    addpath('Fourier_Transform_Inversion');
    n = 400; N = 20000;
    left = 1e-12; right = 400;
    iv_mat_3 = zeros(num_maturity, num_moneyness);
    % log_strikes = linspace(-2*sqrt(V0*dt_maturity),2*sqrt(V0*dt_maturity),num_moneyness);
    % iv_mat_3(1,:) = IVCurveArray(log_strikes, H, rho, nu, lambda, theta, V0, dt_maturity, r, d, n, left, right, N);
    tic;
    for k = 1:num_maturity
        log_strikes = linspace(-2*sqrt(V0*dt_maturity*k),2*sqrt(V0*dt_maturity*k),num_moneyness);
        iv_mat_3(k,:) = IVCurveArray(log_strikes, H, rho, nu, lambda, theta, V0, dt_maturity*k, r, d, n, left, right, N);
        disp(k);
    end
    t3 = toc;
    
    
    %% Euler scheme
    results_5 = euler_scheme(V0, num_maturity, dt_maturity, num_moneyness, num_path_mc, step_maturity,...
        theta, rho, lambda, nu, H, noise_mat_3d_mc, r, eps, 'iv');
    t5 = results_5.t;
    moneyness_mat_5 = results_5.moneyness_mat;
    iv_mat_5 = results_5.iv_mat;
    
    %% Rough Volatility Expansion
    addpath('RV_Expansion');
    
    results_6 = RV_Expansion(V0, num_maturity, dt_maturity, num_moneyness, theta, rho, lambda, nu, H, r);
    t6 = results_6.t;
    moneyness_mat_6 = results_6.moneyness_mat;
    iv_mat_6 = results_6.iv;
    
    %%
    result.H = H;
    result.theta = theta;
    result.rho = rho;
    result.lambda = lambda;
    result.nu = nu;
    result.r = r;
    result.d = d;
    result.V0 = V0;
    result.eps = eps;
    result.dt_maturity = dt_maturity;
    result.num_maturity = num_maturity;
    result.num_moneyness = num_moneyness;
    
    result.moneyness_mat_markov = log(moneyness_mat_1);
    result.iv_mat_markov = iv_mat_1;
    result.t_markov = t1;
    
    result.moneyness_mat_hybrid = log(moneyness_mat_2);
    result.iv_mat_hybrid = iv_mat_2;
    result.t_hybrid = t2;
    
    result.moneyness_mat_fourier = log(moneyness_mat_2);
    result.iv_mat_fourier = iv_mat_3;
    result.t_fourier = t3;
    
    result.moneyness_mat_euler = log(moneyness_mat_5);
    result.iv_mat_euler = iv_mat_5;
    result.t_euler = t5;
    
    result.moneyness_mat_expansion = moneyness_mat_6;
    result.iv_mat_expansion = iv_mat_6;
    result.t_expansion = t6;
    
    %filename = sprintf('output/result_H_%.2f_rho_%.2f.mat', result.H, result.rho);
    save(filename, 'result');
end

for H0 = [0.05, 0.15, 0.25, 0.35, 0.45]
    for rho0 = [-0.1, 0, 0.1]
        calcData(H0, rho0);
    end
end

%% Plot IV Curve

function drawIVcurve(H, rho)
    filename = sprintf('output/result_H_%.2f_rho_%.2f.mat', H, rho);
    data = load(filename);
    result = data.result;
    

    alw = 0.75;    % AxesLineWidth
    fsz = 11;      % Fontsize
    lw = 1.5;      % LineWidth
    msz = 8;       % MarkerSize
    if ismac
        width = 3;     % Width in inches
        height = 2.25;    % Height in inches
    elseif ispc
        width = 4;     
        height = 3;   
    end
    
    indices = [1,2,3,4,5,6,7,8,9,10];
    for i = 1 : 10
        figure;
        pos = get(gcf, 'Position');
        set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
        set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
        hold on;
        plot(result.moneyness_mat_markov(indices(i),:), result.iv_mat_markov(indices(i),:), "DisplayName", "Markov Approximation");
        plot(result.moneyness_mat_hybrid(indices(i),:), result.iv_mat_hybrid(indices(i),:), "DisplayName", "Classical Hybrid");
        plot(result.moneyness_mat_fourier(indices(i),:), result.iv_mat_fourier(indices(i),:), "DisplayName", "Transform Inversion");
        plot(result.moneyness_mat_euler(indices(i),:), result.iv_mat_euler(indices(i),:), "DisplayName", "Euler Scheme");
        plot(result.moneyness_mat_expansion(indices(i),:), result.iv_mat_expansion(indices(i),:), "DisplayName", "Rough Volatility Expansion");
        grid off;
        box on;
        hold off;
    
        xlabel('$\log(K/S)$', 'Interpreter','latex');
        ylabel('Implied Volatility', 'Interpreter','latex');
        legend('Location','best','Interpreter','latex','FontSize',8);
        title(sprintf('IV: $T = %d/252$, $H = %.2f$, $rho = %.2f$', indices(i), H, rho), 'Interpreter','latex');
        print(gcf, '-dpng', '-r300', sprintf('IV_curve/iv_%d_Day_%.2f_H_%.2f_rho_point15H_400nf_50n_neg7rho.png', indices(i), H, rho));
    end
end

for H = [0.05, 0.15, 0.25, 0.35, 0.45]
    for rho = [-0.1, 0, 0.1]
        drawIVcurve(H, rho);
    end
end

%% plot implied volatility surface
function drawIVsurface(H, rho)

    alw = 0.75;    % AxesLineWidth
    fsz = 11;      % Fontsize
    lw = 1.5;      % LineWidth
    msz = 8;       % MarkerSize
    if ismac
        width = 3;     % Width in inches
        height = 2.25;    % Height in inches
    elseif ispc
        width = 4;     
        height = 3;   
    end

    filename = sprintf('output/result_H_%.2f_rho_%.2f.mat', H, rho);
    data = load(filename);
    result = data.result;
    
    tau_vec = (1:result.num_maturity)' * result.dt_maturity;

    figure;
    pos = get(gcf, 'Position');
    set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
    set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
    
    tc = colormapToTrueColor(winter(), result.iv_mat_markov);
    surf(result.moneyness_mat_markov, tau_vec, result.iv_mat_markov, tc, 'FaceColor','interp', ...
        'EdgeAlpha',0.3,'DisplayName','Markov Approximation');
    
    hold on;
    tc = colormapToTrueColor(autumn(), result.iv_mat_fourier);
    surf(result.moneyness_mat_fourier, tau_vec, result.iv_mat_fourier, tc, 'FaceColor','interp', ...
        'EdgeAlpha',0.3,'DisplayName','Transform Inversion');

    hold on;
    tc = colormapToTrueColor(spring(), result.iv_mat_hybrid);
    surf(result.moneyness_mat_hybrid, tau_vec, result.iv_mat_hybrid, tc, 'FaceColor','interp', ...
        'EdgeAlpha',0.3,'DisplayName','Hybrid Scheme');
    
    hold on;
    tc = colormapToTrueColor(summer(), result.iv_mat_euler);
    surf(result.moneyness_mat_euler, tau_vec, result.iv_mat_euler, tc, 'FaceColor','interp', ...
        'EdgeAlpha',0.3,'DisplayName','Euler Scheme');

    hold on;
    tc = colormapToTrueColor(gray(), result.iv_mat_expansion);
    surf(result.moneyness_mat_expansion, tau_vec, result.iv_mat_expansion, tc, 'FaceColor','interp', ...
        'EdgeAlpha',0.3,'DisplayName','Rough Vol Expansion');

    xlabel("$K$", "Interpreter", "latex");
    ylabel("$T$", "Interpreter", "latex");
    zlabel("Implied Volatility", "Interpreter", "latex");
    title(sprintf("$H=%.2f, rho=%.2f$", H, rho), 'Interpreter','latex');
    
    % colorbar;
    % view(45,30);
    legend('Location','northeast','Interpreter','latex');
    grid on;
    box off;
    hold off;
    print(gcf, '-dpng', '-r300', sprintf('IV_surface/iv_surface_%.2f_H_%.2f_rho.png', H, rho));
end

for H = [0.05, 0.15, 0.25, 0.35, 0.45]
    for rho = [-0.1, 0, 0.1]
        drawIVsurface(H, rho);
    end
end

%% Test rho shape
rho_vec = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5];

for H = [0.15, 0.45]
    for rho = rho_vec
        calcData(H, rho);
    end
end

%%

function drawIVrhoSurface(H, num_t)

    alw = 0.75;    % AxesLineWidth
    fsz = 11;      % Fontsize
    lw = 1.5;      % LineWidth
    msz = 8;       % MarkerSize
    if ismac
        width = 3;     % Width in inches
        height = 2.25;    % Height in inches
    elseif ispc
        width = 4;     
        height = 3;   
    end

    rho_vec = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
    iv_mat_markov = zeros(size(rho_vec, 2), 200);
    iv_mat_hybrid = zeros(size(rho_vec, 2), 200);
    iv_mat_fourier = zeros(size(rho_vec, 2), 200);
    iv_mat_euler = zeros(size(rho_vec, 2), 200);
    iv_mat_expansion = zeros(size(rho_vec, 2), 200);
    
    moneyness_mat = zeros(size(rho_vec, 2), 200);
    
    for i = 1:size(rho_vec, 2)
        
        rho = rho_vec(i);
    
        filename = sprintf('output/result_H_%.2f_rho_%.2f.mat', H, rho);
        data = load(filename);
        result = data.result;
    
        iv_mat_markov(i, :) = result.iv_mat_markov(num_t, :); 
        iv_mat_hybrid(i, :) = result.iv_mat_hybrid(num_t, :); 
        iv_mat_fourier(i, :) = result.iv_mat_fourier(num_t, :); 
        iv_mat_euler(i, :) = result.iv_mat_euler(num_t, :);
        iv_mat_expansion(i, :) = result.iv_mat_expansion(num_t, :); 
        
        moneyness_mat(i, :) = result.moneyness_mat_markov(num_t, :); 
    end
    
    figure;
    pos = get(gcf, 'Position');
    set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
    set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
    
    tc = colormapToTrueColor(winter(), iv_mat_markov);
    surf(moneyness_mat, rho_vec, iv_mat_markov, tc, 'FaceColor','interp', ...
        'EdgeAlpha',0.3,'DisplayName','Markov Approximation');
    
    hold on;
    tc = colormapToTrueColor(autumn(), iv_mat_hybrid);
    surf(moneyness_mat, rho_vec, iv_mat_hybrid, tc, 'FaceColor','interp', ...
        'EdgeAlpha',0.3,'DisplayName','Hybrid Scheme');
    
    hold on;
    tc = colormapToTrueColor(spring(), iv_mat_fourier);
    surf(moneyness_mat, rho_vec, iv_mat_fourier, tc, 'FaceColor','interp', ...
        'EdgeAlpha',0.3,'DisplayName','Transform Inversion');
    
    hold on;
    tc = colormapToTrueColor(summer(), iv_mat_euler);
    surf(moneyness_mat, rho_vec, iv_mat_euler, tc, 'FaceColor','interp', ...
        'EdgeAlpha',0.3,'DisplayName','Euler Scheme');
    
    hold on;
    tc = colormapToTrueColor(gray(), iv_mat_expansion);
    surf(moneyness_mat, rho_vec, iv_mat_markov, tc, 'FaceColor','interp', ...
        'EdgeAlpha',0.3,'DisplayName','Rough Vol Expansion');
    
    xlabel("$K$", "Interpreter", "latex");
    ylabel("$\rho$", "Interpreter", "latex");
    zlabel("Implied Volatility", "Interpreter", "latex");
    title(sprintf("$H=%.2f, T=%d$ days", H, num_t), 'Interpreter','latex');
    
    % colorbar;
    % view(45,30);
    legend('Location','northeast','Interpreter','latex');
    grid on;
    box off;
    hold off;
    print(gcf, '-dpng', '-r300', sprintf('IV_rhoSurface/iv_rhoSurface_%.2f_H_%d_days.png', H, num_t));
end

for H = [0.15, 0.45]
    for num_t = 1:10
        drawIVrhoSurface(H, num_t);
    end
end