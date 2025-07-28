% MAIN script for discretization scheme implementation--Lifted model
% Markovian approximation for rough Heston model
clear; clc;
rng(1);
addpath('./Rough_Vol_FPT/');
addpath('./helpers/')
%% Parameters
%  Rough Heston model:  (8.1.3)-(8.1.4) in (Jaber, 2018)
H = 0.15;
theta = 0.05;
rho = -0.1;
lambda = 0.1;
nu = 0.3;
r = 0; d = 0;
V0 = 0.05;
eps = 0.0001;

%  Lifted Heston model: (8.2.1)-(8.2.3) in (Jaber, 2018)
nf = 100; % number of factors (50 would yield reasonable results for 1 day to maturity)
alpha = H + 0.5;
rn = 1 + 10 * nf^(-0.9);
cs = ones(1,nf) .* ((rn^(1-alpha) - 1)*rn^((alpha-1)*(1+nf/2))) ./ (gamma(alpha) * gamma(2-alpha));
xs = ones(1,nf) .* ((1-alpha)/(2-alpha)*(rn^(2-alpha)-1)/(rn^(1-alpha)-1));
for i = 1:nf
    cs(i) = cs(i) * rn^((1-alpha)*i);
    xs(i) = xs(i) * rn^(i-1-nf/2);
end

% time grid definition
t = 1;
num_steps_mc = 200; % number of steps in total
num_path_mc = 4e5;    % Monte Carlo paths
dt = t / num_path_mc; 
J = 2; % hybrid scheme #(Gaussian noise)

% Kernel coefficients and random matrices
noise_mat_3d_mc = get_noise_mat(J, H, num_path_mc, num_steps_mc);

%% Implicit-explicit scheme lifted Heston (Markov Approximation)
u_current = zeros(1,nf);
forward_curve = lambda * theta * sum(cs./xs .* (1-exp(-(1:num_steps_mc)'*dt*xs)),2)'; % func g_0(t)
tic;
logS_markov_approx = logS_simu_markov_approx(V0, num_steps_mc, t,...
    num_path_mc, u_current, forward_curve, cs, xs, ...,
    theta, rho, lambda, nu, H, noise_mat_3d_mc, r, eps);
t1 = toc;

%% Hybrid scheme rough Heston
forward_curve_hyb = lambda * theta * ((1:num_steps_mc)*dt).^(H+1/2) / (H+1/2) / gamma(H+1/2);
tic;
logS_hybrid_scheme = logS_simu_hybrid_scheme2(V0, num_steps_mc, t,...
    num_path_mc, forward_curve_hyb, theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps);
t2=toc;

%% Euler scheme rough Heston
tic;
logS_euler_scheme = logS_simu_euler_scheme(V0, num_steps_mc, t, num_path_mc, ...,
    theta, rho, lambda, nu, H, noise_mat_3d_mc, r, eps);
t3 = toc;

%% Accelerated Hybrid scheme
forward_curve_hyb = lambda * theta * ((1:num_steps_mc)*dt).^(H+1/2) / (H+1/2) / gamma(H+1/2);
tic;
logS_accelerated_hybrid = logS_simu_accelerated_hybrid(V0, num_steps_mc, t, num_path_mc, u_current, forward_curve, cs, xs, ...,
    theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps);
t4 = toc;
%%
a = -0.1;
b = 0.1;
[prob_a_markov_approx, prob_b_markov_approx, prob_in_markov_approx, logS_final_markov_approx] = path_prob(a, b, logS_markov_approx);
[prob_a_hybrid_scheme, prob_b_hybrid_scheme, prob_in_hybrid_scheme, logS_final_hybrid_scheme] = path_prob(a, b, logS_hybrid_scheme);
[prob_a_euler_scheme, prob_b_euler_scheme, prob_in_euler_scheme, logS_final_euler_scheme] = path_prob(a, b, logS_euler_scheme);
[prob_a_accelerated_hybrid, prob_b_accelerated_hybrid, prob_in_accelerated_hybrid, logS_final_accelerated_hybrid] = path_prob(a, b, logS_accelerated_hybrid);

prob_mat = [prob_a_markov_approx, prob_b_markov_approx, prob_in_markov_approx; prob_a_hybrid_scheme, prob_b_hybrid_scheme, prob_in_hybrid_scheme; prob_a_euler_scheme, prob_b_euler_scheme, prob_in_euler_scheme; prob_a_accelerated_hybrid, prob_b_accelerated_hybrid, prob_in_accelerated_hybrid];
disp(prob_mat);

filename = sprintf("FPT/prob_mat_H_%.2f_t_%.2f.csv", H, t);
writematrix(prob_mat, filename);

%%
W_T = logS_final_markov_approx;
prob_in = prob_in_markov_approx;
[counts, edges] = histcounts(W_T);
binWidth = edges(2) - edges(1); % 分箱宽度（假设等宽）
originalArea = sum(counts) * binWidth;
scalingFactor = prob_in / originalArea;
scaledCounts = counts * scalingFactor;

% 绘制直方图
figure;
h = bar(edges(1:end-1), scaledCounts, 1, 'EdgeColor', 'none', 'DisplayName', 'Histogram');

% 获取直方图数据
x = h.XData;                   % 柱子中心坐标 (1×N)
y = h.YData;                   % 柱子高度 (1×N)
w = diff(edges)/2;             % 柱子半宽 (1×N)

% 生成顶部边界路径（仅含柱子的左右上顶点）
boundary_x = [x - w; x + w];   % 左/右边缘 x 坐标 (2×N)
boundary_y = [y; y];           % 对应高度 (2×N)

% 将路径展开为连续点序列
boundary_x = boundary_x(:);    % 列向量: [左1,右1,左2,右2,...]
boundary_y = boundary_y(:);    % 列向量: [y1,y1,y2,y2,...]

% 绘制红色虚线（仅顶部）
hold on;
plot(boundary_x, boundary_y, '--r',...
    'LineWidth', 1.8,...
    'DisplayName', 'Conditional Probability Density');

% 图形装饰
xlabel('Position (x)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Conditional Probability Density', 'FontSize', 12, 'FontWeight', 'bold');
title('Conditional Probability Density for Markov Approximation', 'FontSize', 14);
grid on;
box on;
legend('show');
filename = sprintf("FPT/Markov_H_%.2f_t_%.2f.png", H, t);
saveas(gcf, filename);

%%
W_T = logS_final_hybrid_scheme;
prob_in = prob_in_hybrid_scheme;
[counts, edges] = histcounts(W_T);
binWidth = edges(2) - edges(1); % 分箱宽度（假设等宽）
originalArea = sum(counts) * binWidth;
scalingFactor = prob_in / originalArea;
scaledCounts = counts * scalingFactor;

% 绘制直方图
figure;
h = bar(edges(1:end-1), scaledCounts, 1, 'EdgeColor', 'none', 'DisplayName', 'Histogram');

% 获取直方图数据
x = h.XData;                   % 柱子中心坐标 (1×N)
y = h.YData;                   % 柱子高度 (1×N)
w = diff(edges)/2;             % 柱子半宽 (1×N)

% 生成顶部边界路径（仅含柱子的左右上顶点）
boundary_x = [x - w; x + w];   % 左/右边缘 x 坐标 (2×N)
boundary_y = [y; y];           % 对应高度 (2×N)

% 将路径展开为连续点序列
boundary_x = boundary_x(:);    % 列向量: [左1,右1,左2,右2,...]
boundary_y = boundary_y(:);    % 列向量: [y1,y1,y2,y2,...]

% 绘制红色虚线（仅顶部）
hold on;
plot(boundary_x, boundary_y, '--r',...
    'LineWidth', 1.8,...
    'DisplayName', 'Conditional Probability Density');

% 图形装饰
xlabel('Position (x)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Conditional Probability Density', 'FontSize', 12, 'FontWeight', 'bold');
title('Conditional Probability Density for Hybrid Scheme', 'FontSize', 14);
grid on;
box on;
legend('show');
filename = sprintf("FPT/Hybrid_H_%.2f_t_%.2f.png", H, t);
saveas(gcf, filename);

%%
W_T = logS_final_euler_scheme;
prob_in = prob_in_euler_scheme;
[counts, edges] = histcounts(W_T);
binWidth = edges(2) - edges(1); % 分箱宽度（假设等宽）
originalArea = sum(counts) * binWidth;
scalingFactor = prob_in / originalArea;
scaledCounts = counts * scalingFactor;

% 绘制直方图
figure;
h = bar(edges(1:end-1), scaledCounts, 1, 'EdgeColor', 'none', 'DisplayName', 'Histogram');

% 获取直方图数据
x = h.XData;                   % 柱子中心坐标 (1×N)
y = h.YData;                   % 柱子高度 (1×N)
w = diff(edges)/2;             % 柱子半宽 (1×N)

% 生成顶部边界路径（仅含柱子的左右上顶点）
boundary_x = [x - w; x + w];   % 左/右边缘 x 坐标 (2×N)
boundary_y = [y; y];           % 对应高度 (2×N)

% 将路径展开为连续点序列
boundary_x = boundary_x(:);    % 列向量: [左1,右1,左2,右2,...]
boundary_y = boundary_y(:);    % 列向量: [y1,y1,y2,y2,...]

% 绘制红色虚线（仅顶部）
hold on;
plot(boundary_x, boundary_y, '--r',...
    'LineWidth', 1.8,...
    'DisplayName', 'Conditional Probability Density');

% 图形装饰
xlabel('Position (x)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Conditional Probability Density', 'FontSize', 12, 'FontWeight', 'bold');
title('Conditional Probability Density for Euler Scheme', 'FontSize', 14);
grid on;
box on;
legend('show');
filename = sprintf("FPT/Euler_H_%.2f_t_%.2f.png", H, t);
saveas(gcf, filename);

%%
W_T = logS_final_accelerated_hybrid;
prob_in = prob_in_accelerated_hybrid;
[counts, edges] = histcounts(W_T);
binWidth = edges(2) - edges(1); % 分箱宽度（假设等宽）
originalArea = sum(counts) * binWidth;
scalingFactor = prob_in / originalArea;
scaledCounts = counts * scalingFactor;

figure;
h = bar(edges(1:end-1), scaledCounts, 1, 'EdgeColor', 'none', 'DisplayName', 'Histogram');

% 获取直方图数据
x = h.XData;                   % 柱子中心坐标 (1×N)
y = h.YData;                   % 柱子高度 (1×N)
w = diff(edges)/2;             % 柱子半宽 (1×N)

% 生成顶部边界路径（仅含柱子的左右上顶点）
boundary_x = [x - w; x + w];   % 左/右边缘 x 坐标 (2×N)
boundary_y = [y; y];           % 对应高度 (2×N)

% 将路径展开为连续点序列
boundary_x = boundary_x(:);    % 列向量: [左1,右1,左2,右2,...]
boundary_y = boundary_y(:);    % 列向量: [y1,y1,y2,y2,...]

% 绘制红色虚线（仅顶部）
hold on;
plot(boundary_x, boundary_y, '--r',...
    'LineWidth', 1.8,...
    'DisplayName', 'Conditional Probability Density');

% 图形装饰
xlabel('Position (x)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Conditional Probability Density', 'FontSize', 12, 'FontWeight', 'bold');
title('Conditional Probability Density for Accelerated Hybrid Scheme', 'FontSize', 14);
grid on;
box on;
legend('show');
filename = sprintf("FPT/Accelerated_H_%.2f_t_%.2f.png", H, t);
saveas(gcf, filename);