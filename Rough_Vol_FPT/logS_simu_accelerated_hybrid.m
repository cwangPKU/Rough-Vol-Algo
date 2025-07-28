function logS = logS_simu_accelerated_hybrid(spotV, num_steps_mc, t, num_path_mc, u_current, forward_curve, cs, xs, ...,
    theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps)
    %ACCELERATED_HYBRID Multi-factor accelerated hybrid SDE simulation
    %   INPUTS:
    %     spotV             – initial variance level (scalar)
    %     num_maturity      – number of maturities under 'iv' mode (integer)
    %     dt_maturity       – maturity time interval under 'iv' mode (scalar)
    %     num_moneyness     – # of strike/moneyness points per maturity for 'iv' mode
    %     num_path_mc       – number of Monte-Carlo paths
    %     steps_per_maturity– time‐steps per maturity
    %     u_current         – current factor state (vector length = # paths)
    %     forward_curve     – forward variance curve (1×num_steps)
    %     cs                – factor loading matrix (# factors)
    %     xs                – multi-factor approximation (# factors)
    %     theta             – rough Heston model parameter (scalar)
    %     rho               – rough Heston model correlation (scalar)
    %     lambda            – rough Heston model mean‐reversion rate
    %     nu                – rough Heston model diffusion term multiplier (scalar)
    %     H                 – Hurst index
    %     noise_mat_3d_mc   – pre-generated noise: [paths × steps × (J+2)]
    %     J                 – number of additional noises for hybrid scheme, see Bennedsen et al. (2017)
    %     r                 – risk-free rate
    %     eps               – small floor for variance
    %     mode              – output mode, one of:
    %                          'std'    : return logS_mc_final + logS_mc_markov + timings
    %                          'iv'     : return implied‐vol matrix + maturities + moneyness
    %                          'full'   : return full V_mat paths
    %
    %   OUTPUT:
    %     RESULTS – struct with these possible fields:
    %       .logS_mc_markov  – Markovian log‐price result
    %       .logS_mc_final   – accelerated hybrid final log‐price
    %       .V_mat_mc        – full variance paths (only if mode='path')
    %       .iv_mat          – implied‐volatility surface (mode='iv')
    %       .tau_vec         – vector of maturities (mode='iv')
    %       .moneyness_mat   – grid of moneyness levels (mode='iv')
    %       .t_markov        – CPU time for predictor step
    %       .t_acchyb        – CPU time for predictor + hybrid steps
    %

    dt = t / num_steps_mc;
    num_factor = size(cs, 2);
    u_current_mc = ones(num_path_mc,1) * u_current;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1) + [0, forward_curve]; % initialize variance paths
    %logS_mc_current = zeros(num_path_mc,1); % current stock price = 1
    logS_mc_markov = zeros(num_path_mc,1);
    logS = zeros(num_path_mc,num_steps_mc+1);
    gamma_val = gamma(H+1/2);

    % resize noise matrix (coupling strategy)
    noise_mat_3d_mc = resize_noise_mat(noise_mat_3d_mc, num_steps_mc);

    % Markovian precomputation

    for i = 1:num_steps_mc
        f_mc_current = max(V_mat_mc(:,i), eps);
        delta_u = -lambda*V_mat_mc(:,i)*dt + nu*sqrt(f_mc_current)*dt^(1/2).*noise_mat_3d_mc(:,i,1);
        % u_current_mc = (u_current_mc + delta_u) ./ (1 + xs*dt);
        u_current_mc = (u_current_mc + delta_u) .* exp(-xs*dt);
        V_mat_mc(:,i+1) = V_mat_mc(:,i+1) + sum(cs .* u_current_mc,2); 
        % test Markovian approximation at the meantime
        logS_mc_markov = logS_mc_markov + (r - f_mc_current/2) * dt + ...
            sqrt(f_mc_current)*dt^(1/2) .* (rho*noise_mat_3d_mc(:,i,1) + sqrt(1-rho^2)*noise_mat_3d_mc(:,i,2));
    end



    kernel_dt = ones(num_path_mc,1) * [0, linspace(1,num_steps_mc,num_steps_mc).^(H+1/2) - ...
        linspace(0,num_steps_mc-1,num_steps_mc).^(H+1/2)] / (H+1/2);
    f_mat_mc = sqrt(max(V_mat_mc, eps));
    noise_path_mc = squeeze(noise_mat_3d_mc(:,1:num_steps_mc,1));
    var_path_diff = nu * f_mat_mc .* [noise_path_mc,zeros(num_path_mc,1)];
    var_path_drift = lambda * dt^(1/2) * V_mat_mc;
    tmp_N = 2^nextpow2(2*num_steps_mc-1);
    % Fast Fourier Transform
    kernel_fft = fft(kernel_dt,tmp_N,2);
    var_fft = fft(var_path_diff - var_path_drift,tmp_N,2);
    res = ifft(kernel_fft.*var_fft,tmp_N,2) * dt^H / gamma_val;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1) + [0, forward_curve] ...
        + res(:,1:num_steps_mc+1); % variance path after FFT
    for i = 1:num_steps_mc
        f_mc_current = sqrt(max(V_mat_mc(:,i), eps));
        logS(:, i+1) = logS(:, i) + (r - f_mc_current.^2/2) * dt...
            + dt^(1/2) * f_mc_current .* (rho * noise_mat_3d_mc(:,i,1) + sqrt(1-rho^2) * noise_mat_3d_mc(:,i,2));
        %logS(:, i+1) = logS_mc_current;
        adj_J = min(J, num_steps_mc+1-i);
        for j = 1 : adj_J
            % notice `f_mc_current` and `f_mat_mc` here, `f_mat_mc` are
            % previous values to deduct
            V_mat_mc(:,i+j) = V_mat_mc(:,i+j) + (nu * dt^H * f_mat_mc(:,i) .* noise_mat_3d_mc(:,i,j+2) - ...
                dt^H * nu * f_mat_mc(:,i) .* kernel_dt(:,j+1) .* noise_mat_3d_mc(:,i,1)) / gamma_val;
            % V_mat_mc(:,i+j) = V_mat_mc(:,i+j) + (nu * f_mc_current .* noise_mat_3d_mc(:,i,j+2) - ...
            %     dt^(H-1/2) * nu * f_mat_mc(:,i) .* kernel_dt(:,j+1) .* noise_mat_3d_mc(:,i,1)) / gamma_val;
        end
    end


end

