function logS = logS_simu_euler_scheme(spotV, num_steps_mc, t, num_path_mc, ...,
    theta, rho, lambda, nu, H, noise_mat_3d_mc, r, eps)
    %EULER_SCHEME Euler scheme simulation of rough Heston model
    % INPUT and OUTPUT align with those in `standard_hybrid.m`

    dt = t / num_steps_mc;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1); % initialize variance paths

    %logS_mc_current = zeros(num_path_mc,1);
    logS = zeros(num_path_mc,num_steps_mc+1);

    gamma_val = gamma(H+1/2);
    % resize noise matrix (coupling strategy)
    noise_mat_3d_mc = resize_noise_mat(noise_mat_3d_mc, num_steps_mc);

    for i = 1:num_steps_mc
        f_mc_current = sqrt(max(V_mat_mc(:,i),eps));
        logS(:, i+1) = logS(:, i) + (r - f_mc_current.^2/2) * dt + ...
            dt^(1/2) * f_mc_current .* (rho*noise_mat_3d_mc(:,i,1) + sqrt(1-rho^2)*noise_mat_3d_mc(:,i,2));
        %logS(:, i+1) = logS_mc_current;
        for j = 1:i
            f_mc_j = sqrt(max(V_mat_mc(:,j),eps));
            inner = lambda * (theta - f_mc_j.^2) * dt + dt^(1/2) * nu * f_mc_j .* noise_mat_3d_mc(:,j,1);
            V_mat_mc(:,i+1) = V_mat_mc(:,i+1) + ((i-j+1)*dt)^(H-1/2) * inner / gamma_val;
        end
    end

end

