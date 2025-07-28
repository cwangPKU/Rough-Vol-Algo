function logS = logS_simu_hybrid_scheme2(spotV, num_steps_mc, t, num_path_mc, forward_curve,...
    theta, rho, lambda, nu, H, noise_mat_3d_mc, J, r, eps)


    dt = t / num_steps_mc;
    V_mat_mc = spotV * ones(num_path_mc, num_steps_mc+1) + [0, forward_curve]; % initialize variance paths

    logS = zeros(num_path_mc,num_steps_mc+1);
    % resize noise matrix (coupling strategy)
    noise_mat_3d_mc = resize_noise_mat(noise_mat_3d_mc, num_steps_mc);


    kernel_length = num_steps_mc*2;
    kernel_dt = (linspace(1,kernel_length,kernel_length).^(H+1/2) - ...
        linspace(0,kernel_length-1,kernel_length).^(H+1/2)) / (H+1/2);
    gamma_val = gamma(H+1/2);

    for i = 1:num_steps_mc
        f_mc_current = sqrt(max(V_mat_mc(:,i), eps));
        logS(:, i+1) = logS(:, i) + (r - f_mc_current.^2/2) * dt...
            + dt^(1/2) * f_mc_current .* (rho * noise_mat_3d_mc(:,i,1) + ...
            sqrt(1-rho^2) * noise_mat_3d_mc(:,i,2));
        % update the whole variance path
        j_vec = i+1 : num_steps_mc+1;
        adj_J = min(J, num_steps_mc+1-i);
        tmpd = nu / gamma_val * f_mc_current;
        tmpc = -dt^(H+1/2) * lambda / gamma_val * V_mat_mc(:,i); 
        V_mat_mc(:,j_vec) = V_mat_mc(:,j_vec) + tmpc * kernel_dt(j_vec-i)...
            + tmpd * dt^H .* [zeros(1,adj_J), kernel_dt(J+1:num_steps_mc+1-i)] .* noise_mat_3d_mc(:,i,1); 
        for j = 1:adj_J  % adjustment to nearby terms (more accurate noises)
            V_mat_mc(:,i+j) = V_mat_mc(:,i+j) + dt^H * tmpd .* noise_mat_3d_mc(:,i,j+2);
        end
    end
end

