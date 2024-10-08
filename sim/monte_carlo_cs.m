n = 0;
%m = 99999; %Number of data sets generated by the code - each have different magnetic field parameters
m=1;

data_type = 'test';
test_simulation = 0; % if 1 saves images and data.
num_of_measures = 50; %Number of measurement points (projections)
center_freq = 2875; % MHz
half_window_size = 385; %in MHz
% freqs_to_try = ones(1, n) * 3;%300 samples of 3 freqs together. Usages: 3 or [2,3,4];
freqs_to_try = 3; %simultanious frequencies
N=100;  %num of frequencies within frequency window - linespace
noiseSigma=0.001; %expressed as a fraction of the mean


if test_simulation == 1
    sz_str = strcat('./simulation_testing/');
else
    sz_str = strcat('../data/large_magnetic_90_120-measures_',num2str(num_of_measures),'-N_',num2str(N),'-noise_',num2str(noiseSigma),'/', data_type, '/');
end

if ~exist(sz_str, 'dir')
    mkdir(sz_str)
end

%% Get diamond simulation
for j = n : m
data = mock_diamond2_new(N, center_freq, half_window_size, noiseSigma);
[projs, smp_mtx] = fun_cs(num_of_measures, N, data);

     t = cputime;
                     
        data_struct = struct();
        data_struct = setfield(data_struct, 'projs', projs);
        data_struct = setfield(data_struct, 'magnetic_field', data.B_mag);
        data_struct = setfield(data_struct, 'B_theta', data.B_theta);
        data_struct = setfield(data_struct, 'B_phi', data.B_phi);
        data_struct = setfield(data_struct, 'relevant_window', data.relevant_window);
        data_struct = setfield(data_struct, 'smp_freqs', data.smp_freqs);
        data_struct = setfield(data_struct, 'peak_locs', data.peak_locs);
        data_struct = setfield(data_struct, 'B_projs', data.B_projs);
        data_struct = setfield(data_struct, 'B_vec', data.B_vec);
        data_struct = setfield(data_struct, 'target', data.target);
        data_struct = setfield(data_struct, 'sig', data.sig);
        data_struct = setfield(data_struct, 'signal', data.signal);
        data_struct = setfield(data_struct, 'full_window_size_MHz', 2 * half_window_size);
        data_struct = setfield(data_struct, 'num_pts', data.num_pts);
            
        %switch data_type
        %    case 'train' 
        %        var_name = sprintf('CS_%d_.mat', j);
        %    case 'valid' 
        %        var_name = sprintf('valid_CS_%d_.mat', j);
        %    case 'test' 
        %        var_name = sprintf('test_CS_%d_.mat', j);
        %end
        
        var_name = sprintf('CS_%d.mat', j);
        % PAY ATTENTION TO THE NUMBER SAMPS! DO NOT OVERRIDE!!
        save(fullfile(sz_str, var_name), 'data_struct', '-v7.3');
        
        if test_simulation == 1
            save_smps(j);
        end
        
        t = cputime-t;
        disp(['Sample ' num2str(j) ' Took ' num2str(t) ' seconds total.'])
%         clear 
end

exit
 
% in = [];
% in.tau = 0.0002;
% delx_mode = 'mil';
% in.delx_mode = delx_mode;
% in.debias = 1;
% in.verbose = 0;
% in.plots = 0;
% in.record = 0;
% in.Te = 8;
% in.nonneg = 1;
% 
% x = l1homotopy(smp_mtx, projs, in);
