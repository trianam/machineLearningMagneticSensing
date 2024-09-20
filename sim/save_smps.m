function []= save_smps(j)
    sz_str = strcat('./simulation_testing');
    var_name = sprintf('CS_%d_.mat', j);
    name = fullfile(sz_str, var_name);
    load(name);
    fig = figure;
    plot(data_struct.smp_freqs, data_struct.target);
    title('Magnetic field: ' + string(data_struct.magnetic_field) + ' [Gauss],  \Theta: ' + string(data_struct.B_theta) + ', \Phi: ' + string(data_struct.B_phi))
    subtitle('Large field: B mag \in [90, 120],  \Theta \in [30, 34],  \Phi \in [17, 21]');
    xlabel('MW Frequency [MHz]');
    ylabel('Absorption normalized (Arb. Units)');
    save_name = fullfile(sz_str, 'sample_' + string(j) + '.jpg');
    saveas(fig, save_name); 
    close all
%     title('$\mathcal{Z}$','Interpreter','latex')

end
