
load('freq');
peak_locations_norm_x = [];
peak_locations_unc_norm_x = [];
noiseVec_x=[];
field_vals_x=[];
for k=1:11

    load(['ESR_meas_x_value_',num2str(k)]);
    load(['ESR_ref_x_value_',num2str(k)]);
    meas = squeeze(mean(mean(averaged_meas,1),2));
    ref = squeeze(mean(mean(averaged_ref,1),2));

    norm = (ref-meas)/mean(ref);

    [smooth_data, target_num_pks, target_guess] = getFitGuess(freq, ...
        norm, 0.002);

    [full_fit, full_params, ~, full_residuals, full_conf] = lorentzian_fit_lf(...
        freq,norm, 2, 2, target_num_pks, target_guess);

    [locs, locs_unc] = getFitVals(full_params, ...
        full_conf, 'Peak');

B_meas_XYZ = MagneticFieldCalculate_renana_and_ido(locs);
[phi, theta, r] = cart2sph(B_meas_XYZ(1), B_meas_XYZ(2),B_meas_XYZ(3));
phi = rad2deg(phi);
theta = rad2deg(theta);
field = [r,theta,phi];
field_vals_x(k,:) = field;
    peak_locations_norm_x(k,:) = locs;
    peak_locations_unc_norm_x(k,:) = locs_unc;
    noiseVec_x(k)=std(norm(1:20));

end
save('peak_locations_norm_x','peak_locations_norm_x')
save('peak_locations_unc_norm_x','peak_locations_unc_norm_x')
save('noiseVec_x','noiseVec_x')
save('field_vals_x','field_vals_x')

peak_locations_norm_x_2 = [];
peak_locations_unc_norm_x_2 = [];
noiseVec_x_2=[];
field_vals_x_2=[];

for k=1:11

    load(['ESR_meas_x_value_2_',num2str(k)]);
    load(['ESR_ref_x_value_2_',num2str(k)]);
    meas = squeeze(mean(mean(averaged_meas,1),2));
    ref = squeeze(mean(mean(averaged_ref,1),2));

    norm = (ref-meas)/mean(ref);
    [smooth_data, target_num_pks, target_guess] = getFitGuess(freq, ...
        norm, 0.002);
    if target_num_pks==8

    [full_fit, full_params, ~, full_residuals, full_conf] = lorentzian_fit_lf(...
        freq,norm, 2, 2, target_num_pks, target_guess);

    [locs, locs_unc] = getFitVals(full_params, ...
        full_conf, 'Peak');

    B_meas_XYZ = MagneticFieldCalculate_renana_and_ido(locs);
[phi, theta, r] = cart2sph(B_meas_XYZ(1), B_meas_XYZ(2),B_meas_XYZ(3));
phi = rad2deg(phi);
theta = rad2deg(theta);
field = [r,theta,phi];
field_vals_x_2(k,:) = field;
    peak_locations_norm_x_2(k,:) = locs;
    peak_locations_unc_norm_x_2(k,:) = locs_unc;
    noiseVec_x_2(k)=std(norm(1:20));
    else
        [smooth_data, target_num_pks, target_guess] = getFitGuess(freq, ...
        norm, 0.0015);
        [full_fit, full_params, ~, full_residuals, full_conf] = lorentzian_fit_lf(...
        freq,norm, 2, 2, target_num_pks, target_guess);

    [locs, locs_unc] = getFitVals(full_params, ...
        full_conf, 'Peak');

    B_meas_XYZ = MagneticFieldCalculate_renana_and_ido(locs);
[phi, theta, r] = cart2sph(B_meas_XYZ(1), B_meas_XYZ(2),B_meas_XYZ(3));
phi = rad2deg(phi);
theta = rad2deg(theta);
field = [r,theta,phi];
field_vals_x_2(k,:) = field;
    peak_locations_norm_x_2(k,:) = locs;
    peak_locations_unc_norm_x_2(k,:) = locs_unc;
    noiseVec_x_2(k)=std(norm(1:20));
    end
end
save('peak_locations_norm_x_2','peak_locations_norm_x_2')
save('peak_locations_unc_norm_x_2','peak_locations_unc_norm_x_2')
save('noiseVec_x_2','noiseVec_x_2')
save('field_vals_x_2','field_vals_x_2')


peak_locations_norm_x_3 = [];
peak_locations_unc_norm_x_3 = [];
noiseVec_x_3=[];
field_vals_x_3=[];

for k=1:11

    load(['ESR_meas_x_value_3_',num2str(k)]);
    load(['ESR_ref_x_value_3_',num2str(k)]);
    meas = squeeze(mean(mean(averaged_meas,1),2));
    ref = squeeze(mean(mean(averaged_ref,1),2));

    norm = (ref-meas)/mean(ref);

    [smooth_data, target_num_pks, target_guess] = getFitGuess(freq, ...
        norm, 0.002);
if target_num_pks==8
    [full_fit, full_params, ~, full_residuals, full_conf] = lorentzian_fit_lf(...
        freq,norm, 2, 2, target_num_pks, target_guess);

    [locs, locs_unc] = getFitVals(full_params, ...
        full_conf, 'Peak');

    B_meas_XYZ = MagneticFieldCalculate_renana_and_ido(locs);
[phi, theta, r] = cart2sph(B_meas_XYZ(1), B_meas_XYZ(2),B_meas_XYZ(3));
phi = rad2deg(phi);
theta = rad2deg(theta);
field = [r,theta,phi];
field_vals_x_3(k,:) = field;

    peak_locations_norm_x_3(k,:) = locs;
    peak_locations_unc_norm_x_3(k,:) = locs_unc;
    noiseVec_x_3(k)=std(norm(1:20));
else
    [smooth_data, target_num_pks, target_guess] = getFitGuess(freq, ...
        norm, 0.0015);
 [full_fit, full_params, ~, full_residuals, full_conf] = lorentzian_fit_lf(...
        freq,norm, 2, 2, target_num_pks, target_guess);

    [locs, locs_unc] = getFitVals(full_params, ...
        full_conf, 'Peak');

    B_meas_XYZ = MagneticFieldCalculate_renana_and_ido(locs);
[phi, theta, r] = cart2sph(B_meas_XYZ(1), B_meas_XYZ(2),B_meas_XYZ(3));
phi = rad2deg(phi);
theta = rad2deg(theta);
field = [r,theta,phi];
field_vals_x_3(k,:) = field;

    peak_locations_norm_x_3(k,:) = locs;
    peak_locations_unc_norm_x_3(k,:) = locs_unc;
    noiseVec_x_3(k)=std(norm(1:20));

end
end
save('peak_locations_norm_x_3','peak_locations_norm_x_3')
save('peak_locations_unc_norm_x_3','peak_locations_unc_norm_x_3')
save('noiseVec_x_3','noiseVec_x_3')
save('field_vals_x_3','field_vals_x_3')


peak_locations_norm_y = [];
peak_locations_unc_norm_y = [];
noiseVec_y=[];
field_vals_y=[];

for k=1:13

    load(['ESR_meas_y_value_',num2str(k)]);
    load(['ESR_ref_y_value_',num2str(k)]);
    meas = squeeze(mean(mean(averaged_meas,1),2));
    ref = squeeze(mean(mean(averaged_ref,1),2));

    norm = (ref-meas)/mean(ref);

    [smooth_data, target_num_pks, target_guess] = getFitGuess(freq, ...
        norm, 0.01);
if target_num_pks==8
    [full_fit, full_params, ~, full_residuals, full_conf] = lorentzian_fit_lf(...
        freq,norm, 2, 2, target_num_pks, target_guess);
 [locs, locs_unc] = getFitVals(full_params, ...
        full_conf, 'Peak');
B_meas_XYZ = MagneticFieldCalculate_renana_and_ido(locs);
[phi, theta, r] = cart2sph(B_meas_XYZ(1), B_meas_XYZ(2),B_meas_XYZ(3));
phi = rad2deg(phi);
theta = rad2deg(theta);
field = [r,theta,phi];
field_vals_y(k,:) = field;

    peak_locations_norm_y(k,:) = locs;
    peak_locations_unc_norm_y(k,:) = locs_unc;
    noiseVec_y(k)=std(norm(1:20));
else
    [smooth_data, target_num_pks, target_guess] = getFitGuess(freq, ...
        norm, 0.008);

    [full_fit, full_params, ~, full_residuals, full_conf] = lorentzian_fit_lf(...
        freq,norm, 2, 2, target_num_pks, target_guess);

B_meas_XYZ = MagneticFieldCalculate_renana_and_ido(locs);
[phi, theta, r] = cart2sph(B_meas_XYZ(1), B_meas_XYZ(2),B_meas_XYZ(3));
phi = rad2deg(phi);
theta = rad2deg(theta);
field = [r,theta,phi];
field_vals_y(k,:) = field;
    [locs, locs_unc] = getFitVals(full_params, ...
        full_conf, 'Peak');
    peak_locations_norm_y(k,:) = locs;
    peak_locations_unc_norm_y(k,:) = locs_unc;
    noiseVec_y(k)=std(norm(1:20));
end

end
save('peak_locations_norm_y','peak_locations_norm_y')
save('peak_locations_unc_norm_y','peak_locations_unc_norm_y')
save('noiseVec_y','noiseVec_y')
save('field_vals_y','field_vals_y')


peak_locations_norm_y_2 = [];
peak_locations_unc_norm_y_2 = [];
noiseVec_y_2=[];
field_vals_y_2=[];

for k=1:13

    load(['ESR_meas_y_value_2_',num2str(k)]);
    load(['ESR_ref_y_value_2_',num2str(k)]);
    meas = squeeze(mean(mean(averaged_meas,1),2));
    ref = squeeze(mean(mean(averaged_ref,1),2));

    norm = (ref-meas)/mean(ref);
    [smooth_data, target_num_pks, target_guess] = getFitGuess(freq, ...
        norm, 0.01);

    [full_fit, full_params, ~, full_residuals, full_conf] = lorentzian_fit_lf(...
        freq,norm, 2, 2, target_num_pks, target_guess);

    [locs, locs_unc] = getFitVals(full_params, ...
        full_conf, 'Peak');
B_meas_XYZ = MagneticFieldCalculate_renana_and_ido(locs);
[phi, theta, r] = cart2sph(B_meas_XYZ(1), B_meas_XYZ(2),B_meas_XYZ(3));
phi = rad2deg(phi);
theta = rad2deg(theta);
field = [r,theta,phi];
field_vals_y_2(k,:) = field;
    peak_locations_norm_y_2(k,:) = locs;
    peak_locations_unc_norm_y_2(k,:) = locs_unc;
    noiseVec_y_2(k)=std(norm(1:20));

end
save('peak_locations_norm_y_2','peak_locations_norm_y_2')
save('peak_locations_unc_norm_y_2','peak_locations_unc_norm_y_2')
save('noiseVec_y_2','noiseVec_y_2')
save('field_vals_y_2','field_vals_y_2')


peak_locations_norm_y_3 = [];
peak_locations_unc_norm_y_3 = [];
noiseVec_y_3=[];
field_vals_y_3=[];

for k=1:13

    load(['ESR_meas_y_value_3_',num2str(k)]);
    load(['ESR_ref_y_value_3_',num2str(k)]);
    meas = squeeze(mean(mean(averaged_meas,1),2));
    ref = squeeze(mean(mean(averaged_ref,1),2));

    norm = (ref-meas)/mean(ref);

    [smooth_data, target_num_pks, target_guess] = getFitGuess(freq, ...
        norm, 0.01);

    [full_fit, full_params, ~, full_residuals, full_conf] = lorentzian_fit_lf(...
        freq,norm, 2, 2, target_num_pks, target_guess);

    [locs, locs_unc] = getFitVals(full_params, ...
        full_conf, 'Peak');
B_meas_XYZ = MagneticFieldCalculate_renana_and_ido(locs);
[phi, theta, r] = cart2sph(B_meas_XYZ(1), B_meas_XYZ(2),B_meas_XYZ(3));
phi = rad2deg(phi);
theta = rad2deg(theta);
field = [r,theta,phi];
field_vals_y_3(k,:) = field;
    peak_locations_norm_y_3(k,:) = locs;
    peak_locations_unc_norm_y_3(k,:) = locs_unc;
    noiseVec_y_3(k)=std(norm(1:20));

end
save('peak_locations_norm_y_3','peak_locations_norm_y_3')
save('peak_locations_unc_norm_y_3','peak_locations_unc_norm_y_3')
save('noiseVec_y_3','noiseVec_y_3')
save('field_vals_y_3','field_vals_y_3')


peak_locations_norm_z = [];
peak_locations_unc_norm_z = [];
noiseVec_z=[];
field_vals_z=[];

for k=1:21

    load(['ESR_meas_z_value_',num2str(k)]);
    load(['ESR_ref_z_value_',num2str(k)]);
    meas = squeeze(mean(mean(averaged_meas,1),2));
    ref = squeeze(mean(mean(averaged_ref,1),2));

    norm = (ref-meas)/mean(ref);

    [smooth_data, target_num_pks, target_guess] = getFitGuess(freq, ...
        norm, 0.01);

    [full_fit, full_params, ~, full_residuals, full_conf] = lorentzian_fit_lf(...
        freq,norm, 2, 2, target_num_pks, target_guess);

    [locs, locs_unc] = getFitVals(full_params, ...
        full_conf, 'Peak');
B_meas_XYZ = MagneticFieldCalculate_renana_and_ido(locs);
[phi, theta, r] = cart2sph(B_meas_XYZ(1), B_meas_XYZ(2),B_meas_XYZ(3));
phi = rad2deg(phi);
theta = rad2deg(theta);
field = [r,theta,phi];
field_vals_z(k,:) = field;
    peak_locations_norm_z(k,:) = locs;
    peak_locations_unc_norm_z(k,:) = locs_unc;
    noiseVec_z(k)=std(norm(1:20));

end
save('peak_locations_norm_z','peak_locations_norm_z')
save('peak_locations_unc_norm_z','peak_locations_unc_norm_z')
save('noiseVec_z','noiseVec_z')
save('field_vals_z','field_vals_z')
