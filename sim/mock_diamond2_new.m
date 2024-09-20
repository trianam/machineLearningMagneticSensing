classdef mock_diamond2_new < handle
    % To see linshape: 
    % plot(cs.smp_freqs, data_struct.target)
    % title('magnetic field: ' + string(data_struct.magnetic_field) + ' [G],  Theta: ' + string(data_struct.B_theta) + ', Phi: ' + string(data_struct.B_phi))
    % ylable
    
    properties
        num_pts;
        real_num_poits;
        curr_cs_per;
        
        relevant_window;

        B_theta;
        B_phi;
        B_mag;
        center_freq;
        half_window_size;
        
        % Noise Simulation
        add_noise;
        sigma;
        %add_noise = true;
        %sigma = 0.002; %expressed as a fraction of the mean
        
        % Target Peak Amplitude
        peak_amp = 0.01;
        
        % Derived Quantities
        peak_locs = []; % will contain peak locations based on Magnetic Field
        smp_freqs = []; % where we measure in our simulation
        real_smp_freqs = [];
        target = []; % lineshape we want to reconstruct
        
        % model of what we actually measure
        sig = [];
        ref = [];
        B_projs;
        B_vec;
        
        MHz_full_window;
        % for combining 'sig' and 'ref' into a single array for compatibility
        % with adaptive_reconstruction class
        signal = zeros(1,1,2);
    end
    
    properties %(Hidden)
        % Orientations of Diamond (in lab frame) assumed fixed
        diamond_unit_vecs = (1/sqrt(3))*[1 1 1;...
            1 -1 -1;...
            -1 1 -1;
            -1 -1 1];
        
        % Lorentzian Properties
        % center_freq = 2875; % MHz
        width = 10; % MHz
        kernel = lorentzian_kernel('N');
        move_center = false;
        center_offset = 0;
        shifts
        
        raster_flag = false;
        
        chk_noise = [];
    end
    
    methods
        function obj = mock_diamond2_new(N, center_freq, half_window_size, noiseSigma)
%         obj.B_theta = 50;
%         obj.B_phi = 20;
%          
        obj.B_theta = (34 - 30).*rand(1,1) + 30;
        obj.B_phi = (22 - 18).*rand(1,1) + 18 - 1;
        obj.B_mag = (120 - 90).*rand(1,1) + 90; % large values.
%         obj.B_mag = (120 - 55).*rand(1,1) + 55; % small values.
        
        obj.num_pts = N;
        obj.center_freq = center_freq;
        obj.half_window_size = half_window_size;

        if noiseSigma > 0
            obj.add_noise = true;
        else
            obj.add_noise = false;
        end
        obj.sigma = noiseSigma;

        % mock_experiment with no arguments assumes default values
        % and creates a mock lineshape for running a Compressed
        % Sensing simulation

        % Calculate Peak Locations given the Magnetic Field
        B_vec = obj.pol2xyz();
        obj.B_vec = B_vec;
        B_projs = sum(obj.diamond_unit_vecs.*B_vec,2);
        obj.B_projs = B_projs;
        detunings = 2.8*B_projs; % 2.8 [MHz/G] is the MHz/G ratio distance from center.
                                 % B_proj is the B|| on each NV orientation. 
                                 % detuning is the actuall distance
                                 % in MHz fron the center for each
                                 % orientation dimension.
%         full_peak_window = max(obj.center_freq+detunings)-min(obj.center_freq-detunings);
        % full_peak_window is the full (MHz!!) window of all
        % lineshape since it takes once the maximum MHz point by the
        % max(center + the 4D detuning) minus the min(center -
        % detuning).

        if obj.move_center
            max_to_move = 20;
            obj.center_offset = max_to_move*(2*rand-1);
            %obj.center_offset =30;
            obj.shifts = 0.5*(2*rand(8,1)-1);
        else
            obj.shifts = 0;
            obj.center_offset = 0;
        end

        obj.peak_locs = sort([obj.center_freq - detunings; ...
            obj.center_freq+detunings]) + obj.center_offset +...
            obj.shifts;
        
        obj.relevant_window = obj.peak_locs(end) - obj.peak_locs(1);

        % half_full_peak_window = 400;
%         half_full_peak_window = 385; % THIS IS THE OLD: obj.half_window_size

        
        obj.smp_freqs = obj.center_freq+...
        linspace(-obj.half_window_size, obj.half_window_size, obj.num_pts);  
      
        % get actual lineshape and simulated signal and reference
        obj.target = obj.getLineShape(obj.smp_freqs);
        % get a mock raster scan
        obj.sig = obj.getRaster(obj.smp_freqs);
%                 [obj.sig, obj.ref] = obj.getRaster(obj.smp_freqs);

        end
     
        
        function  vec = pol2xyz(obj, varargin)
            %pol2xyz Converts vector in spherical to rectangular
            %coordinates
            %   Assumes angles are given in degrees.
            if length(varargin)==2
                theta = varargin{1};
                phi = varargin{2};
                vec = [sind(theta)*cosd(phi) sind(theta)*sind(phi) cosd(theta)];
            elseif length(varargin)==3
                mag = varargin{1};
                theta = varargin{2};
                phi = varargin{3};
                vec = mag*[sind(theta)*cosd(phi) sind(theta)*sind(phi) cosd(theta)];
            elseif isempty(varargin)
                % defaults to magnetic field vector
%                  obj.B_theta = (50 - 10).*rand(1,1) + 10; % was B_phi = 70;
%                  obj.B_phi = (90 - 50).*rand(1,1) + 50; % was B_theta = 20;
%                 obj.B_mag = (125 - 45).*rand(1,1) + 45;
               vec = obj.B_mag*[sind(obj.B_theta)*cosd(obj.B_phi) sind(obj.B_theta)*sind(obj.B_phi) cosd(obj.B_theta)];
               
%                 mag = randi([90 110]); % was obj.B_mag;
%                 theta = (22 - 18).*rand(1,1) + 18; % was obj.B_theta;
%                 phi = (71 - 69).*rand(1,1) + 69; % was obj.B_phi;
%                 vec = mag*[sind(theta)*cosd(phi) sind(theta)*sind(phi) cosd(theta)];
            else
                error('Vector should be given as [R,theta,phi] or [theta,phi].')
            end
        end
        
        function  lineshape = getLineShape(obj,freqs)
            %get the lineshape for the diamond at frequencies 'freqs'
            amp = obj.peak_amp/(2/(pi*obj.width));
            L = @(detuning) amp*obj.kernel(freqs,obj.width, detuning);
            lineshape = zeros(length(freqs),1);
            for i=1:length(obj.peak_locs)
                lineshape = lineshape + L(obj.peak_locs(i))';
            end
        end
        
        function  targ = getRaster(obj,varargin)
            %Get a Simulated Raster Scan with or without noise
            if isempty(varargin)
                freqs = obj.smp_freqs;
            else
                freqs = varargin{1};
            end
            
            
            if obj.raster_flag             
                targ = obj.target;
            else
                npts = length(freqs);
                raster_lineshape = obj.getLineShape(freqs);
                
                
                if obj.add_noise
                    sim_ref = ones(npts,1)+obj.sigma*randn(npts,1);
                    sim_sig = sim_ref-raster_lineshape+obj.sigma*randn(npts,1);
                else
                    sim_ref = ones(npts,1);
                    sim_sig = sim_ref-obj.target;
                end
                targ = (sim_ref-sim_sig)/mean(sim_ref);
            end
            
            
        end
        
        function  obj = getMeasurement(obj,sample_freqs)
            %Get a Simulated Raster Scan with or without noise
            num_samples = length(sample_freqs);
            indices = zeros(num_samples,1);
            for i=1:(num_samples)
                indices(i) = find(obj.smp_freqs==sample_freqs(i));
            end
            
            if obj.add_noise
                if obj.raster_flag
                    ref_noise = obj.sigma*randn*mean(obj.ref);
                    obj.signal(1,1,2) = mean(obj.ref(indices))+ref_noise;
                    obj.signal(1,1,1) = ref_noise+sum(obj.sig(indices))+obj.sigma*randn*mean(obj.ref);
                    obj.chk_noise = [obj.chk_noise; ref_noise];
                    
                else
                    obj.signal(1,1,2) = 1+obj.sigma*randn;
                    %obj.signal(1,1,1) = obj.signal(1,1,2)+obj.sigma*randn-mean(obj.target(indices));
                    obj.signal(1,1,1) = obj.signal(1,1,2)+obj.sigma*randn-sum(obj.target(indices));
                end
                
                
                
            else
                obj.signal(1,1,2) = 1;
                obj.signal(1,1,1) = 1-sum(obj.target(indices));
            end
            
        end
        
        function  [] = show_plots(obj)
            % plot
            figure
            hold on
            title('Orientation of Magnetic Field and Diamond')
            for i=1:length(obj.diamond_unit_vecs)
                x = [0 obj.diamond_unit_vecs(i,1)];
                y = [0 obj.diamond_unit_vecs(i,2)];
                z = [0 obj.diamond_unit_vecs(i,3)];
                line(x,y,z,'Color','Blue')
                scatter3(x(2),y(2),z(2),'b','filled')
                
            end
            
            B_vec = [sind(obj.B_theta)*cosd(obj.B_phi) sind(obj.B_theta)*sind(obj.B_phi) cosd(obj.B_theta)];
            quiver3(0,0,0,B_vec(1),B_vec(2),B_vec(3),'Color','Red');
            view(-28.2,10.2);
            
            figure
            plot(obj.smp_freqs,obj.target)
            grid on
            xlabel('Frequency (MHz)')
            ylabel('Zero-Mean Lineshape')
            title_str = ['Lineshape Model, ' num2str(obj.B_mag)...
                ' G, \theta=' num2str(obj.B_theta) char(176) ', \phi=' ...
                num2str(obj.B_phi) char(176)];
            title(title_str)
            xlim([obj.smp_freqs(1) obj.smp_freqs(end)])
            
        end
        
        
    end
end



function y = lorentzian_kernel(varargin)
% Different choices of lorentzian kernel
% OUTPUT is an anonymous function whose form is a 1-D lorentzian with
% properties determined by varargin
% INPUT is a string:
% 'ZMN' or an empty string for Zero-Mean Normalized Lorentzian
% 'ZMU' for Zero-Mean Unormalized
% 'N' for Normalized
% 'U' for Unnormalized
switch length(varargin)
    case 0
        y = @(x,w,x0) (2/(pi*w)).*(1./(1+(2.*(x-x0)/w).^2))-...
            mean((2/(pi*w)).*(1./(1+(2.*(x-x0)/w).^2)));
    case 1
        if ~ischar(varargin{1})
            error('Choice for Lorentzian Kernel should be "ZMU", "ZMN", "U", or "N".');
        end
        switch varargin{1}
            case 'ZMU'
                y = @(x,w,x0) (1./(1+(2.*(x-x0)./w).^2))-...
                    mean((1./(1+(2.*(x-x0)./w).^2)));
            case 'ZMN'
                y = @(x,w,x0) (2/(pi*w)).*(1./(1+(2.*(x-x0)./w).^2))-...
                    mean((2/(pi*w)).*(1./(1+(2.*(x-x0)./w).^2)));
            case 'U'
                y = @(x,w,x0) (1./(1+(2.*(x-x0)./w).^2));
            case 'N'
                y = @(x,w,x0) (2/(pi*w)).*(1./(1+(2.*(x-x0)./w).^2));
            otherwise
                error('Choice for Lorentzian Kernel should be "ZMU", "ZMN", "U", or "N".');
        end
    otherwise
        error('Choice for Lorentzian Kernel should be "ZMU", "ZMN", "U", or "N".');
end
end

