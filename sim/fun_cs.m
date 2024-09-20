function [projs, smp_mtx] = fun_cs(M, N, data)%, smp_freqs, target)
    target = data.target;
%     smp_freqs = data.smp_freqs;
    smp_mtx = zeros(M, N);
    for i = 1 : M
        measures_indices = randperm(N, 3);
        smp_mtx(i, measures_indices) = 1;
    end
    projs = smp_mtx * target;
    
end

