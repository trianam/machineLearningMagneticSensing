function [smooth_data, num_pks, initial_guess,locs] = getFitGuess(t, data, ...
    contrast, prominence, smoothSpan)
% call with [smooth_data, num_pks, initial_guess,locs] = getFitGuess(x,y,0.006)
% check if num_pks is 8

% Guess the parameters of the peaks in data based on variable 't'
%smooth_data = smooth(data,0.06,'loess'); %kills the hyperfine, we dont need it.
if smoothSpan > 0
    smooth_data = smooth(data,smoothSpan,'loess'); %kills the hyperfine, we dont need it.
else
    smooth_data = data;
end
[~,locs] = findpeaks(smooth_data ,'MinPeakHeight', contrast, 'MinPeakProminence', prominence);%in the last parameter insert the contrast of the smaiiest peak
num_pks=length(locs);
initial_guess=zeros(1,num_pks+1);
for i=1:num_pks
    initial_guess(2*i-(2-i))=t(locs(i)); %peak location
    initial_guess(2*i-(2-i)+1)= 7;     %width
    %initial_guess(2*i-(2-i)+2)=-abs(smooth_data(locs(i))-min(smooth_data))/abs(min(smooth_data));        %contrast
    initial_guess(2*i-(2-i)+2)=-abs(smooth_data(locs(i))-min(smooth_data));
    %initial_guess(2*i-(2-i)+2)=0.005;
end
initial_guess(num_pks*3+1)=min(smooth_data);  %offset
end