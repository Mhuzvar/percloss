close all
clear

%% setting variables

fs = 44100;
nfft = floor(2048*(fs/48000));
step = fix(nfft/2)+1;

f = 1019.5;

slen = 9*step+nfft;
t = [1:slen]/fs;
swave = sin(2*pi*f*t);
nfac = 0;
for i = 1:10
    chunk = swave((i-1)*step+1:(i-1)*step+nfft);
    spec = fft(chunk);
    nfac = nfac+max(abs(spec))/10;
    figure(1)
    hold on
    plot(chunk)
    figure(2)
    hold on
    plot(abs(spec))
end
yline(nfac)
hold off