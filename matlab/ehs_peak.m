close all
clear

%% Generating input
% assuming BxF (batch x frequency)

npts=128;
f = 1:npts;
x = [4*(f/npts).*(1-f/npts).^4+0.2*sin(pi/2+6*pi*f/npts); 5*(f/npts).*(1-f/npts).^4+(0.3*exp(-5*f/npts)).*sin(5*pi/8+6*pi*f/npts)];
x = x+0.009*randn(2,npts);
figure(1)
plot(x')
hold on

%% peak detection

if npts>32
    N=8;                                    % fine from 64 up to 128 (safely)
    if npts>128
        N=16;                               % fine for 256 and 512 (larger should not be possible)
    end
    xf = filtfilt(ones(1,N)./N, [1,0],x')';
else
    xf=x;                                   % 32 and shorter should not be smoothed
end

range = max(x,[],2)-min(x,[],2);


xd = [[0;0], xf(:,2:end)-xf(:,1:end-1)];
sgn = [xd(:,1:end-1).*xd(:,2:end), [0;0]];

extidx = sgn<0;
pkidx = [sign(xd(:,1:end-1))-sign(xd(:,2:end))==2, zeros(2,1,"logical")];
validx = [sign(xd(:,1:end-1))-sign(xd(:,2:end))==-2, zeros(2,1,"logical")];


for i=[1,2]
    scatter(f(pkidx(i,:)),   0.01+x(i, pkidx(i,:))','g^')
    scatter(f(validx(i,:)), -0.01+x(i, validx(i,:))','rv')
    scatter(f(extidx(i,:)),x(i,extidx(i,:)),'k+')
end



hold off

% N=5;
% figure
% freqz([1,0.85],[1,0]);
% hold on
% freqz(ones(1,N)./N,1);
