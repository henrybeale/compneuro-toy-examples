
% parameters 
C = 1; 
R = 10; 
theta = -55; 
u_reset = -80; 
u_0 = -70; 
sigma = .5;  % noise variance

% time 
fs = 1024; 
dt = 1/fs*1e3;  % ms
time = 0:dt:200;

% input stimulus 
I = 2 .* (time > 10 & time < 110); 

u = time*0 + u_0;
spikes = time*0;
for t = 1:numel(time)-1
    if u(t) >= theta
        u(t+1) = u_reset;
        spikes(t) = 1;
    else
        du = ((u_0-u(t))/R + I(t))/C + randn*sigma;
        u(t+1) = u(t) + dt*du;
    end
end

spike_times = find(spikes);

clf
nexttile
hold on 
plot(time, u)
xline(time(spikes==1))

nexttile
hold on 
plot(time, I)