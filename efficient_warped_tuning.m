
% prior distribution
s = linspace(-pi,pi,100); 
p_s = 2-abs(sin(s)); 
p_s = p_s ./ sum(p_s);  % normalise 

% create function that warps orientation using the 
cdf_p_s = cumsum(p_s); 
F = @(x) interp1(s, cdf_p_s*2*pi-pi, x); 
Finv = @(x) interp1(cdf_p_s*2*pi-pi, s, x);

% circular tuning curves for M neurons
M = 8; 
phis = -pi:2*pi/M:pi-2*pi/M;
kappa = 3; 
tf = @(x) exp(kappa*cos(x)-kappa);

tuning = tf(s'-phis);
warped_tuning = tf(F(s)'-phis);

clf
nexttile
hold on 
plot(s, p_s)
xlabel('Orientation'); ylabel('Probability')
title('Orientation prior')

nexttile
hold on 
plot(s, cdf_p_s)
xlabel('Orientation'); ylabel('Cumulative probability')
title('Cumulative prior')

nexttile
hold on 
plot(s, tuning)
xlabel('Orientation'); ylabel('Firing rate')
title('Tuning curves')

nexttile
hold on 
plot(s, warped_tuning)
xlabel('Orientation'); ylabel('Firing rate')
title('Warped tuning curves')