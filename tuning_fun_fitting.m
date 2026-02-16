
% show the tuning function 
kappa = 3; 
Amp = 30; 
B = 10; 
tf = @(x) Amp*exp(kappa*cos(x)-kappa) + B; 

% simulate data
n = 12; 
x = linspace(-pi,pi-2*pi/n,n);
y = tf(x) + randn(size(x));

% fit data with least squares
fn = @(x, P) P(1)*exp(P(2)*cos(x)-P(2)) + P(3); 
obj_fn = @(P) sum((fn(x, P) - y).^2,'all');
p0 = rand(1,3)*3; 
P = fminunc(obj_fn, p0)

clf
nexttile
hold on 
xx = linspace(-pi,pi,1e3);
plot(xx, tf(xx))
scatter(x, y, 'filled')
plot(xx, fn(xx, P))


legend({'Ground truth','Simulated data','Fit'})
xlabel('Orientation (2x rad.)')
ylabel('Firing rate')
title('Tuning function')
