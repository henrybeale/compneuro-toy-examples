
% simulate from the ddm 
n_trials = 10000;
rt = nan(n_trials,1); 
choice = nan(n_trials,1);

% parameters
v = .8; % drift rate
a = 1; % threshold 
z = .5; % starting point bias 
t = .2; % non-decision time
s = 1; % noise variance 

% time vector for simulation
dt = .001;
time = 0:dt:25;  % seconds

x = nan(n_trials, numel(time));
for n = 1:n_trials
    % start point
    x(n,1) = z*a; 
    
    for j = 2:numel(time)
        % cumulative evidence
        x(n,j) = x(n,j-1) + v*dt + s*sqrt(dt)*randn; 

        % threshold crossin
        if x(n,j)>=a || x(n,j)<=0
            choice(n) = 1; 
            rt(n) = t + dt*j; % add non-decision time
            if x(n,j)<=0
                choice(n) = choice(n)*-1;
            end
            break
        end
    end
end

clf
nexttile
hold on
yline([0, a])
for n = 1:10
    plot(time(time < rt(n)), x(n,time<rt(n)))
    scatter(rt(n)-t, a*(choice(n)+1)/2, 'k', 'filled')
end
xlabel('Decision time')
ylabel('Evidence')
title('Simulated trials')

nexttile
hold on 
rdat = rt .* choice; 

xx = linspace(-3,3,1000);
[pdf, xx] = kde(rdat, 'evaluationpoints',xx,'ProbabilityFcn','pdf'); 
plot(xx, pdf)
xlabel('Signed reaction time')
ylabel('Density')
title('RT-choice distribution')