
% set up a dense grid
dgrid = 1e3; % grid density
x = linspace(0, 100, dgrid);  
s = linspace(0, 100, dgrid); 

% prior over world-states, s
p_s = normpdf(s, 50, 5);  
p_s = p_s ./ sum(p_s);

% likelihood of x for every state s
p_x_s = nan(dgrid,dgrid); 
likfun = @(x,s) normpdf(x, s, 10);
for i = 1:dgrid
    p_x_s(i,:) = likfun(x,s(i));  % [s, x]
end

% compute posteriors for every observed x
p_s_x = nan(dgrid,dgrid); 
p_x = nan(1,dgrid);
for i = 1:dgrid
    num = p_x_s(:,i) .* p_s'; 
    p_x(i) = sum(num);  % marginal evidence for x
    p_s_x(:,i) = num./p_x(i);
end

p_x = p_x ./ sum(p_x);

% policy: apply deterministic MAP to every observed x (i.e. observer chooses
% the most probable s value based on posteriors)
a = nan(dgrid,1);  % chosen state estimate for each x 
for i = 1:dgrid
    [~, max_ind] = max(p_s_x(:,i));
    a(i) = s(max_ind);
end

% reward: deterministic function of action, a, and true world-state, s. 
r = -abs(a'-s');  % distance/error [s, a(x)]

% utility is an identity function of reward (i.e. all external rewards
% treated identically by the observer)
u = r; 

% expected utility given policy:
E_u = sum(p_x .* sum(p_s_x.*r, 1))  % eq 2 with deterministic variables


clf
nexttile
hold on 
plot(s,p_s)
% plot(s, sum(p_s_x,1))
xlabel('s')
title('Prior')

nexttile
hold on 
plot(x-mean(x), likfun(x-mean(x),0))
xlabel('x - s')
title('Likelihood')

nexttile
hold on 
plot(x, p_x)
xlabel('x')
title('Marginal Evidence')

nexttile
hold on 
imagesc(x,s,p_x_s)
colorbar
xlabel('x'); ylabel('s')
title('Likelihoods')

nexttile
hold on 
imagesc(x,s,p_s_x)
colorbar
xlabel('x'); ylabel('s')
title('Posteriors')

nexttile
hold on 
plot(x, a)
xlabel('x'); ylabel('a')
title('Actions (MAP policy)')

nexttile
hold on 
imagesc(s,a,r)
colorbar
xlabel('a(x)'); ylabel('s')
title('Reward')
