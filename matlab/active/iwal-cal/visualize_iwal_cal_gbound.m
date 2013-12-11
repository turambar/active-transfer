C0 = 2.^(-2:4);
col = { 'k-', 'b-', 'g-', 'r-', 'c-', 'm-', ...
        'k--', 'b--', 'g--', 'r--', 'c--', 'm--' };

figure;
subplot(2, 2, 1)
hold on;
K   = [10 100:100:10000 ];
for i=1:length(C0)
    gbound = iwal_cal_gbound(K, C0(i));
    plot(K, gbound, col{i})
end
axis([ min(K) max(K) 0 2 ])
legend('1/4', '1/2', '1', '2', '4', '8', '16', 'Location', 'Best')

subplot(2, 2, 2)
hold on;
K   = 10:10:1000;
for i=1:length(C0)
    gbound = iwal_cal_gbound(K, C0(i));
    plot(K, gbound, col{i})
end
axis([ min(K) max(K) 0 1 ])
legend('1/4', '1/2', '1', '2', '4', '8', '16', 'Location', 'Best')

subplot(2, 2, 3)
hold on;
K   = [10 100:100:10000 ];
for i=1:length(C0)
    gbound = iwal_cal_gbound(K, C0(i));
    plot(K, gbound, col{i})
end
axis([ min(K) max(K) 0 0.5 ])
legend('1/4', '1/2', '1', '2', '4', '8', '16', 'Location', 'Best')

subplot(2, 2, 4)
hold on;
K   = 10:10:1000;
for i=1:length(C0)
    gbound = iwal_cal_gbound(K, C0(i));
    plot(K, gbound, col{i})
end
axis([ min(K) max(K) 0 0.5 ])
legend('1/4', '1/2', '1', '2', '4', '8', '16', 'Location', 'Best')