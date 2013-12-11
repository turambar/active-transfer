c1 = 5 + 2*sqrt(2);
c2 = 5;
C0 = [ 8 2 1 0.25 ];
k = [ 10 100 1000 10000 100000 ];
col = { 'b', 'c', 'r', 'm', 'g' };
G = 0:0.01:1;

figure;
for i=1:length(C0)
    subplot(2, 2, i)
    title(sprintf('C0=%d', C0(i)))
    xlabel('G_k')
    ylabel('P_k')
    hold on;
    for j=1:length(k)
        P = zeros(size(G));
        for gi=1:length(G)
            Gbound = iwal_cal_gbound(k(j), C0(i));
%             Gbound = sqrt(C0(i) * log(k(j)) / (k(j)-1)) + C0(i) * log(k(j)) / (k(j)-1);

            if abs(G(gi)) <= Gbound
                fprintf('G=%.5f\t<= Gb=%.5f\n', G(gi), Gbound);
                P(gi) = 1;
            else
                temp = iwal_cal_query_probability(k(j), G(gi), C0(i), c1, c2);
                P(gi) = temp;
            end
        end
        plot(G, P, col{j});
%         keyboard
    end
    legend('10', '100', '1000', '10000', '100000')
    title(sprintf('c0=%.2f c1=%.2f c2=%.2f', C0(i), c1, c2))
end

c1 = 1;
c2 = 1;
figure;
for i=1:length(C0)
    subplot(2, 2, i)
    title(sprintf('C0=%d', C0(i)))
    xlabel('G_k')
    ylabel('P_k')
    hold on;
    for j=1:length(k)
        P = zeros(size(G));
        for gi=1:length(G)
            Gbound = iwal_cal_gbound(k(j), C0(i));
%             Gbound = sqrt(C0(i) * log(k(j)) / (k(j)-1)) + C0(i) * log(k(j)) / (k(j)-1);
            if abs(G(gi)) <= Gbound
                fprintf('G=%.5f\t<= Gb=%.5f\n', G(gi), Gbound);
                P(gi) = 1;
            else
                temp = iwal_cal_query_probability(k(j), G(gi), C0(i), c1, c2);
                P(gi) = temp;
            end
        end
        plot(G, P, col{j});
%         keyboard
    end
    legend('10', '100', '1000', '10000', '100000')
    title(sprintf('c0=%.2f c1=%.2f c2=%.2f', C0(i), c1, c2))
end
