% CPGA vs PGA vs SGA porovnanie
% -----------------------------------------

clc; clear; close all;

numgen = 1500;    % number of generations
lpop = 250;       % number of chromosomes in population (total)
lstring = 10;     % number of genes in a chromosome
M = 500;          % search space max
runs = 5;         % pocet behov pre priemerovanie

Space = [ones(1,lstring)*(-M); ones(1,lstring)*M];
Delta = Space(2,:)/100;

% pre CPGA
rows = 10; cols = 25;  % 10x25 grid
assert(rows*cols == lpop, 'rows*cols must equal lpop');

% pre PGA
numIslands = 5;
perIsland = lpop / numIslands;
migInterval = 50;
migSize = 5;

% vysledkove polia
allCPGA = zeros(runs, numgen);
allPGA = zeros(runs, numgen);
allSGA = zeros(runs, numgen);
bestCPGA = zeros(1, runs);
bestPGA = zeros(1, runs);
bestSGA = zeros(1, runs);
avgCPGA = zeros(1, numgen);
avgPGA = zeros(1, numgen);
avgSGA = zeros(1, numgen);

figure(1); clf; hold on; grid on;
title('CPGA vs PGA vs SGA - priemer a najlepsie behy');
xlabel('Generacia'); ylabel('Fitness');

for r = 1:runs
    fprintf('Run %d / %d\n', r, runs);

    % =======================================================
    % CPGA
    % =======================================================
    Pop = genrpop(lpop, Space);
    cpgaEvol = zeros(1, numgen);
    for gen = 1:numgen
        Fit = eggholder(Pop);
        cpgaEvol(gen) = min(Fit);

        for i = 1:rows
            for j = 1:cols
                ind = (i-1)*cols + j;
                ip1 = mod(i, rows) + 1;
                jp1 = mod(j, cols) + 1;
                n1Ind = (ip1-1)*cols + j;
                n2Ind = (ip1-1)*cols + jp1;

                neighbors = [Pop(ind,:); Pop(n1Ind,:); Pop(n2Ind,:)];
                Work1 = crossov(neighbors,1,0);
                Work2 = mutx(neighbors,0.15,Space);
                Work1 = muta(Work1,0.15,Delta,Space);
                group = [Work1; Work2; neighbors];

                groupFit = eggholder(group);
                Best = selbest(group,groupFit,[1,0]);
                new = selrand(group,groupFit,2);

                Pop(ind,:) = Best(1,:);
                Pop(n1Ind,:) = new(1,:);
                Pop(n2Ind,:) = new(2,:);
            end
        end
    end
    allCPGA(r,:) = cpgaEvol;
    bestCPGA(r) = min(cpgaEvol);
    avgCPGA = avgCPGA + cpgaEvol;

    % =======================================================
    % PGA (island parallel GA)
    % =======================================================
    gcpobj = parpool(numIslands,"AutoAddClientPath",true,"SpmdEnabled",true);
    addAttachedFiles(gcpobj,"genetic");
    perIslandData = zeros(numIslands, numgen);

    spmd
        Pop = genrpop(perIsland, Space);
        for gen = 1:numgen
            Fit = eggholder(Pop);
            perIslandData(spmdIndex, gen) = min(Fit);

            Best=selbest(Pop,Fit,[3,0]);
            Old=selrand(Pop,Fit,17);
            Work1=selsus(Pop,Fit,10);
            Work2=selsus(Pop,Fit,20);
            Work1=crossov(Work1,1,0);
            Work2=mutx(Work2,0.2,Space);
            Work2=muta(Work2,0.2,Delta,Space);
            Pop=[Best;Old;Work1;Work2];

            if mod(gen,migInterval)==0
                migrants = selbest(Pop,Fit,[migSize,0]);
                nextIsland = mod(spmdIndex, numlabs) + 1;
                spmdSend(migrants, nextIsland);
                incoming = spmdReceive(mod(spmdIndex-2,numlabs)+1);
                [~, worstIdx] = sort(Fit,'descend');
                Pop(worstIdx(1:migSize),:) = incoming;
            end
        end
    end
    pData = cat(1, perIslandData{:});
    mergedPGA = min(pData, [], 1);
    allPGA(r,:) = mergedPGA;
    bestPGA(r) = min(mergedPGA);
    avgPGA = avgPGA + mergedPGA;
    delete(gcpobj);

    % =======================================================
    % SGA (single population)
    % =======================================================
    PopSGA = genrpop(lpop, Space);
    sgaEvol = zeros(1, numgen);
    for gen = 1:numgen
        FitSGA = eggholder(PopSGA);
        sgaEvol(gen) = min(FitSGA);

        Best = selbest(PopSGA,FitSGA,[5,0]);
        Old = selrand(PopSGA,FitSGA,lpop-30);
        Work1 = selsus(PopSGA,FitSGA,5);
        Work2 = selsus(PopSGA,FitSGA,20);
        Work1 = crossov(Work1,1,0);
        Work2 = mutx(Work2,0.15,Space);
        Work2 = muta(Work2,0.15,Delta,Space);
        PopSGA = [Best;Old;Work1;Work2];
        if size(PopSGA,1) > lpop
            PopSGA = PopSGA(1:lpop,:);
        elseif size(PopSGA,1) < lpop
            PopSGA = [PopSGA; genrpop(lpop-size(PopSGA,1),Space)];
        end
    end
    allSGA(r,:) = sgaEvol;
    bestSGA(r) = min(sgaEvol);
    avgSGA = avgSGA + sgaEvol;

    % volitelne: kresli jednotlivy beh (tencou ciarou)
    plot(cpgaEvol,'k','LineWidth',0.5);
    plot(mergedPGA,'k','LineWidth',0.5);
    plot(sgaEvol,'k','LineWidth',0.5);
end

% =======================================================
% Priemery a najlepsie behy
% =======================================================
avgCPGA = avgCPGA / runs;
avgPGA = avgPGA / runs;
avgSGA = avgSGA / runs;

[~, idxBestC] = min(bestCPGA);
[~, idxBestP] = min(bestPGA);
[~, idxBestS] = min(bestSGA);

hCavg = plot(avgCPGA,'Color',[0 0.6 1],'LineWidth',3);
hCmin = plot(allCPGA(idxBestC,:),'Color','g','LineWidth',1.8);
hPavg = plot(avgPGA,'Color',[0 1 1],'LineWidth',3);
hPmin = plot(allPGA(idxBestP,:),'Color',[0.5 0 0.5],'LineWidth',1.8);
hSavg = plot(avgSGA,'r','LineWidth',2.5);
hSmin = plot(allSGA(idxBestS,:),'Color',[0 0 0.7],'LineWidth',1.8);

legend([hCavg hCmin hPavg hPmin hSavg hSmin], ...
 {'Priemer CPGA','Najlepsi CPGA','Priemer PGA','Najlepsi PGA','Priemer SGA','Najlepsi SGA'}, ...
 'Location','northeastoutside');

fprintf('\nCPGA: mean(best)=%.6g, global best=%.6g\n', mean(bestCPGA), min(bestCPGA));
fprintf('PGA: mean(best)=%.6g, global best=%.6g\n', mean(bestPGA), min(bestPGA));
fprintf('SGA: mean(best)=%.6g, global best=%.6g\n', mean(bestSGA), min(bestSGA));
